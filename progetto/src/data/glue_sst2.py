import os
import torch
import requests
import zipfile
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from ..utils.positional_encoding import FourierPositionalEncoding

class SST2Dataset(Dataset):
    def __init__(self, data_path, split='train', seq_len=512):
        self.seq_len = seq_len
        self.data = self._load_data(data_path, split)
        
    def _load_data(self, data_path, split):
        df = pd.read_csv(data_path, sep='\t')
        return df
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['sentence']
        label = row['label']
        
        # Byte-level tokenization (UTF-8)
        bytes_list = list(text.encode('utf-8'))
        
        # Pad or truncate
        if len(bytes_list) > self.seq_len:
            bytes_list = bytes_list[:self.seq_len]
        else:
            bytes_list = bytes_list + [0] * (self.seq_len - len(bytes_list))
            
        # Add special token for CLS? 
        # For simplicity, we just use the raw bytes. Perceiver IO usually appends a CLS token or uses a learnable query.
        # Here we will rely on the model's query mechanism (Perceiver IO uses specific output query for class).
        
        return torch.tensor(bytes_list, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class SST2PerceiverDataModule:
    def __init__(self, data_dir, batch_size=64, num_workers=4, seq_len=512, 
                 fourier_dim=64, max_frequencies=64, num_frequency_bands=6,
                 use_positional_encoding=True):
        self.data_dir = data_dir
        self.sst2_dir = os.path.join(data_dir, 'SST-2')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len
        
        # Positional Encoding settings
        self.fourier_dim = fourier_dim
        self.max_frequencies = max_frequencies
        self.num_frequency_bands = num_frequency_bands
        self.use_positional_encoding = use_positional_encoding

        # Stesso positional encoding del pre-training MLM (src/data/wikitext103.py):
        # stessa classe, stesse bande lineari, stesse coordinate in [0, 1]. Prima qui
        # c'era un Fourier scritto a mano con frequenze logaritmiche su [-1, 1]: il
        # segnale posizionale era diverso e il primo cross-attention del checkpoint
        # (input_dim 270 vs 386) non si trasferiva -- il fine-tuning ripartiva da pesi
        # casuali proprio nel modulo che legge il testo.
        self.pos_encodings = None
        self.input_dim = 257
        if self.use_positional_encoding:
            self._setup_pos_encoding()
            self.input_dim += self.pos_encoding.out_dim

    def _setup_pos_encoding(self):
        self.pos_encoding = FourierPositionalEncoding(
            num_bands=self.num_frequency_bands,
            max_freq=self.max_frequencies,
            num_pos_feats=1,
        )
        positions = torch.arange(self.seq_len).float() / max(1, (self.seq_len - 1))
        with torch.no_grad():
            self.pos_encodings = self.pos_encoding(positions.unsqueeze(-1))

    def download_data(self):
        if not os.path.exists(self.sst2_dir):
            os.makedirs(self.sst2_dir)
            print("Downloading SST-2 dataset...")
            url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
            r = requests.get(url)
            zip_path = os.path.join(self.data_dir, "SST-2.zip")
            with open(zip_path, 'wb') as f:
                f.write(r.content)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Clean up
            os.remove(zip_path)
            print("SST-2 dataset downloaded and extracted.")

    def setup(self, stage=None):
        self.download_data()
        self.train_dataset = SST2Dataset(os.path.join(self.sst2_dir, 'train.tsv'), split='train', seq_len=self.seq_len)
        self.val_dataset = SST2Dataset(os.path.join(self.sst2_dir, 'dev.tsv'), split='dev', seq_len=self.seq_len) # GLUE 'dev' is used as validation

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def preprocess_batch(self, batch):
        """
        Preprocesses a batch for the model.
        Returns a dictionary with 'inputs' and 'labels'
        """
        inputs, labels = batch
        
        # Perceiver expects inputs in a specific format.
        # For text, we can pass the byte indices directly if the model handles embedding,
        # OR we can one-hot encode them here.
        # The current Perceiver implementation (src/perceiver/perceiver.py) expects:
        # inputs: [Batch, N, C]
        
        # We need to one-hot encode the bytes: [Batch, SeqLen] -> [Batch, SeqLen, 256]
        # And concatenate positional encoding.
        
        B, L = inputs.shape
        vocab_size = 257 # Match MLM vocab size (256 + 1 for mask) for weight compatibility
        
        # One-hot encoding
        inputs_one_hot = torch.nn.functional.one_hot(inputs, num_classes=vocab_size).float() # [B, L, 257]
        
        if self.use_positional_encoding:
            pos_features = self.pos_encodings[:L].to(inputs.device)      # [L, out_dim]
            pos_features = pos_features.unsqueeze(0).expand(B, -1, -1)   # [B, L, out_dim]
            model_input = torch.cat([inputs_one_hot, pos_features], dim=-1)
        else:
            model_input = inputs_one_hot
            
        return {'inputs': model_input, 'labels': labels}

