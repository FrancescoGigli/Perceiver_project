# src/data/wikitext103.py
# WikiText-103 byte-level MLM datamodule for Perceiver-IO.

import os
import io
import tarfile
import zipfile
import urllib.request
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..utils.positional_encoding import FourierPositionalEncoding

WIKITEXT103_URL = "https://s3.amazonaws.com/fast-ai-nlp/wikitext-103.tgz"
WIKITEXT103_URL_MIRROR = "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-v1.zip" # Fallback if needed, but FastAI is reliable


class WikiText103ByteDataset(Dataset):
    def __init__(self, byte_tensor: torch.Tensor, seq_len: int, mask_prob: float, mask_token_id: int):
        self.byte_tensor = byte_tensor
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.num_chunks = byte_tensor.numel() // seq_len

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        chunk = self.byte_tensor[start:end]

        labels = chunk.clone()
        mask = torch.rand(self.seq_len) < self.mask_prob
        input_ids = chunk.clone()
        input_ids[mask] = self.mask_token_id

        return input_ids, labels, mask


class WikiText103PerceiverDataModule:
    def __init__(
        self,
        data_dir="./data",
        batch_size=16,
        num_workers=2,
        seq_len=2048,
        mask_prob=0.15,
        fourier_dim=64,
        max_frequencies=64.0,
        num_frequency_bands=6,
        wikitext103_zip_path=None,
        use_positional_encoding=True,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.fourier_dim = fourier_dim
        self.max_frequencies = max_frequencies
        self.num_frequency_bands = num_frequency_bands
        self.wikitext103_zip_path = wikitext103_zip_path
        self.use_positional_encoding = use_positional_encoding

        self.vocab_size = 256
        self.mask_token_id = 256

        if self.use_positional_encoding:
            self._setup_pos_encoding()
        else:
            self.pos_encodings = None

        self.input_dim = self.vocab_size + 1
        if self.use_positional_encoding:
            self.input_dim += self.fourier_dim

    def _setup_pos_encoding(self):
        self.pos_encoding = FourierPositionalEncoding(
            dim=self.fourier_dim,
            max_frequencies=self.max_frequencies,
            num_frequency_bands=self.num_frequency_bands,
            num_pos_feats=1,
            circular=True,
        )

        positions = torch.arange(self.seq_len).float() / max(1, (self.seq_len - 1))
        coords = positions.unsqueeze(-1)
        with torch.no_grad():
            self.pos_encodings = self.pos_encoding(coords)

    def _download_wikitext103(self):
        target_dir = os.path.join(self.data_dir, "wikitext-103")
        os.makedirs(target_dir, exist_ok=True)

        # FastAI tgz extracts to 'wikitext-103' folder
        train_path = os.path.join(target_dir, "wikitext-103", "wiki.train.tokens")
        valid_path = os.path.join(target_dir, "wikitext-103", "wiki.valid.tokens")

        # Fallback for FastAI CSV format
        train_csv_path = os.path.join(target_dir, "wikitext-103", "train.csv")
        test_csv_path = os.path.join(target_dir, "wikitext-103", "test.csv")
        
        # Check if files already exist
        if (os.path.exists(train_path) and os.path.exists(valid_path)) or \
           (os.path.exists(train_csv_path) and os.path.exists(test_csv_path)):
            return

        archive_path = os.path.join(target_dir, "wikitext-103.tgz")
        if self.wikitext103_zip_path:
            archive_path = self.wikitext103_zip_path
        
        if not os.path.exists(archive_path):
            print(f"Downloading WikiText-103 from {WIKITEXT103_URL}...")
            try:
                urllib.request.urlretrieve(WIKITEXT103_URL, archive_path)
            except Exception as e:
                print(f"Download failed: {e}")
                if WIKITEXT103_URL_MIRROR:
                     print("Trying mirror...")
                     archive_path = os.path.join(target_dir, "wikitext-103-v1.zip")
                     urllib.request.urlretrieve(WIKITEXT103_URL_MIRROR, archive_path)

        print("Extracting WikiText-103...")
        if archive_path.endswith('.tgz') or archive_path.endswith('.tar.gz'):
             with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(target_dir)
        elif archive_path.endswith('.zip'):
             with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(target_dir)

    @staticmethod
    def _load_bytes(file_path: str) -> torch.Tensor:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find file: {file_path}")
        print(f"Loading bytes from {file_path}...")
        # Check if it's a CSV and maybe cleaner read needed? 
        # For byte-level MLM, extra quotes/commas are just more bytes.
        with io.open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        byte_data = text.encode("utf-8")
        return torch.tensor(list(byte_data), dtype=torch.long)

    def setup(self):
        self._download_wikitext103()
        # The zip extracts a folder named 'wikitext-103'
        base_dir = os.path.join(self.data_dir, "wikitext-103", "wikitext-103")
        train_path = os.path.join(base_dir, "wiki.train.tokens")
        valid_path = os.path.join(base_dir, "wiki.valid.tokens")

        # Fallback to CSV if tokens not found
        if not os.path.exists(train_path) and os.path.exists(os.path.join(base_dir, "train.csv")):
            print("Using train.csv/test.csv format")
            train_path = os.path.join(base_dir, "train.csv")
            # Use test.csv as valid if valid.csv not present
            valid_path = os.path.join(base_dir, "test.csv") 

        train_bytes = self._load_bytes(train_path)
        valid_bytes = self._load_bytes(valid_path)
        
        print(f"WikiText-103 loaded. Train bytes: {len(train_bytes):,}, Valid bytes: {len(valid_bytes):,}")

        self.train_dataset = WikiText103ByteDataset(
            train_bytes, self.seq_len, self.mask_prob, self.mask_token_id
        )
        self.val_dataset = WikiText103ByteDataset(
            valid_bytes, self.seq_len, self.mask_prob, self.mask_token_id
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def preprocess_batch(self, batch):
        input_ids, labels, mask = batch
        one_hot = F.one_hot(input_ids, num_classes=self.vocab_size + 1).float()

        if self.use_positional_encoding:
            pos_enc = self.pos_encodings.unsqueeze(0).expand(one_hot.size(0), -1, -1)
            inputs = torch.cat([one_hot, pos_enc.to(one_hot.device)], dim=-1)
        else:
            inputs = one_hot

        return {
            "inputs": inputs,
            "labels": labels,
            "mask": mask,
        }
