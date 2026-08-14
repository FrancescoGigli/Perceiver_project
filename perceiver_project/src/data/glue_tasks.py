# src/data/glue_tasks.py
# Unified data loader for all GLUE benchmark tasks.
# Supports: CoLA, MRPC, STS-B, QQP, MNLI, QNLI, RTE
# Uses byte-level tokenization (UTF-8) consistent with Perceiver IO paper.

import os
import torch
import requests
import zipfile
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


# Task configurations
GLUE_TASKS = {
    'cola': {
        'url': 'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',
        'dir_name': 'CoLA',
        'train_file': 'train.tsv',
        'dev_file': 'dev.tsv',
        'sentence_cols': ['sentence'],  # single sentence
        'label_col': 'label',
        'label_type': 'int',       # 0/1
        'num_classes': 2,
        'skip_header': False,
        'tsv_columns': ['source', 'label', 'original_label', 'sentence'],
    },
    'mrpc': {
        'url': None,  # Facebook URL returns 403; use HuggingFace datasets instead
        'dir_name': 'MRPC',
        'train_file': 'train.tsv',
        'dev_file': 'dev.tsv',
        'sentence_cols': ['#1 String', '#2 String'],  # sentence pair
        'label_col': 'Quality',
        'label_type': 'int',
        'num_classes': 2,
        'skip_header': False,
        'tsv_columns': None,  # has header
        'hf_dataset': ('glue', 'mrpc'),  # HuggingFace dataset name
    },
    'stsb': {
        'url': 'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',
        'dir_name': 'STS-B',
        'train_file': 'train.tsv',
        'dev_file': 'dev.tsv',
        'sentence_cols': ['sentence1', 'sentence2'],
        'label_col': 'score',
        'label_type': 'float',    # regression 0-5
        'num_classes': 1,         # regression output
        'skip_header': False,
        'tsv_columns': None,
    },
    'qqp': {
        'url': 'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',
        'dir_name': 'QQP',
        'train_file': 'train.tsv',
        'dev_file': 'dev.tsv',
        'sentence_cols': ['question1', 'question2'],
        'label_col': 'is_duplicate',
        'label_type': 'int',
        'num_classes': 2,
        'skip_header': False,
        'tsv_columns': None,
    },
    'mnli': {
        'url': 'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
        'dir_name': 'MNLI',
        'train_file': 'train.tsv',
        'dev_file': 'dev_matched.tsv',
        'sentence_cols': ['sentence1', 'sentence2'],
        'label_col': 'gold_label',
        'label_type': 'str',      # entailment/contradiction/neutral
        'num_classes': 3,
        'skip_header': False,
        'tsv_columns': None,
        'label_map': {'entailment': 0, 'neutral': 1, 'contradiction': 2},
    },
    'qnli': {
        'url': 'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',
        'dir_name': 'QNLI',
        'train_file': 'train.tsv',
        'dev_file': 'dev.tsv',
        'sentence_cols': ['question', 'sentence'],
        'label_col': 'label',
        'label_type': 'str',
        'num_classes': 2,
        'skip_header': False,
        'tsv_columns': None,
        'label_map': {'entailment': 0, 'not_entailment': 1},
    },
    'rte': {
        'url': 'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
        'dir_name': 'RTE',
        'train_file': 'train.tsv',
        'dev_file': 'dev.tsv',
        'sentence_cols': ['sentence1', 'sentence2'],
        'label_col': 'label',
        'label_type': 'str',
        'num_classes': 2,
        'skip_header': False,
        'tsv_columns': None,
        'label_map': {'entailment': 0, 'not_entailment': 1},
    },
}

# Separator byte for sentence pairs (0xFF = 255, not used in standard UTF-8 text)
SEPARATOR_BYTE = 255


class GLUEDataset(Dataset):
    """
    Generic GLUE task dataset with byte-level tokenization.
    Supports single-sentence and sentence-pair tasks.
    """
    def __init__(self, data_path, task_name, split='train', seq_len=512):
        self.seq_len = seq_len
        self.task_config = GLUE_TASKS[task_name]
        self.task_name = task_name
        self.data = self._load_data(data_path, split)
    
    def _load_data(self, data_path, split):
        config = self.task_config
        
        if config.get('tsv_columns'):
            # CoLA doesn't have a header row
            df = pd.read_csv(data_path, sep='\t', header=None,
                             names=config['tsv_columns'],
                             quoting=3,  # QUOTE_NONE - handle malformed quotes
                             on_bad_lines='skip')
        else:
            df = pd.read_csv(data_path, sep='\t',
                             quoting=3,
                             on_bad_lines='skip')
        
        # For MNLI, filter out rows with '-' label (unlabeled)
        if self.task_name == 'mnli':
            df = df[df[config['label_col']].isin(['entailment', 'neutral', 'contradiction'])]
        
        # For string labels, map to integers
        if config['label_type'] == 'str' and 'label_map' in config:
            df = df[df[config['label_col']].isin(config['label_map'].keys())]
            df = df.copy()
            df[config['label_col']] = df[config['label_col']].map(config['label_map'])
        
        # Drop rows with NaN in sentence or label columns
        cols_to_check = config['sentence_cols'] + [config['label_col']]
        df = df.dropna(subset=cols_to_check)
        
        return df.reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        config = self.task_config
        
        # Get text: single sentence or sentence pair
        if len(config['sentence_cols']) == 1:
            text = str(row[config['sentence_cols'][0]])
        else:
            # Sentence pair: concatenate with separator byte
            s1 = str(row[config['sentence_cols'][0]])
            s2 = str(row[config['sentence_cols'][1]])
            text = s1 + chr(SEPARATOR_BYTE) + s2
        
        # Byte-level tokenization (UTF-8)
        bytes_list = list(text.encode('utf-8', errors='replace'))
        
        # Pad or truncate
        if len(bytes_list) > self.seq_len:
            bytes_list = bytes_list[:self.seq_len]
        else:
            bytes_list = bytes_list + [0] * (self.seq_len - len(bytes_list))
        
        # Get label
        label = row[config['label_col']]
        if config['label_type'] == 'float':
            label_tensor = torch.tensor(float(label), dtype=torch.float)
        else:
            label_tensor = torch.tensor(int(label), dtype=torch.long)
        
        return torch.tensor(bytes_list, dtype=torch.long), label_tensor


class GLUEPerceiverDataModule(pl.LightningDataModule):
    """
    LightningDataModule for all GLUE tasks.
    Handles download, preprocessing, and batching.
    """
    def __init__(self, task_name, data_dir, batch_size=64, num_workers=4, seq_len=512,
                 fourier_dim=64, max_frequencies=64, num_frequency_bands=6,
                 use_positional_encoding=True):
        super().__init__()
        assert task_name in GLUE_TASKS, f"Unknown GLUE task: {task_name}. Available: {list(GLUE_TASKS.keys())}"
        
        self.task_name = task_name
        self.task_config = GLUE_TASKS[task_name]
        self.data_dir = data_dir
        self.task_dir = os.path.join(data_dir, self.task_config['dir_name'])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len
        
        # Positional Encoding settings
        self.fourier_dim = fourier_dim
        self.max_frequencies = max_frequencies
        self.num_frequency_bands = num_frequency_bands
        self.use_positional_encoding = use_positional_encoding
        self.input_dim = 257 + (self.fourier_dim * 2 + 1) if use_positional_encoding else 257
        
        # Task-specific properties
        self.num_classes = self.task_config['num_classes']
        self.is_regression = (self.task_config['label_type'] == 'float')
    
    def _has_data_files(self):
        """Check if the task directory has the expected train and dev files."""
        train_path = os.path.join(self.task_dir, self.task_config['train_file'])
        dev_path = os.path.join(self.task_dir, self.task_config['dev_file'])
        return os.path.exists(train_path) and os.path.exists(dev_path)

    def _download_from_hf(self):
        """Download MRPC (or other tasks) from HuggingFace datasets library."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace 'datasets' library is required for MRPC download. "
                "Install with: pip install datasets"
            )
        
        hf_name, hf_config = self.task_config['hf_dataset']
        print(f"Downloading {self.task_name.upper()} from HuggingFace datasets ({hf_name}/{hf_config})...")
        ds = load_dataset(hf_name, hf_config)
        
        os.makedirs(self.task_dir, exist_ok=True)
        
        # Save train split
        train_path = os.path.join(self.task_dir, self.task_config['train_file'])
        train_df = pd.DataFrame({
            'Quality': ds['train']['label'],
            '#1 ID': range(len(ds['train'])),
            '#2 ID': range(len(ds['train'])),
            '#1 String': ds['train']['sentence1'],
            '#2 String': ds['train']['sentence2'],
        })
        train_df.to_csv(train_path, sep='\t', index=False)
        
        # Save validation split
        dev_path = os.path.join(self.task_dir, self.task_config['dev_file'])
        dev_df = pd.DataFrame({
            'Quality': ds['validation']['label'],
            '#1 ID': range(len(ds['validation'])),
            '#2 ID': range(len(ds['validation'])),
            '#1 String': ds['validation']['sentence1'],
            '#2 String': ds['validation']['sentence2'],
        })
        dev_df.to_csv(dev_path, sep='\t', index=False)
        
        print(f"MRPC: saved {len(train_df)} train, {len(dev_df)} dev samples to {self.task_dir}")

    def download_data(self):
        if self._has_data_files():
            return  # Already downloaded
        
        # Use HuggingFace datasets for tasks with no direct URL (e.g. MRPC)
        if self.task_config.get('hf_dataset'):
            self._download_from_hf()
            return
        
        url = self.task_config.get('url')
        if not url:
            raise ValueError(f"No download URL or HuggingFace dataset configured for {self.task_name}")
        
        os.makedirs(self.task_dir, exist_ok=True)
        print(f"Downloading {self.task_name.upper()} dataset from {url}...")
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            zip_path = os.path.join(self.data_dir, f"{self.task_config['dir_name']}.zip")
            with open(zip_path, 'wb') as f:
                f.write(r.content)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            os.remove(zip_path)
            print(f"{self.task_name.upper()} dataset downloaded and extracted to {self.task_dir}")
        except Exception as e:
            print(f"Error downloading {self.task_name.upper()}: {e}")
            print(f"Please manually download from {url} and extract to {self.task_dir}")
            raise
    
    def setup(self, stage=None):
        self.download_data()
        
        train_path = os.path.join(self.task_dir, self.task_config['train_file'])
        dev_path = os.path.join(self.task_dir, self.task_config['dev_file'])
        
        self.train_dataset = GLUEDataset(train_path, self.task_name, split='train', seq_len=self.seq_len)
        self.val_dataset = GLUEDataset(dev_path, self.task_name, split='dev', seq_len=self.seq_len)
        
        print(f"GLUE {self.task_name.upper()}: {len(self.train_dataset)} train, {len(self.val_dataset)} dev samples")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
    
    def preprocess_batch(self, batch):
        """
        Preprocesses a batch for the Perceiver model.
        Same byte-level one-hot + Fourier PE as SST-2.
        """
        inputs, labels = batch
        
        B, L = inputs.shape
        vocab_size = 257  # 256 bytes + 1 for mask token (compatibility with MLM)
        
        # One-hot encoding
        inputs_one_hot = torch.nn.functional.one_hot(inputs, num_classes=vocab_size).float()
        
        if self.use_positional_encoding:
            pos = torch.linspace(-1, 1, L, device=inputs.device)
            pos = pos.view(1, L, 1).expand(B, L, 1)
            
            encodings = [pos]
            full_freqs = torch.logspace(0, np.log10(self.max_frequencies), self.fourier_dim,
                                        base=10, device=inputs.device)
            
            for freq in full_freqs:
                encodings.append(torch.sin(pos * freq * np.pi))
                encodings.append(torch.cos(pos * freq * np.pi))
            
            pos_features = torch.cat(encodings, dim=-1)
            model_input = torch.cat([inputs_one_hot, pos_features], dim=-1)
        else:
            model_input = inputs_one_hot
        
        return {'inputs': model_input, 'labels': labels}
