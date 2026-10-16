import numpy as np
import torch
import torch.utils.data as data
from pathlib import Path


class EmbDataset(data.Dataset):

    def __init__(self, data_path, standardize=False):

        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.embeddings = np.load(data_path)
        mapping_path = Path(str(data_path) + ".item_ids.npy")
        self.item_ids = np.load(mapping_path) if mapping_path.exists() else np.arange(len(self.embeddings))
        if (self.item_ids.ndim != 1 or len(self.item_ids) != len(self.embeddings)
                or not np.issubdtype(self.item_ids.dtype, np.integer)
                or len(np.unique(self.item_ids)) != len(self.item_ids) or (self.item_ids < 0).any()):
            raise ValueError("Invalid embedding row-to-item ID mapping")
        if (self.embeddings.ndim != 2 or not len(self.embeddings) or not self.embeddings.shape[1]
                or not np.isfinite(self.embeddings).all() or (np.linalg.norm(self.embeddings, axis=1) == 0).any()):
            raise ValueError("SID inputs must contain finite product vectors; do not replace missing embeddings with zeros")
        
        if standardize:
            embeddings = self.embeddings.astype(np.float32)
            mean = embeddings.mean(axis=0)
            std = embeddings.std(axis=0)
            self.embeddings = (embeddings - mean) / np.maximum(std, 1e-5)
            
        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        print(f"Embeddings stats - min: {self.embeddings.min():.6f}, max: {self.embeddings.max():.6f}, mean: {self.embeddings.mean():.6f}")
        
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
