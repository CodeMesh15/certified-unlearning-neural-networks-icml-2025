"""
Graph Dataset Loaders for Certified Unlearning
Supports popular graph benchmark datasets
"""

import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
import pickle
import os
from scipy.sparse import coo_matrix
import requests
import gzip
from pathlib import Path


class GraphDataset(Dataset):
    """Base class for graph datasets"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.node_features = None
        self.adj_matrix = None
        self.labels = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Returns (node_features, adj_matrix, label) for a single node"""
        return (
            jnp.array(self.node_features[idx]),
            jnp.array(self.adj_matrix[idx]),
            jnp.array(self.labels[idx])
        )
    
    def get_full_graph(self) -> Dict:
        """Returns the complete graph data"""
        return {
            'features': jnp.array(self.node_features),
            'adj': jnp.array(self.adj_matrix),
            'labels': jnp.array(self.labels),
            'train_mask': jnp.array(self.train_mask),
            'val_mask': jnp.array(self.val_mask),
            'test_mask': jnp.array(self.test_mask),
        }
    
    @staticmethod
    def normalize_adj(adj: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
        """
        Normalize adjacency matrix: D^{-1/2} A D^{-1/2}
        
        Args:
            adj: Adjacency matrix [num_nodes, num_nodes]
            add_self_loops: Whether to add self-loops
        
        Returns:
            Normalized adjacency matrix
        """
        if add_self_loops:
            adj = adj + np.eye(adj.shape[0])
        
        # Degree matrix
        degree = np.sum(adj, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
        
        # D^{-1/2}
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        
        # D^{-1/2} A D^{-1/2}
        adj_normalized = D_inv_sqrt @ adj @ D_inv_sqrt
        
        return adj_normalized


class CoraDataset(GraphDataset):
    """
    Cora citation network dataset
    - 2708 scientific publications (nodes)
    - 5429 citation links (edges)
    - 7 classes
    - 1433 features (bag-of-words)
    """
    
    URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    
    def __init__(self, data_dir: str = "./data", normalize: bool = True):
        super().__init__(data_dir)
        self.normalize = normalize
        self._load_data()
    
    def _download(self):
        """Download Cora dataset"""
        import tarfile
        import urllib.request
        
        tar_path = self.data_dir / "cora.tgz"
        
        if not tar_path.exists():
            print(f"Downloading Cora dataset to {tar_path}...")
            urllib.request.urlretrieve(self.URL, tar_path)
            
            # Extract
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(self.data_dir)
    
    def _load_data(self):
        """Load Cora dataset"""
        self._download()
        
        cora_dir = self.data_dir / "cora"
        content_file = cora_dir / "cora.content"
        cites_file = cora_dir / "cora.cites"
        
        # Load content (features and labels)
        content = np.genfromtxt(content_file, dtype=str)
        
        # Paper IDs
        paper_ids = content[:, 0]
        id_to_idx = {pid: idx for idx, pid in enumerate(paper_ids)}
        
        # Features
        features = content[:, 1:-1].astype(float)
        
        # Labels
        label_names = content[:, -1]
        unique_labels = np.unique(label_names)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        labels = np.array([label_to_idx[label] for label in label_names])
        
        # Load citations (edges)
        cites = np.genfromtxt(cites_file, dtype=str)
        
        # Build adjacency matrix
        num_nodes = len(paper_ids)
        adj = np.zeros((num_nodes, num_nodes), dtype=float)
        
        for cite in cites:
            if cite[0] in id_to_idx and cite[1] in id_to_idx:
                src_idx = id_to_idx[cite[0]]
                dst_idx = id_to_idx[cite[1]]
                adj[dst_idx, src_idx] = 1.0  # Directed: cited -> citing
        
        # Make undirected
        adj = adj + adj.T
        adj = (adj > 0).astype(float)
        
        # Normalize adjacency
        if self.normalize:
            adj = self.normalize_adj(adj)
        
        # Create train/val/test splits (standard split: 140/500/1000)
        num_train = 140
        num_val = 500
        num_test = 1000
        
        indices = np.random.permutation(num_nodes)
        train_idx = indices[:num_train]
        val_idx = indices[num_train:num_train + num_val]
        test_idx = indices[num_train + num_val:num_train + num_val + num_test]
        
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        self.node_features = features
        self.adj_matrix = adj
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_classes = len(unique_labels)
        self.num_features = features.shape[1]


class CiteseerDataset(GraphDataset):
    """
    Citeseer citation network dataset
    - 3312 scientific publications (nodes)
    - 4732 citation links (edges)
    - 6 classes
    - 3703 features
    """
    
    URL = "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz"
    
    def __init__(self, data_dir: str = "./data", normalize: bool = True):
        super().__init__(data_dir)
        self.normalize = normalize
        self._load_data()
    
    def _download(self):
        """Download Citeseer dataset"""
        import tarfile
        import urllib.request
        
        tar_path = self.data_dir / "citeseer.tgz"
        
        if not tar_path.exists():
            print(f"Downloading Citeseer dataset to {tar_path}...")
            urllib.request.urlretrieve(self.URL, tar_path)
            
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(self.data_dir)
    
    def _load_data(self):
        """Load Citeseer dataset (similar structure to Cora)"""
        self._download()
        
        citeseer_dir = self.data_dir / "citeseer"
        content_file = citeseer_dir / "citeseer.content"
        cites_file = citeseer_dir / "citeseer.cites"
        
        # Load content
        content = np.genfromtxt(content_file, dtype=str)
        
        paper_ids = content[:, 0]
        id_to_idx = {pid: idx for idx, pid in enumerate(paper_ids)}
        
        features = content[:, 1:-1].astype(float)
        
        label_names = content[:, -1]
        unique_labels = np.unique(label_names)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        labels = np.array([label_to_idx[label] for label in label_names])
        
        # Load citations
        cites = np.genfromtxt(cites_file, dtype=str)
        
        num_nodes = len(paper_ids)
        adj = np.zeros((num_nodes, num_nodes), dtype=float)
        
        for cite in cites:
            if cite[0] in id_to_idx and cite[1] in id_to_idx:
                src_idx = id_to_idx[cite[0]]
                dst_idx = id_to_idx[cite[1]]
                adj[dst_idx, src_idx] = 1.0
        
        adj = adj + adj.T
        adj = (adj > 0).astype(float)
        
        if self.normalize:
            adj = self.normalize_adj(adj)
        
        # Splits
        num_train = 120
        num_val = 500
        num_test = 1000
        
        indices = np.random.permutation(num_nodes)
        train_idx = indices[:num_train]
        val_idx = indices[num_train:num_train + num_val]
        test_idx = indices[num_train + num_val:num_train + num_val + num_test]
        
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        self.node_features = features
        self.adj_matrix = adj
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_classes = len(unique_labels)
        self.num_features = features.shape[1]


class PubmedDataset(GraphDataset):
    """
    Pubmed diabetes citation network
    - 19717 scientific publications (nodes)
    - 44338 citation links (edges)
    - 3 classes
    - 500 features
    """
    
    URL = "https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz"
    
    def __init__(self, data_dir: str = "./data", normalize: bool = True):
        super().__init__(data_dir)
        self.normalize = normalize
        self._load_data()
    
    def _download(self):
        """Download Pubmed dataset"""
        import tarfile
        import urllib.request
        
        tar_path = self.data_dir / "pubmed.tgz"
        
        if not tar_path.exists():
            print(f"Downloading Pubmed dataset to {tar_path}...")
            urllib.request.urlretrieve(self.URL, tar_path)
            
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(self.data_dir)
    
    def _load_data(self):
        """Load Pubmed dataset"""
        self._download()
        
        pubmed_dir = self.data_dir / "Pubmed-Diabetes"
        
        # Pubmed has different file format
        # This is simplified - actual implementation may vary
        node_file = pubmed_dir / "data/Pubmed-Diabetes.NODE.paper.tab"
        edge_file = pubmed_dir / "data/Pubmed-Diabetes.DIRECTED.cites.tab"
        
        # Load nodes
        with open(node_file, 'r') as f:
            lines = f.readlines()[2:]  # Skip headers
        
        paper_ids = []
        features_list = []
        labels_list = []
        
        for line in lines:
            parts = line.strip().split('\t')
            paper_id = parts[0]
            label = int(parts[1].split('=')[1]) - 1  # 0-indexed
            features = [float(x.split('=')[1]) for x in parts[2:-1]]
            
            paper_ids.append(paper_id)
            features_list.append(features)
            labels_list.append(label)
        
        id_to_idx = {pid: idx for idx, pid in enumerate(paper_ids)}
        features = np.array(features_list, dtype=float)
        labels = np.array(labels_list)
        
        # Load edges
        with open(edge_file, 'r') as f:
            lines = f.readlines()[2:]
        
        num_nodes = len(paper_ids)
        adj = np.zeros((num_nodes, num_nodes), dtype=float)
        
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                src = parts[1].split(':')[1]
                dst = parts[3].split(':')[1]
                
                if src in id_to_idx and dst in id_to_idx:
                    adj[id_to_idx[dst], id_to_idx[src]] = 1.0
        
        adj = adj + adj.T
        adj = (adj > 0).astype(float)
        
        if self.normalize:
            adj = self.normalize_adj(adj)
        
        # Splits
        num_train = 60
        num_val = 500
        num_test = 1000
        
        indices = np.random.permutation(num_nodes)
        train_idx = indices[:num_train]
        val_idx = indices[num_train:num_train + num_val]
        test_idx = indices[num_train + num_val:num_train + num_val + num_test]
        
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        self.node_features = features
        self.adj_matrix = adj
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_classes = 3
        self.num_features = features.shape[1]


class GraphDataModule:
    """Data module for graph datasets compatible with certified unlearning framework"""
    
    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "./data",
        forget_ratio: float = 0.1,
        batch_size: int = 32,
        seed: int = 42,
    ):
        """
        Args:
            dataset_name: One of 'cora', 'citeseer', 'pubmed'
            data_dir: Directory to store data
            forget_ratio: Ratio of training data to forget
            batch_size: Batch size (for mini-batch training if needed)
            seed: Random seed
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.forget_ratio = forget_ratio
        self.batch_size = batch_size
        self.seed = seed
        
        np.random.seed(seed)
        
        # Load dataset
        if self.dataset_name == "cora":
            self.dataset = CoraDataset(data_dir)
        elif self.dataset_name == "citeseer":
            self.dataset = CiteseerDataset(data_dir)
        elif self.dataset_name == "pubmed":
            self.dataset = PubmedDataset(data_dir)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        self._create_splits()
    
    def _create_splits(self):
        """Create train/forget/retain/val/test splits"""
        graph_data = self.dataset.get_full_graph()
        
        train_indices = np.where(graph_data['train_mask'])[0]
        
        # Split train into forget and retain
        num_forget = int(len(train_indices) * self.forget_ratio)
        forget_indices = np.random.choice(train_indices, num_forget, replace=False)
        retain_indices = np.setdiff1d(train_indices, forget_indices)
        
        # Create masks
        self.forget_mask = np.zeros_like(graph_data['train_mask'])
        self.forget_mask[forget_indices] = True
        
        self.retain_mask = np.zeros_like(graph_data['train_mask'])
        self.retain_mask[retain_indices] = True
        
        print(f"Dataset: {self.dataset_name}")
        print(f"Total nodes: {len(graph_data['labels'])}")
        print(f"Train nodes: {len(train_indices)}")
        print(f"Forget nodes: {len(forget_indices)}")
        print(f"Retain nodes: {len(retain_indices)}")
        print(f"Val nodes: {np.sum(graph_data['val_mask'])}")
        print(f"Test nodes: {np.sum(graph_data['test_mask'])}")
    
    def get_dataloaders(self):
        """
        Returns dataloaders compatible with the trainer
        For transductive learning, we use full graph batches
        """
        graph_data = self.dataset.get_full_graph()
        
        # For GNNs on transductive tasks, we typically use full graph
        # But return splits as separate "dataloaders"
        class FullGraphDataLoader:
            def __init__(self, features, adj, labels, mask):
                self.features = features
                self.adj = adj
                self.labels = labels
                self.mask = mask
                self.dataset = self  # For compatibility
            
            def __iter__(self):
                # Return full graph with mask
                masked_labels = jnp.where(
                    self.mask,
                    self.labels,
                    -1  # Ignore index
                )
                yield (self.features, self.adj), masked_labels
            
            def __len__(self):
                return 1  # Single batch (full graph)
        
        train_loader = FullGraphDataLoader(
            graph_data['features'],
            graph_data['adj'],
            graph_data['labels'],
            graph_data['train_mask']
        )
        
        forget_loader = FullGraphDataLoader(
            graph_data['features'],
            graph_data['adj'],
            graph_data['labels'],
            self.forget_mask
        )
        
        retain_loader = FullGraphDataLoader(
            graph_data['features'],
            graph_data['adj'],
            graph_data['labels'],
            self.retain_mask
        )
        
        val_loader = FullGraphDataLoader(
            graph_data['features'],
            graph_data['adj'],
            graph_data['labels'],
            graph_data['val_mask']
        )
        
        test_loader = FullGraphDataLoader(
            graph_data['features'],
            graph_data['adj'],
            graph_data['labels'],
            graph_data['test_mask']
        )
        
        return {
            'train': train_loader,
            'forget': forget_loader,
            'retain': retain_loader,
            'val': val_loader,
            'test': test_loader,
            'num_classes': self.dataset.num_classes,
            'num_features': self.dataset.num_features,
        }
