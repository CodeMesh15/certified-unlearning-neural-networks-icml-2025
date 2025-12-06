"""
Graph Neural Network Models for Certified Unlearning
Implements various GNN architectures compatible with the certified unlearning framework
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Optional, Tuple
import numpy as np


class GCNLayer(nn.Module):
    """Graph Convolutional Network Layer (Kipf & Welling, 2017)"""
    features: int
    activation: Callable = nn.relu
    use_bias: bool = True
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, adj: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Normalized adjacency matrix [num_nodes, num_nodes]
            train: Training mode flag
        Returns:
            Updated node features [num_nodes, features]
        """
        # Linear transformation
        x = nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.glorot_uniform(),
        )(x)
        
        # Graph convolution: X' = AXW
        x = jnp.matmul(adj, x)
        
        # Apply activation
        if self.activation is not None:
            x = self.activation(x)
        
        # Dropout
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        
        return x


class GATLayer(nn.Module):
    """Graph Attention Network Layer (Veličković et al., 2018)"""
    features: int
    num_heads: int = 1
    concat_heads: bool = True
    activation: Callable = nn.elu
    dropout_rate: float = 0.0
    attention_dropout: float = 0.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, adj: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Binary adjacency matrix [num_nodes, num_nodes]
            train: Training mode flag
        Returns:
            Updated node features [num_nodes, features * num_heads] if concat else [num_nodes, features]
        """
        num_nodes = x.shape[0]
        
        # Multi-head attention
        outputs = []
        for _ in range(self.num_heads):
            # Linear transformation
            h = nn.Dense(features=self.features)(x)  # [num_nodes, features]
            
            # Compute attention coefficients
            # a^T [Wh_i || Wh_j] for all edges (i,j)
            a_left = nn.Dense(features=1, use_bias=False)(h)  # [num_nodes, 1]
            a_right = nn.Dense(features=1, use_bias=False)(h)  # [num_nodes, 1]
            
            # Broadcasting: [num_nodes, num_nodes]
            e = a_left + a_right.T
            e = nn.leaky_relu(e, negative_slope=0.2)
            
            # Mask attention to existing edges only
            e = jnp.where(adj > 0, e, -1e9)
            
            # Softmax to get attention weights
            alpha = nn.softmax(e, axis=1)
            
            # Apply attention dropout
            if self.attention_dropout > 0:
                alpha = nn.Dropout(rate=self.attention_dropout, deterministic=not train)(alpha)
            
            # Aggregate neighborhood features
            h_prime = jnp.matmul(alpha, h)
            outputs.append(h_prime)
        
        # Combine heads
        if self.concat_heads:
            output = jnp.concatenate(outputs, axis=-1)
        else:
            output = jnp.mean(jnp.stack(outputs, axis=0), axis=0)
        
        # Apply activation
        if self.activation is not None:
            output = self.activation(output)
        
        # Dropout
        if self.dropout_rate > 0:
            output = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(output)
        
        return output


class GINLayer(nn.Module):
    """Graph Isomorphism Network Layer (Xu et al., 2019)"""
    features: int
    epsilon: float = 0.0
    learn_epsilon: bool = False
    activation: Callable = nn.relu
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, adj: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            train: Training mode flag
        Returns:
            Updated node features [num_nodes, features]
        """
        # Learnable epsilon or fixed
        if self.learn_epsilon:
            eps = self.param('epsilon', nn.initializers.constant(self.epsilon), (1,))
        else:
            eps = self.epsilon
        
        # Aggregate neighbors
        neighbor_sum = jnp.matmul(adj, x)
        
        # Add self-loop with epsilon
        h = (1 + eps) * x + neighbor_sum
        
        # MLP: two-layer with activation
        h = nn.Dense(features=self.features)(h)
        if self.activation is not None:
            h = self.activation(h)
        
        h = nn.Dense(features=self.features)(h)
        if self.activation is not None:
            h = self.activation(h)
        
        # Dropout
        if self.dropout_rate > 0:
            h = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(h)
        
        return h


class GraphSAGELayer(nn.Module):
    """GraphSAGE Layer (Hamilton et al., 2017)"""
    features: int
    aggregator: str = 'mean'  # 'mean', 'pool', 'lstm'
    activation: Callable = nn.relu
    normalize: bool = True
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, adj: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            train: Training mode flag
        Returns:
            Updated node features [num_nodes, features]
        """
        # Aggregate neighborhood
        if self.aggregator == 'mean':
            # Mean aggregation
            degree = jnp.sum(adj, axis=1, keepdims=True) + 1e-6
            neighbor_agg = jnp.matmul(adj, x) / degree
        elif self.aggregator == 'pool':
            # Max pooling aggregation
            # First transform then pool
            x_transformed = nn.Dense(features=x.shape[-1])(x)
            x_transformed = self.activation(x_transformed) if self.activation else x_transformed
            # Mask non-neighbors
            masked = jnp.where(adj[:, :, None] > 0, x_transformed[None, :, :], -jnp.inf)
            neighbor_agg = jnp.max(masked, axis=1)
            neighbor_agg = jnp.where(jnp.isfinite(neighbor_agg), neighbor_agg, 0.0)
        else:
            raise ValueError(f"Aggregator {self.aggregator} not supported")
        
        # Concatenate self and neighborhood
        h = jnp.concatenate([x, neighbor_agg], axis=-1)
        
        # Transform
        h = nn.Dense(features=self.features)(h)
        
        # Normalize
        if self.normalize:
            h = h / (jnp.linalg.norm(h, axis=1, keepdims=True) + 1e-6)
        
        # Activation
        if self.activation is not None:
            h = self.activation(h)
        
        # Dropout
        if self.dropout_rate > 0:
            h = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(h)
        
        return h


class GCN(nn.Module):
    """Multi-layer Graph Convolutional Network"""
    hidden_dims: Tuple[int, ...] = (64, 32)
    num_classes: int = 10
    dropout_rate: float = 0.5
    activation: Callable = nn.relu
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, adj: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Normalized adjacency matrix [num_nodes, num_nodes]
            train: Training mode flag
        Returns:
            Logits [num_nodes, num_classes]
        """
        # Stack GCN layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = GCNLayer(
                features=hidden_dim,
                activation=self.activation,
                dropout_rate=self.dropout_rate if i < len(self.hidden_dims) - 1 else 0,
            )(x, adj, train=train)
        
        # Output layer
        x = GCNLayer(
            features=self.num_classes,
            activation=None,
            dropout_rate=0.0,
        )(x, adj, train=train)
        
        return x


class GAT(nn.Module):
    """Multi-layer Graph Attention Network"""
    hidden_dims: Tuple[int, ...] = (64, 32)
    num_classes: int = 10
    num_heads: Tuple[int, ...] = (8, 1)
    dropout_rate: float = 0.6
    attention_dropout: float = 0.6
    activation: Callable = nn.elu
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, adj: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Binary adjacency matrix [num_nodes, num_nodes]
            train: Training mode flag
        Returns:
            Logits [num_nodes, num_classes]
        """
        # Stack GAT layers
        for i, (hidden_dim, n_heads) in enumerate(zip(self.hidden_dims, self.num_heads)):
            x = GATLayer(
                features=hidden_dim,
                num_heads=n_heads,
                concat_heads=(i < len(self.hidden_dims) - 1),
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                attention_dropout=self.attention_dropout,
            )(x, adj, train=train)
        
        # Output layer
        x = GATLayer(
            features=self.num_classes,
            num_heads=1,
            concat_heads=False,
            activation=None,
            dropout_rate=0.0,
            attention_dropout=0.0,
        )(x, adj, train=train)
        
        return x


class GIN(nn.Module):
    """Multi-layer Graph Isomorphism Network"""
    hidden_dims: Tuple[int, ...] = (64, 64, 32)
    num_classes: int = 10
    dropout_rate: float = 0.5
    epsilon: float = 0.0
    learn_epsilon: bool = False
    pooling: str = 'sum'  # 'sum', 'mean', 'max' for graph-level tasks
    activation: Callable = nn.relu
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, adj: jnp.ndarray, 
                 batch_idx: Optional[jnp.ndarray] = None,
                 train: bool = False) -> jnp.ndarray:
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            batch_idx: Batch indices for graph-level prediction [num_nodes]
            train: Training mode flag
        Returns:
            Logits [num_nodes, num_classes] for node-level or [num_graphs, num_classes] for graph-level
        """
        # Stack GIN layers
        for hidden_dim in self.hidden_dims:
            x = GINLayer(
                features=hidden_dim,
                epsilon=self.epsilon,
                learn_epsilon=self.learn_epsilon,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
            )(x, adj, train=train)
        
        # Graph-level pooling if batch_idx provided
        if batch_idx is not None:
            num_graphs = int(jnp.max(batch_idx)) + 1
            if self.pooling == 'sum':
                x = jax.ops.segment_sum(x, batch_idx, num_graphs)
            elif self.pooling == 'mean':
                x = jax.ops.segment_sum(x, batch_idx, num_graphs)
                counts = jax.ops.segment_sum(jnp.ones((x.shape[0], 1)), batch_idx, num_graphs)
                x = x / (counts + 1e-6)
            elif self.pooling == 'max':
                x = jax.ops.segment_max(x, batch_idx, num_graphs)
        
        # Output layer
        x = nn.Dense(features=self.num_classes)(x)
        
        return x


class GraphSAGE(nn.Module):
    """Multi-layer GraphSAGE"""
    hidden_dims: Tuple[int, ...] = (64, 32)
    num_classes: int = 10
    aggregator: str = 'mean'
    dropout_rate: float = 0.5
    activation: Callable = nn.relu
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, adj: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            train: Training mode flag
        Returns:
            Logits [num_nodes, num_classes]
        """
        # Stack GraphSAGE layers
        for hidden_dim in self.hidden_dims:
            x = GraphSAGELayer(
                features=hidden_dim,
                aggregator=self.aggregator,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
            )(x, adj, train=train)
        
        # Output layer
        x = nn.Dense(features=self.num_classes)(x)
        
        return x


# Model factory for GNN models
class GNNModelFactory:
    @staticmethod
    def create_model(model_name: str, num_classes: int, **kwargs):
        """
        Create GNN model by name
        
        Args:
            model_name: One of 'gcn', 'gat', 'gin', 'graphsage'
            num_classes: Number of output classes
            **kwargs: Additional model-specific arguments
        
        Returns:
            GNN model instance
        """
        if model_name == "gcn":
            return GCN(
                num_classes=num_classes,
                hidden_dims=kwargs.get('hidden_dims', (64, 32)),
                dropout_rate=kwargs.get('dropout_rate', 0.5),
            )
        elif model_name == "gat":
            return GAT(
                num_classes=num_classes,
                hidden_dims=kwargs.get('hidden_dims', (64, 32)),
                num_heads=kwargs.get('num_heads', (8, 1)),
                dropout_rate=kwargs.get('dropout_rate', 0.6),
                attention_dropout=kwargs.get('attention_dropout', 0.6),
            )
        elif model_name == "gin":
            return GIN(
                num_classes=num_classes,
                hidden_dims=kwargs.get('hidden_dims', (64, 64, 32)),
                dropout_rate=kwargs.get('dropout_rate', 0.5),
                epsilon=kwargs.get('epsilon', 0.0),
                learn_epsilon=kwargs.get('learn_epsilon', False),
                pooling=kwargs.get('pooling', 'sum'),
            )
        elif model_name == "graphsage":
            return GraphSAGE(
                num_classes=num_classes,
                hidden_dims=kwargs.get('hidden_dims', (64, 32)),
                aggregator=kwargs.get('aggregator', 'mean'),
                dropout_rate=kwargs.get('dropout_rate', 0.5),
            )
        else:
            raise ValueError(f"Model {model_name} not supported. Choose from: gcn, gat, gin, graphsage")
