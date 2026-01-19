# Schedule-Aware Spatiotemporal Graph Attention Network (SA-STGAT)
## Implementation Plan for NYC Subway Headway Prediction

---

## Executive Summary

This document outlines the transition from our current ConvLSTM architecture to a **Graph Attention Network (GAT)** specifically designed for the NYC subway's branching topology. The key insight: ConvLSTM treats the subway as a 2D image, but it's actually a **directed graph** with branches and merges.

### Why the Switch?

| Issue | ConvLSTM (Current) | SA-STGAT (Proposed) |
|-------|-------------------|---------------------|
| **Topology** | Treats line as 2D grid (75×2) | Respects actual graph structure |
| **Branches** | Artificial padding for Y-shape | Natural node connectivity |
| **Attention** | Fixed convolution weights | Dynamic attention per neighbor |
| **Schedule** | Concatenated as channel | Injected via Transform Attention |

---

## 1. Core Architectural Concepts

### 1.1 Graph Construction

Instead of distance bins arranged in a grid, we model stations as **nodes** and track segments as **edges**:

```
         A02 (Inwood)
           ↓
         A05
           ↓
          ...
           ↓
         A55 (Rockaway Blvd) ← BRANCH POINT
          /  \
         ↓    ↓
   A57→A65   H01→H04 (Broad Channel) ← SHUTTLE JUNCTION
  (Lefferts)    /  \
               ↓    ↓
            H06→H11  H12→H15
         (Far Rock) (Rock Park)
```

**Adjacency Matrix**: Binary matrix where `A[i,j] = 1` if trains can travel from node `i` to node `j`.

### 1.2 Why Graph Attention?

Standard GNNs use **fixed** edge weights (e.g., distance-based). GAT computes **dynamic** attention:

```
α_ij = softmax(LeakyReLU(a^T [W*h_i || W*h_j]))
```

**Translation**: "How much should node `i` pay attention to neighbor `j` right now?"

For subway: If upstream station has a stalled train → high attention (blocking signal).
If upstream is clear → low attention (irrelevant).

### 1.3 Encoder-Decoder with Schedule Injection

```
┌─────────────────────────────────────────────────────────────┐
│                     SA-STGAT Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ENCODER                          DECODER                   │
│  ────────                         ───────                   │
│  Historical Headways (30 min)     Future Schedule (15 min)  │
│         ↓                              ↓                    │
│  ┌──────────────┐               ┌──────────────┐           │
│  │ Spatial GAT  │               │ Terminal     │           │
│  │ (neighbors)  │               │ Headways     │           │
│  └──────────────┘               │ (known)      │           │
│         ↓                       └──────────────┘           │
│  ┌──────────────┐                     ↓                    │
│  │ Temporal     │               ┌──────────────┐           │
│  │ Attention    │──────────────→│ TRANSFORM    │           │
│  │ (time steps) │   Context     │ ATTENTION    │           │
│  └──────────────┘               └──────────────┘           │
│         ↓                             ↓                    │
│    Context Matrix              ┌──────────────┐            │
│                                │ Spatial GAT  │            │
│                                │ (propagate)  │            │
│                                └──────────────┘            │
│                                       ↓                    │
│                                 Predictions                │
│                               (all stations)               │
└─────────────────────────────────────────────────────────────┘
```

**Transform Attention**: The decoder asks "Given the dispatcher's plan at terminals (Query), what historical patterns (Key/Value) are relevant?"

---

## 2. Data Representation Changes

### 2.1 Current Format (ConvLSTM)

```python
# Grid-based tensors
X_history = (batch, 30, 75, 2, 1)  # 75 bins × 2 directions
Y_target  = (batch, 15, 75, 2, 1)
Schedule  = (batch, 15, 4, 1)      # 4 terminals
```

### 2.2 New Format (SA-STGAT)

```python
# Graph-based tensors
X_history = (batch, 30, N_nodes, C_features)  # N_nodes ≈ 44 stations
Y_target  = (batch, 15, N_nodes, 1)           # Headway per station
Schedule  = (batch, 15, N_nodes, 1)           # Sparse: only terminals filled

# Graph structure
A = (N_nodes, N_nodes)  # Adjacency matrix (sparse)
edge_index = (2, E)     # PyG format: [source_nodes, target_nodes]
```

### 2.3 Node Features

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `headway` | Time since last train | MinMax to [0,1] |
| `scheduled_headway` | Static schedule value | MinMax to [0,1] |
| `deviation` | headway - scheduled | Clip to [-1, 1] |
| `is_terminal` | Binary flag | 0 or 1 |
| `is_branch_point` | Binary flag | 0 or 1 |

---

## 3. Implementation Plan

### Phase 1: Graph Construction (Day 1)

**Goal**: Build the A-line adjacency matrix from GTFS data.

```python
# src/data/graph.py

import numpy as np
import pandas as pd

class SubwayGraph:
    """Build graph structure from GTFS stops and routes."""
    
    # A-line stations in order (simplified)
    A_LINE_STATIONS = [
        # Trunk (Inwood → Rockaway Blvd)
        'A02', 'A05', 'A09', 'A10', 'A11', 'A12', 'A14', 'A15',
        'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A24',
        'A25', 'A27', 'A28', 'A30', 'A31', 'A32', 'A33', 'A34',
        'A36', 'A38', 'A40', 'A41', 'A42', 'A43', 'A44', 'A45',
        'A46', 'A47', 'A48', 'A50', 'A51', 'A52', 'A53', 'A54', 'A55',
        # Lefferts branch
        'A57', 'A59', 'A60', 'A61', 'A63', 'A64', 'A65',
        # Rockaway branch (via H01-H04 connector)
        'H01', 'H02', 'H03', 'H04',
        # Far Rockaway
        'H06', 'H07', 'H08', 'H09', 'H10', 'H11',
        # Rockaway Park
        'H12', 'H13', 'H14', 'H15',
    ]
    
    TERMINAL_STATIONS = ['A02', 'A65', 'H11', 'H15']
    
    # Branch/merge points
    BRANCH_POINTS = {
        'A55': ['A57', 'H01'],  # Rockaway Blvd → Lefferts OR Rockaway
        'H04': ['H06', 'H12'],  # Broad Channel → Far Rock OR Rock Park
    }
    
    def __init__(self):
        self.stations = self.A_LINE_STATIONS
        self.n_nodes = len(self.stations)
        self.station_to_idx = {s: i for i, s in enumerate(self.stations)}
        self.idx_to_station = {i: s for i, s in enumerate(self.stations)}
        
    def build_adjacency_matrix(self, direction='southbound'):
        """
        Build directed adjacency matrix.
        
        For southbound: edges go Inwood → terminals
        For northbound: edges go terminals → Inwood
        """
        A = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        
        if direction == 'southbound':
            # Trunk: A02 → A55
            for i in range(40):  # 0 to 39 (A02 to A54)
                A[i, i+1] = 1
            
            # Branch at A55 (idx 40)
            A[40, 41] = 1  # A55 → A57 (Lefferts)
            A[40, 48] = 1  # A55 → H01 (Rockaway connector)
            
            # Lefferts: A57 → A65
            for i in range(41, 47):
                A[i, i+1] = 1
            
            # Connector: H01 → H04
            for i in range(48, 51):
                A[i, i+1] = 1
            
            # Branch at H04 (idx 51)
            A[51, 52] = 1  # H04 → H06 (Far Rockaway)
            A[51, 58] = 1  # H04 → H12 (Rockaway Park)
            
            # Far Rockaway: H06 → H11
            for i in range(52, 57):
                A[i, i+1] = 1
            
            # Rockaway Park: H12 → H15
            for i in range(58, 61):
                A[i, i+1] = 1
                
        else:  # northbound - reverse all edges
            A_south = self.build_adjacency_matrix('southbound')
            A = A_south.T
            
        # Add self-loops
        A = A + np.eye(self.n_nodes)
        
        return A
    
    def get_edge_index(self, direction='southbound'):
        """Convert to PyTorch Geometric edge_index format."""
        A = self.build_adjacency_matrix(direction)
        src, dst = np.where(A > 0)
        return np.stack([src, dst], axis=0)
    
    def get_terminal_mask(self):
        """Binary mask: 1 for terminal stations."""
        mask = np.zeros(self.n_nodes, dtype=np.float32)
        for term in self.TERMINAL_STATIONS:
            mask[self.station_to_idx[term]] = 1
        return mask


# Test
if __name__ == "__main__":
    graph = SubwayGraph()
    A = graph.build_adjacency_matrix('southbound')
    print(f"Nodes: {graph.n_nodes}")
    print(f"Edges: {A.sum() - graph.n_nodes}")  # Exclude self-loops
    print(f"Terminals: {graph.TERMINAL_STATIONS}")
```

### Phase 2: GAT Model (Day 2)

**Goal**: Build the SA-STGAT model using Spektral (Keras) or PyTorch Geometric.

#### Option A: Spektral (Keras - matches current stack)

```python
# src/models/stgat.py

import tensorflow as tf
from tensorflow import keras
from keras import layers
from spektral.layers import GATConv

class SpatialGATBlock(layers.Layer):
    """Graph Attention for spatial aggregation."""
    
    def __init__(self, channels, heads=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gat = GATConv(
            channels, 
            attn_heads=heads,
            concat_heads=True,
            dropout_rate=dropout,
            activation='elu'
        )
        self.proj = layers.Dense(channels)
        self.norm = layers.LayerNormalization()
        
    def call(self, inputs, training=False):
        x, adj = inputs
        # x: (batch, nodes, features)
        # adj: (nodes, nodes) adjacency
        
        h = self.gat([x, adj])
        h = self.proj(h)
        return self.norm(x + h)  # Residual


class TemporalAttentionBlock(layers.Layer):
    """Self-attention across time steps."""
    
    def __init__(self, d_model, heads=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=heads,
            key_dim=d_model // heads,
            dropout=dropout
        )
        self.ffn = keras.Sequential([
            layers.Dense(d_model * 4, activation='gelu'),
            layers.Dense(d_model)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        
    def call(self, x, training=False):
        # x: (batch, time, nodes, features)
        batch, time, nodes, feat = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Reshape for temporal attention: (batch*nodes, time, features)
        x_flat = tf.reshape(x, (-1, time, feat))
        
        # Self-attention over time
        attn = self.mha(x_flat, x_flat, training=training)
        x_flat = self.norm1(x_flat + attn)
        
        # FFN
        ffn_out = self.ffn(x_flat)
        x_flat = self.norm2(x_flat + ffn_out)
        
        return tf.reshape(x_flat, (batch, time, nodes, feat))


class TransformAttention(layers.Layer):
    """Bridge encoder (history) and decoder (future schedule)."""
    
    def __init__(self, d_model, heads=4, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=heads,
            key_dim=d_model // heads
        )
        self.norm = layers.LayerNormalization()
        
    def call(self, query, key_value, training=False):
        """
        query: decoder input (batch, future_steps, nodes, d)
        key_value: encoder output (batch, history_steps, nodes, d)
        """
        b, fT, n, d = tf.shape(query)[0], tf.shape(query)[1], tf.shape(query)[2], tf.shape(query)[3]
        hT = tf.shape(key_value)[1]
        
        # Flatten spatial dim for cross-attention
        q = tf.reshape(query, (b, fT * n, d))
        kv = tf.reshape(key_value, (b, hT * n, d))
        
        attn = self.mha(q, kv, training=training)
        out = self.norm(q + attn)
        
        return tf.reshape(out, (b, fT, n, d))


class SASTGAT(keras.Model):
    """Schedule-Aware Spatiotemporal Graph Attention Network."""
    
    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 2,
        lookback: int = 30,
        horizon: int = 15,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.n_nodes = n_nodes
        self.lookback = lookback
        self.horizon = horizon
        
        # Embeddings
        self.node_embedding = layers.Embedding(n_nodes, d_model)
        self.time_proj = layers.Dense(d_model)
        self.input_proj = layers.Dense(d_model)
        
        # Encoder
        self.encoder_spatial = [
            SpatialGATBlock(d_model, n_heads, dropout) 
            for _ in range(n_encoder_layers)
        ]
        self.encoder_temporal = [
            TemporalAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_encoder_layers)
        ]
        
        # Transform Attention (history → future)
        self.transform_attn = TransformAttention(d_model, n_heads)
        
        # Decoder
        self.decoder_spatial = [
            SpatialGATBlock(d_model, n_heads, dropout)
            for _ in range(n_decoder_layers)
        ]
        
        # Output
        self.output_proj = layers.Dense(1)
        
    def call(self, inputs, training=False):
        """
        inputs:
            history: (batch, lookback, nodes, features)
            schedule: (batch, horizon, nodes, 1)  # Terminal headways
            adjacency: (nodes, nodes)
        """
        history, schedule, adj = inputs['history'], inputs['schedule'], inputs['adj']
        
        batch_size = tf.shape(history)[0]
        
        # === ENCODER ===
        # Project input features
        h = self.input_proj(history)  # (batch, lookback, nodes, d_model)
        
        # Add node embeddings
        node_ids = tf.range(self.n_nodes)
        node_emb = self.node_embedding(node_ids)  # (nodes, d_model)
        h = h + node_emb[None, None, :, :]
        
        # Encoder layers (alternate spatial/temporal)
        for spatial, temporal in zip(self.encoder_spatial, self.encoder_temporal):
            # Spatial: process each timestep
            h_list = []
            for t in range(self.lookback):
                h_t = spatial([h[:, t], adj], training=training)
                h_list.append(h_t)
            h = tf.stack(h_list, axis=1)
            
            # Temporal: process across time
            h = temporal(h, training=training)
        
        encoder_output = h  # (batch, lookback, nodes, d_model)
        
        # === DECODER ===
        # Initialize decoder with schedule (sparse: only terminals have values)
        d = self.input_proj(schedule)  # (batch, horizon, nodes, d_model)
        d = d + node_emb[None, None, :, :]
        
        # Transform attention: link history to future
        d = self.transform_attn(d, encoder_output, training=training)
        
        # Decoder spatial layers (propagate terminal info through graph)
        for spatial in self.decoder_spatial:
            d_list = []
            for t in range(self.horizon):
                d_t = spatial([d[:, t], adj], training=training)
                d_list.append(d_t)
            d = tf.stack(d_list, axis=1)
        
        # Output projection
        output = self.output_proj(d)  # (batch, horizon, nodes, 1)
        
        return output


def build_stgat_model(config):
    """Factory function matching existing interface."""
    from src.data.graph import SubwayGraph
    
    graph = SubwayGraph()
    
    model = SASTGAT(
        n_nodes=graph.n_nodes,
        n_features=config.get('n_features', 3),
        d_model=config.get('d_model', 64),
        n_heads=config.get('n_heads', 4),
        n_encoder_layers=config.get('n_encoder_layers', 3),
        n_decoder_layers=config.get('n_decoder_layers', 2),
        lookback=config.get('lookback', 30),
        horizon=config.get('horizon', 15),
        dropout=config.get('dropout', 0.1),
    )
    
    return model, graph.build_adjacency_matrix()
```

#### Option B: PyTorch Geometric (More mature GNN ecosystem)

```python
# src/models/stgat_pyg.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class SpatialGAT(nn.Module):
    """GAT layer for spatial message passing."""
    
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.gat = GATv2Conv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x, edge_index):
        # x: (batch * nodes, features)
        h = self.gat(x, edge_index)
        return self.norm(x + h)


class TemporalTransformer(nn.Module):
    """Transformer for temporal modeling."""
    
    def __init__(self, d_model, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (batch, time, d_model)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class SASTGAT_PyG(nn.Module):
    """PyTorch Geometric implementation."""
    
    def __init__(self, n_nodes, in_features, d_model=64, heads=4, 
                 n_layers=3, lookback=30, horizon=15):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.lookback = lookback
        self.horizon = horizon
        
        self.input_proj = nn.Linear(in_features, d_model)
        self.node_emb = nn.Embedding(n_nodes, d_model)
        
        # Encoder
        self.enc_spatial = nn.ModuleList([
            SpatialGAT(d_model, d_model, heads) for _ in range(n_layers)
        ])
        self.enc_temporal = nn.ModuleList([
            TemporalTransformer(d_model, heads) for _ in range(n_layers)
        ])
        
        # Cross-attention (history → future)
        self.cross_attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        
        # Decoder
        self.dec_spatial = nn.ModuleList([
            SpatialGAT(d_model, d_model, heads) for _ in range(2)
        ])
        
        self.output = nn.Linear(d_model, 1)
        
    def forward(self, x_hist, x_sched, edge_index):
        """
        x_hist: (batch, lookback, nodes, features)
        x_sched: (batch, horizon, nodes, 1)
        edge_index: (2, edges)
        """
        B, L, N, _ = x_hist.shape
        H = self.horizon
        
        # Project and add embeddings
        h = self.input_proj(x_hist)
        node_emb = self.node_emb(torch.arange(N, device=x_hist.device))
        h = h + node_emb.unsqueeze(0).unsqueeze(0)
        
        # Encoder
        for spatial, temporal in zip(self.enc_spatial, self.enc_temporal):
            # Spatial per timestep
            h_new = []
            for t in range(L):
                h_t = h[:, t].reshape(B * N, -1)
                h_t = spatial(h_t, edge_index)
                h_new.append(h_t.reshape(B, N, -1))
            h = torch.stack(h_new, dim=1)
            
            # Temporal per node
            h = h.permute(0, 2, 1, 3).reshape(B * N, L, -1)
            h = temporal(h)
            h = h.reshape(B, N, L, -1).permute(0, 2, 1, 3)
        
        enc_out = h  # (B, L, N, d)
        
        # Decoder input from schedule
        d = self.input_proj(x_sched) + node_emb.unsqueeze(0).unsqueeze(0)
        
        # Cross-attention
        d_flat = d.reshape(B, H * N, -1)
        enc_flat = enc_out.reshape(B, L * N, -1)
        d_flat, _ = self.cross_attn(d_flat, enc_flat, enc_flat)
        d = d_flat.reshape(B, H, N, -1)
        
        # Decoder spatial
        for spatial in self.dec_spatial:
            d_new = []
            for t in range(H):
                d_t = d[:, t].reshape(B * N, -1)
                d_t = spatial(d_t, edge_index)
                d_new.append(d_t.reshape(B, N, -1))
            d = torch.stack(d_new, dim=1)
        
        return self.output(d)
```

### Phase 3: Dataset Adapter (Day 3)

**Goal**: Modify dataset to output graph format instead of grid format.

```python
# src/data/graph_dataset.py

import numpy as np
import tensorflow as tf
from src.data.graph import SubwayGraph

class GraphDatasetBuilder:
    """Converts station-level headway data to graph format."""
    
    def __init__(self, config):
        self.graph = SubwayGraph()
        self.lookback = config.get('lookback', 30)
        self.horizon = config.get('horizon', 15)
        
    def load_station_headways(self, arrivals_file):
        """
        Convert raw arrivals to per-station headway time series.
        
        Returns: dict {station_id: np.array of headways at each minute}
        """
        import pandas as pd
        
        arrivals = pd.read_csv(arrivals_file)
        arrivals['arrival_time'] = pd.to_datetime(arrivals['arrival_time'])
        
        # Get time range
        t_min = arrivals['arrival_time'].min().floor('min')
        t_max = arrivals['arrival_time'].max().ceil('min')
        n_minutes = int((t_max - t_min).total_seconds() / 60)
        
        # Time index
        time_index = pd.date_range(t_min, periods=n_minutes, freq='1min')
        
        # Per-station headway series
        station_headways = {}
        
        for station in self.graph.stations:
            # Filter arrivals at this station
            sta_arrivals = arrivals[
                arrivals['stop_id'].str.startswith(station)
            ].copy()
            
            if len(sta_arrivals) == 0:
                station_headways[station] = np.zeros(n_minutes)
                continue
            
            sta_arrivals = sta_arrivals.sort_values('arrival_time')
            sta_arrivals['headway'] = sta_arrivals['arrival_time'].diff().dt.total_seconds() / 60
            
            # Bin into minutes
            sta_arrivals['minute'] = sta_arrivals['arrival_time'].dt.floor('min')
            binned = sta_arrivals.groupby('minute')['headway'].last()
            
            # Align to full index
            headways = binned.reindex(time_index).ffill().fillna(0).values
            station_headways[station] = np.clip(headways, 0, 30)
            
        return station_headways, time_index
    
    def build_node_features(self, station_headways):
        """
        Stack station headways into node feature matrix.
        
        Returns: np.array (T, N_nodes, C_features)
        """
        T = len(list(station_headways.values())[0])
        N = self.graph.n_nodes
        
        # Feature: headway only for now
        X = np.zeros((T, N, 1), dtype=np.float32)
        
        for station, headways in station_headways.items():
            if station in self.graph.station_to_idx:
                idx = self.graph.station_to_idx[station]
                X[:, idx, 0] = headways
        
        return X
    
    def build_schedule_tensor(self, planned_schedule_file):
        """
        Load planned schedule and format for graph.
        
        planned_schedule: (T, 4, 1) → (T, N_nodes, 1) sparse
        """
        planned = np.load(planned_schedule_file)
        T = planned.shape[0]
        N = self.graph.n_nodes
        
        schedule = np.zeros((T, N, 1), dtype=np.float32)
        
        terminal_mapping = {
            0: self.graph.station_to_idx['A02'],   # Inwood
            1: self.graph.station_to_idx['A65'],   # Lefferts
            2: self.graph.station_to_idx['H11'],   # Far Rockaway
            3: self.graph.station_to_idx['H15'],   # Rockaway Park
        }
        
        for ch, node_idx in terminal_mapping.items():
            schedule[:, node_idx, 0] = planned[:, ch, 0]
        
        return schedule
    
    def create_tf_dataset(self, X, schedule, batch_size=64, shuffle=True):
        """Create TensorFlow dataset for training."""
        
        T, N, C = X.shape
        lookback = self.lookback
        horizon = self.horizon
        window = lookback + horizon
        
        adj = self.graph.build_adjacency_matrix()
        
        def generator():
            indices = list(range(T - window))
            if shuffle:
                np.random.shuffle(indices)
            
            for i in indices:
                x_hist = X[i : i + lookback]           # (30, N, C)
                x_sched = schedule[i + lookback : i + window]  # (15, N, 1)
                y = X[i + lookback : i + window, :, 0:1]  # (15, N, 1)
                
                yield {
                    'history': x_hist,
                    'schedule': x_sched,
                    'adj': adj
                }, y
        
        output_sig = (
            {
                'history': tf.TensorSpec((lookback, N, C), tf.float32),
                'schedule': tf.TensorSpec((horizon, N, 1), tf.float32),
                'adj': tf.TensorSpec((N, N), tf.float32),
            },
            tf.TensorSpec((horizon, N, 1), tf.float32)
        )
        
        ds = tf.data.Dataset.from_generator(generator, output_signature=output_sig)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return ds
```

### Phase 4: Training Integration (Day 4)

**Goal**: Wire into existing training infrastructure.

```python
# src/experiments/run_stgat.py

from src.models.stgat import build_stgat_model
from src.data.graph_dataset import GraphDatasetBuilder
from src.training.trainer import train_model
from src.config import Config

def run_stgat_experiment():
    config = Config()
    
    # Build graph dataset
    builder = GraphDatasetBuilder(config)
    station_headways, time_index = builder.load_station_headways(
        'data/nyc_subway_a_line_arrivals_2025.csv'
    )
    X = builder.build_node_features(station_headways)
    schedule = builder.build_schedule_tensor('data/schedule_matrix_4terminal_planned.npy')
    
    # Split
    T = X.shape[0]
    train_end = int(T * 0.7)
    val_end = int(T * 0.85)
    
    train_ds = builder.create_tf_dataset(X[:train_end], schedule[:train_end])
    val_ds = builder.create_tf_dataset(X[train_end:val_end], schedule[train_end:val_end], shuffle=False)
    
    # Build model
    model, adj = build_stgat_model({
        'n_features': X.shape[-1],
        'd_model': 64,
        'n_heads': 4,
        'n_encoder_layers': 3,
        'n_decoder_layers': 2,
    })
    
    # Train
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4, clipnorm=1.0),
        loss='huber',
        metrics=['mae']
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        ]
    )
    
    return model
```

---

## 4. Key Differences from ConvLSTM

| Aspect | ConvLSTM | SA-STGAT |
|--------|----------|----------|
| **Spatial** | 2D Conv on 75×2 grid | GAT on 44-node graph |
| **Temporal** | LSTM hidden states | Transformer attention |
| **Branches** | Padded into grid | Natural graph edges |
| **Schedule** | Concatenated channel | Transform attention injection |
| **Attention** | None | Multi-head (spatial + temporal) |
| **Interpretability** | Low | High (attention weights) |

---

## 5. Expected Outcomes

### Best Case
- Break through 155s plateau to ~120-140s RMSE
- Attention weights reveal which upstream stations matter most
- Model learns branch-specific propagation patterns

### Realistic Case
- Modest improvement (145-155s) but better interpretability
- Cleaner handling of branch points
- Foundation for adding more features (dwell time, passenger load)

### Worst Case
- Similar performance (~155s) confirming data/noise ceiling
- But: cleaner architecture for future extensions

---

## 6. Dependencies

```bash
# Option A: Spektral (Keras)
pip install spektral

# Option B: PyTorch Geometric
pip install torch torch-geometric torch-scatter torch-sparse
```

---

## 7. Timeline

| Day | Task |
|-----|------|
| 1 | Graph construction from GTFS |
| 2 | SA-STGAT model implementation |
| 3 | Dataset adapter (arrivals → graph tensors) |
| 4 | Training integration |
| 5 | First training run + debugging |
| 6 | Hyperparameter tuning |
| 7 | Evaluation + comparison with ConvLSTM |

---

## References

1. Original paper concept (ConvLSTM baseline)
2. Graph Attention Networks (Veličković et al., 2018)
3. GMAN: Graph Multi-Attention Network (Zheng et al., 2020)
4. DCRNN: Diffusion Convolutional RNN (Li et al., 2018)
5. Spektral: Graph Neural Networks in Keras
6. PyTorch Geometric documentation
