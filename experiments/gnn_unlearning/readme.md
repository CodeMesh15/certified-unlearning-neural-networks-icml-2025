# GNN Certified Unlearning Experiments

This directory contains configurations and experiments for certified unlearning on Graph Neural Networks (GNNs).

## Experiment Configurations

We provide three pre-configured experiments:

1. **GCN on Cora** (`config_gcn_cora.yaml`)
   - Graph Convolutional Network
   - Citation network with 2708 papers, 7 classes
   - Uses PABI (Privacy Amplification by Iteration)

2. **GAT on Citeseer** (`config_gat_citeseer.yaml`)
   - Graph Attention Network
   - Citation network with 3312 papers, 6 classes
   - Uses Contractive Coefficients method

3. **GIN on Pubmed** (`config_gin_pubmed.yaml`)
   - Graph Isomorphism Network
   - Biomedical citation network with 19717 papers, 3 classes
   - Uses PABI with decaying privacy allocation

## Running Experiments

### Single Experiment

```bash
python experiment.py --config experiments/gnn_unlearning/config_gcn_cora.yaml
```

This runs all experiments in the directory using 4 GPUs with 1 job per GPU.

## Adaptive Privacy Mechanisms

The experiments support multiple adaptive privacy allocation strategies:

- **uniform**: Equal budget allocation across all steps
- **decaying**: More privacy early, less later (decay_rate < 1)
- **increasing**: Less privacy early, more later (growth_rate > 1)
- **loss_adaptive**: Allocate based on training loss magnitude
- **grad_norm_adaptive**: Allocate based on gradient norms

Configure in the `adaptive_privacy` section of the YAML file.

## Key Hyperparameters

### Model Architecture
- `hidden_dims`: Hidden layer dimensions
- `dropout_rate`: Dropout probability
- `num_heads`: (GAT only) Number of attention heads

### Unlearning
- `algorithm`: Unlearning method (iteration, contractive_coefficients, dp-baseline, retrain)
- `init_model_clip`: Initial model clipping threshold
- `init_sigma`: Initial noise standard deviation
- `delta`: Privacy parameter delta
- `grad_clip`: Gradient clipping threshold

### Privacy Budget
- `epsilon_renyi_target`: Target Rényi epsilon (for PABI)
- `epsilon_target`: Target epsilon (for other methods)
- `delta`: Target delta

## Expected Results

| Model | Dataset | Method | Accuracy (Train) | Accuracy (Forget) | Accuracy (Test) | Privacy (ε, δ) |
|-------|---------|--------|------------------|-------------------|-----------------|----------------|
| GCN | Cora | PABI | ~81% | ~20% | ~79% | (1.0, 1e-5) |
| GAT | Citeseer | Contractive | ~72% | ~15% | ~70% | (1.0, 1e-5) |
| GIN | Pubmed | PABI | ~79% | ~33% | ~77% | (0.5, 1e-6) |

*Note: Results may vary based on random seed and hyperparameters*

## Extending to New Datasets

To add a new graph dataset:

1. Implement dataset class in `src/data/graph_dataset.py`
2. Add to `GraphDataModule`
3. Create configuration YAML with appropriate parameters
4. Run experiments



