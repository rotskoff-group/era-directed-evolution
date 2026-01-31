# era-directed-evolution
Official repository for "Efficient, Few-shot Directed Evolution with Energy Rank Alignment".

## Environment setup (uv)
Create and sync the environment, then install the project in editable mode:

```bash
uv sync
uv pip install -e .
```

## Commands
All training/inference scripts use Hydra configs in `pera/scripts/cfgs/`.

### Training
Train a transformer model:
```bash
pera_train
```

Override config values with Hydra-style overrides, for example:
```bash
pera_train train.trainer_args.max_epochs=100 train.trainer_args.devices=1
```

### Inference (sampling)
Sample sequences from a trained model:
```bash
pera_sample
```

Example override:
```bash
pera_sample infer.target=GB1 infer.num_samples=512 infer.network_filename=/path/to/checkpoint.pt
```

### Full landscape probability distribution
Compute per-sequence log-probabilities for a dataset:
```bash
pera_compute_landscape
```

Example override:
```bash
pera_compute_landscape compute_landscape.data=GB1 compute_landscape.data_root_path=./data
```

## Data location
Place dataset files under the repository `data/` subdirectory. The configs read from:
```
data/<dataset>/scale2max/<dataset>.csv
data/<dataset>/<dataset>.fasta
```

If your data lives elsewhere, set `infer.data_root_path` or
`compute_landscape.data_root_path` in the Hydra overrides or config files.
