# era-directed-evolution
Official repository for "Efficient, Few-shot Directed Evolution with Energy Rank Alignment". All necessary datasets and files from experiments in the paper can be found on our Huggingface Model and Datasets repositories.

## Environment setup (uv)
Create and sync the environment, then install the project in editable mode:

```bash
uv sync
uv pip install -e .
```
If you encounter a problem regarding `import ctypes`, this indicates a python version that is too new. In that case,
activate the environment with an older python version, such as 3.10.12:
```
>>> uv venv --python=3.10.12 .venv
>>> uv pip install --upgrade uv
>>> uv sync
```


## Commands
All training/inference scripts use Hydra configs in `pera/scripts/cfgs/`.

### Training
Train the provided ESM3-1.4B model checkpoint with ERA, SFT, or DPO:
```bash
pera_train
```

Override config values with Hydra-style overrides, for example:
```bash
pera_train train.trainer_args.max_epochs=100 train.trainer_args.devices=1
```

### Inference (sampling)
Sample sequences from a trained ESM3-1.4B model:
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
