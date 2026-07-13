# Efficient, Few-shot Directed Evolution with Energy Rank Alignment

ERA is a protein sequence modeling toolkit for few-shot directed evolution using energy rank alignment. This page is a quick overview of the project and how to get started running the training and inference workflows.

## Highlights
- Hydra-driven training and inference with minimal CLI commands.
- Geometric transformer architecture for sequence modeling.
- Support for computing full landscape probabilities.

## Quickstart
Set up the environment and install the package in editable mode:

```bash
uv sync
uv pip install -e .
```

## Core commands
All commands use configs under `pera/scripts/cfgs/` and accept Hydra overrides.

Train a transformer model:
```bash
pera_train
```

Sample sequences from a trained model:
```bash
pera_sample
```

Compute per-sequence log-probabilities for a dataset:
```bash
pera_compute_landscape
```

## Data layout
Place datasets under the repository `data/` subdirectory and follow this layout:

```text
data/<dataset>/scale2max/<dataset>.csv
data/<dataset>/<dataset>.fasta
```

If your data lives elsewhere, set `infer.data_root_path` or
`compute_landscape.data_root_path` via Hydra overrides or config files.

## Where to look next
- Repository overview and details: `README.md`
- Configs: `pera/scripts/cfgs/`
- Models and layers: `pera/nn/`
