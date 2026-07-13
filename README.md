# era-directed-evolution
Official repository for "Efficient, Few-shot Directed Evolution with Energy Rank Alignment".

## Quick Start

### Prerequisites
- Python ≥3.10
- CUDA-capable GPU (recommended)
- [uv](https://github.com/astral-sh/uv) package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/rotskoff-group/era-directed-evolution.git
cd era-directed-evolution

# Create and sync the environment
uv sync
uv pip install -e .
```

### Basic Workflow
ERA-based directed evolution follows three main steps:

1. **Compute Landscape**: Generate reference log-probabilities and energy values for your dataset
2. **Train**: Align the model using Energy Rank Alignment (ERA)
3. **Sample**: Generate new candidate sequences

**Minimal Example:**
```bash
# 1. Compute landscape (reference log-probs and energies)
pera_compute_landscape compute_landscape.data=GB1 compute_landscape.data_root_path=./data

# 2. Train the model with ERA
pera_train

# 3. Sample new sequences
pera_sample infer.network_filename=lightning_logs/version_0/checkpoints/best_model.ckpt
```

---

## Input Data Format

### Directory Structure
Place your dataset files in the following structure:
```
data/
└── <dataset_name>/
    ├── scale2max/
    │   └── <dataset_name>.csv
    └── <dataset_name>.fasta
```

### CSV Format Requirements
The CSV file in `data/<dataset>/scale2max/<dataset>.csv` must contain at least these columns:

- **`muts`**: Mutation positions (e.g., `"V39 D40 G41 V54"`)
- **`AAs`**: Amino acid sequence at mutated positions (e.g., `"KENG"`)
- **`fitness`**: Measured fitness value (positive float)

**Example CSV:**
```csv
muts,AAs,fitness
V39 D40 G41 V54,VDGV,1.0
V39 D40 G41 V54,KENG,2.5
V39 D40 G41 V54,ATVL,0.8
```

### FASTA Format
The FASTA file at `data/<dataset>/<dataset>.fasta` should contain the parent (wild-type) sequence:
```
>GB1_parent
MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE
```

**Note:** For nucleotide sequences (like DHFR), the code automatically translates to amino acids.

If your data is located elsewhere, override the path:
```bash
pera_compute_landscape compute_landscape.data_root_path=/path/to/your/data
```

---

## Reward Specification

**Energy Calculation:**
Fitness values are converted to energies:
```python
energy = -log(fitness)  # for fitness > 0
```
Lower fitness → higher energy (less favorable)

## Key Hyperparameters

Configure these in [pera/scripts/cfgs/train/train.yaml](pera/scripts/cfgs/train/train.yaml):

### β (beta)
- **Location:** `lightning_model_args.beta`
- **Default:** `-10.0`
- **Description:** Controls the width of the ERA target distribution. Larger magnitudes make the distribution tighter. Flipping the sign flips the lower energy → better model output convention since the energy is the negative reward.

### γ (gamma)
- **Location:** `lightning_model_args.gamma`
- **Default:** `0.0`
- **Description:** KL divergence regularization weight. Controls how much the policy stays close to the reference model. Higher γ → stronger regularization (more conservative). Lower γ → more aggressive optimization.

### Number of Rounds (Epochs)
- **Location:** `trainer_args.max_epochs`
- **Default:** `25`
- **Description:** Maximum training epochs.

### Batch Size
- **Location:** `nn.batch_size` (in [pera/scripts/cfgs/nn/geometric_transformer.yaml](pera/scripts/cfgs/nn/geometric_transformer.yaml))
- **Default:** `4`
- **Description:** Number of sequence pairs per training batch. Limited by GPU memory.

**Override example:**
```bash
pera_train train.trainer_args.max_epochs=5000 train.lightning_model_args.beta=-15.0 nn.batch_size=8
```

---

## Example: How to Run An Experiment

This example demonstrates one complete alignment round on the GB1 dataset.

### Step 1: Prepare Your Data
Ensure your data is organized as described in [Input Data Format](#input-data-format).

For GB1, you should have:
```
data/GB1/scale2max/GB1.csv
data/GB1/GB1.fasta
```

### Step 2: Compute Landscape
Generate reference log-probabilities and energies for all sequences in your dataset:

```bash
pera_compute_landscape \
  compute_landscape.data=GB1 \
  compute_landscape.data_root_path=./data \
  compute_landscape.network_filename=/path/to/pretrained/model.pt \
  compute_landscape.output_filename=GB1_landscape.hdf5
```

**Expected output:** `GB1_landscape.hdf5`
- Output fields:
  - `unmasked_sequences_decoded`: Amino acid sequences
  - `unmasked_sequence_tokens`: Tokenized sequences
  - `energies`: Energy values (from fitness)
  - `ref_logps`: Reference model log-probabilities

### Step 3: Train with ERA
Train the model using Energy Rank Alignment:

```bash
pera_train \
  global_args.dataset_filename=./GB1_landscape.hdf5 \
  train.trainer_args.max_epochs=25 \
  train.trainer_args.devices=1 \
  train.lightning_model_args.beta=-10.0 \
  train.lightning_model_args.gamma=0.0
```

**Monitor training:**
```bash
tensorboard --logdir=lightning_logs
```

**Expected output:** Trained model checkpoint at `lightning_logs/version_*/checkpoints/best_model.ckpt`

### Step 4: Sample New Sequences
Generate candidate sequences from the trained model:

```bash
pera_sample \
  infer.target=GB1 \
  infer.num_samples=96 \
  infer.network_filename=./lightning_logs/version_0/checkpoints/best_model.ckpt \
  infer.data_root_path=./data \
  infer.output_filename=GB1_samples.pt \
  infer.sampling_temperature=1.0
```

---

## Advanced Usage

### All Available Commands

All commands use Hydra configs in [pera/scripts/cfgs/](pera/scripts/cfgs/).

#### Training
Train a transformer model:
```bash
pera_train
```

Override config values with Hydra-style overrides:
```bash
pera_train train.trainer_args.max_epochs=100 train.trainer_args.devices=2
```

#### Sampling (Inference)
Sample sequences from a trained model:
```bash
pera_sample
```

Example with overrides:
```bash
pera_sample \
  infer.target=GB1 \
  infer.num_samples=512 \
  infer.network_filename=/path/to/checkpoint.pt
```

#### Compute Landscape Log-Probabilities
Compute per-sequence log-probabilities for a dataset:
```bash
pera_compute_landscape
```

Example with overrides:
```bash
pera_compute_landscape \
  compute_landscape.data=GB1 \
  compute_landscape.data_root_path=./data \
  compute_landscape.batch_size=8
```

---

## Configuration Files

Key configuration files:
- [pera/scripts/cfgs/train_transformer.yaml](pera/scripts/cfgs/train_transformer.yaml): Main training config
- [pera/scripts/cfgs/train/train.yaml](pera/scripts/cfgs/train/train.yaml): Training hyperparameters (β, γ, optimizer, etc.)
- [pera/scripts/cfgs/nn/geometric_transformer.yaml](pera/scripts/cfgs/nn/geometric_transformer.yaml): Model architecture
- [pera/scripts/cfgs/infer_transformer.yaml](pera/scripts/cfgs/infer_transformer.yaml): Inference config
- [pera/scripts/cfgs/compute_landscape.yaml](pera/scripts/cfgs/compute_landscape.yaml): Landscape computation config

---

## Citation
If you use this code, please cite:
```bibtex
@article{era-directed-evolution,
  title={Efficient, Few-shot Directed Evolution with Energy Rank Alignment},
  author={Ibarraran, Sebastian and Chennakesavalu, Shriram and Hu, Frank and Rotskoff, Grant},
  year={2026}
}
```
