import copy
import re

import hydra
import h5py
import numpy as np
import pandas as pd
import torch
from Bio.Seq import Seq
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from omegaconf import OmegaConf

from pera.nn import BidirectionalModel


@hydra.main(version_base="1.3", config_path="../cfgs", config_name="compute_landscape")
def main(cfg):
    compute_config = cfg.compute_landscape

    device = torch.device(compute_config.device)
    sampling_temperature = compute_config.sampling_temperature

    model_config = OmegaConf.load(compute_config.model_config_filename)
    nn_config = model_config.nn
    train_config = model_config.train

    OmegaConf.update(train_config, "lightning_model_args.sampling_temperature", sampling_temperature)
    OmegaConf.update(train_config, "lightning_model_args.better_energy", compute_config.better_energy)
    esm_model = BidirectionalModel(nn_config["model"],
                                   nn_config["model_args"],
                                   **train_config["lightning_model_args"]).to(device)
    esm_model.load_model_from_ckpt(compute_config.network_filename)
    esm_model.eval()
    print("")
    residue_token_info = nn_config["model_args"]["residue_token_info"]
    structure_token_info = nn_config["model_args"]["struc_token_info"]
    mask_token_sequence = residue_token_info["mask"]

    data = compute_config.data
    data_root_path = compute_config.data_root_path
    batch_size = compute_config.batch_size
    output_filename = compute_config.output_filename

    sequence_tokenizer = EsmSequenceTokenizer()

    if data.startswith("TrpB"):
        df = pd.read_csv(f"{data_root_path}/TrpB/scale2max/{data}.csv")
        with open(f"{data_root_path}/TrpB/TrpB.fasta", "r") as file:
            parent_sequence_decoded = file.readlines()[1].strip()

    elif data == "DHFR":
        df = pd.read_csv(f"{data_root_path}/{data}/scale2max/{data}.csv")
        with open(f"{data_root_path}/{data}/{data}.fasta", "r") as file:
            nucleotide_seq = file.readlines()[1].strip()
        nucleotide_seq = Seq(nucleotide_seq)
        parent_sequence_decoded = str(nucleotide_seq.translate())  # Translate to amino acid sequence

    else:
        df = pd.read_csv(f"{data_root_path}/{data}/scale2max/{data}.csv")
        with open(f"{data_root_path}/{data}/{data}.fasta", "r") as file:
            parent_sequence_decoded = file.readlines()[1].strip()

    muts_row_index = compute_config.gb1_muts_index if data == "GB1" else 0
    muts = df["muts"].iloc[muts_row_index]

    numbers = re.findall(r"\d+", muts)
    mask_indices = list(map(int, numbers))

    parent_sequence = torch.tensor(sequence_tokenizer.encode(parent_sequence_decoded,
                                                             add_special_tokens=True), device=device).unsqueeze(0).long()
    sequence_length = parent_sequence.shape[1]

    all_mutated_sequences = []
    all_fitness_values = []
    for row in df.iterrows():
        mutated_sequence = copy.copy(parent_sequence_decoded)
        mutation = row[1]["AAs"]
        # Skip sequences with stop codons
        if "*" in mutation:
            continue
        for i in range(len(mask_indices)):
            mutated_sequence = mutated_sequence[:mask_indices[i] - 1] + mutation[i] + mutated_sequence[mask_indices[i]:]
        all_mutated_sequences.append(mutated_sequence)
        all_fitness_values.append(row[1]["fitness"])

    all_fitness_values = np.array(all_fitness_values)
    all_fitness_values = np.where(all_fitness_values > 0, -np.log(all_fitness_values), 10)
    all_unmasked_sequences = torch.tensor(
        np.array(sequence_tokenizer(list(all_mutated_sequences))["input_ids"]), device=device
    ).long()
    print(all_unmasked_sequences.shape)

    all_masked_sequences = all_unmasked_sequences.clone()
    all_masked_sequences[:, mask_indices] = mask_token_sequence

    sequence_length = all_unmasked_sequences.shape[1]

    sequence_id = torch.ones((all_unmasked_sequences.shape[0], sequence_length), device=device).long() * 1

    structure_tokens = torch.ones((1, sequence_length), device=device).long() * structure_token_info["mask"]
    structure_tokens[:, 0] = structure_token_info["bos"]
    structure_tokens[:, -1] = structure_token_info["eos"]

    coords = torch.inf * torch.ones((1, sequence_length, 3, 3), device=device)

    average_plddt = torch.ones((1), device=device)

    per_res_plddt = torch.zeros((1, sequence_length), device=device)
    ss8_tokens = torch.zeros((1, sequence_length), device=device).long()
    sasa_tokens = torch.zeros((1, sequence_length), device=device).long()

    function_tokens = torch.zeros((1, sequence_length, 8), device=device).long()
    residue_annotation_tokens = torch.zeros((1, sequence_length, 16), device=device).long()

    num_batches = (all_unmasked_sequences.shape[0] + batch_size - 1) // batch_size

    with h5py.File(output_filename, "w") as f:
        f.create_dataset("unmasked_sequences_decoded", data=np.array(all_mutated_sequences, dtype="S"))
        f.create_dataset("unmasked_sequence_tokens", data=all_unmasked_sequences.cpu().numpy())
        f.create_dataset("energies", data=all_fitness_values)
        all_logps = []

        for i in range(num_batches):
            print(i)
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, all_unmasked_sequences.shape[0])
            batch_masked_sequences = all_masked_sequences[batch_start:batch_end]
            batch_unmasked_sequences = all_unmasked_sequences[batch_start:batch_end]
            batch_structure_tokens = structure_tokens.repeat(batch_end - batch_start, 1)
            batch_coords = coords.repeat(batch_end - batch_start, 1, 1, 1)
            batch_average_plddt = average_plddt.repeat(batch_end - batch_start)
            batch_per_res_plddt = per_res_plddt.repeat(batch_end - batch_start, 1)
            batch_ss8_tokens = ss8_tokens.repeat(batch_end - batch_start, 1)
            batch_sasa_tokens = sasa_tokens.repeat(batch_end - batch_start, 1)
            batch_function_tokens = function_tokens.repeat(batch_end - batch_start, 1, 1)
            batch_residue_annotation_tokens = residue_annotation_tokens.repeat(batch_end - batch_start, 1, 1)
            batch_sequence_id = sequence_id[batch_start:batch_end]
            batch_masked_indices = (batch_masked_sequences == mask_token_sequence).float()

            with torch.no_grad():
                forward = esm_model.nn(sequence_tokens=batch_masked_sequences,
                                       structure_tokens=batch_structure_tokens,
                                       average_plddt=batch_average_plddt,
                                       per_res_plddt=batch_per_res_plddt,
                                       ss8_tokens=batch_ss8_tokens,
                                       sasa_tokens=batch_sasa_tokens,
                                       function_tokens=batch_function_tokens,
                                       residue_annotation_tokens=batch_residue_annotation_tokens,
                                       sequence_id=batch_sequence_id,
                                       bb_coords=batch_coords)
                logits = forward["sequence_logits"].detach()

                logps = torch.nn.functional.log_softmax(logits / sampling_temperature, dim=-1)
                logps = torch.gather(logps, dim=-1, index=batch_unmasked_sequences.unsqueeze(-1)).squeeze(-1)

                logps = (logps * batch_masked_indices).sum(-1).detach().cpu()
                all_logps.append(logps)

        f.create_dataset("ref_logps", data=torch.cat(all_logps, dim=0).numpy())

    with h5py.File(output_filename, "r") as f:
        for key in f.keys():
            print(key, f[key].shape)


if __name__ == "__main__":
    main()
