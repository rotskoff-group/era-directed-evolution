import os
import re

import hydra
import pandas as pd
import torch
from Bio.Seq import Seq
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from omegaconf import OmegaConf

from pera.nn import BidirectionalModel, sample_components_from_bidirectional_transformer


@hydra.main(version_base="1.3", config_path="../cfgs", config_name="infer_transformer")
def main(cfg):
    infer_config = cfg.infer

    device = torch.device(infer_config.device)
    sampling_temperature = infer_config.sampling_temperature

    model_config = OmegaConf.load(infer_config.model_config_filename)
    nn_config = model_config["nn"]
    train_config = model_config["train"]

    OmegaConf.update(train_config, "lightning_model_args.sampling_temperature", sampling_temperature)
    OmegaConf.update(train_config, "lightning_model_args.better_energy", infer_config.better_energy)



    esm_model = BidirectionalModel(nn_config["model"],
                                   nn_config["model_args"],
                                   **train_config["lightning_model_args"]).to(device)
    esm_model.load_model_from_ckpt(infer_config.network_filename)
    esm_model.eval()
    print("")
    residue_token_info = nn_config["model_args"]["residue_token_info"]
    structure_token_info = nn_config["model_args"]["struc_token_info"]
    mask_token_sequence = residue_token_info["mask"]
    bos_token_sequence = residue_token_info["bos"]
    eos_token_sequence = residue_token_info["eos"]
    pad_token_sequence = residue_token_info["pad"]

    save_folder_name = infer_config.save_folder_name
    os.makedirs(save_folder_name, exist_ok=True)

    data = infer_config.target
    num_samples = infer_config.num_samples
    data_root_path = infer_config.data_root_path

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

    muts_row_index = infer_config.gb1_muts_index if data == "GB1" else 0
    muts = df["muts"].iloc[muts_row_index]

    numbers = re.findall(r"\d+", muts)
    mask_indices = list(map(int, numbers))
    num_to_generate_per_mask = infer_config.num_to_generate_per_mask

    parent_sequence = torch.tensor(sequence_tokenizer.encode(parent_sequence_decoded,
                                                             add_special_tokens=True), device=device).unsqueeze(0).long()
    sequence_length = parent_sequence.shape[1]

    all_masked_sequences = []
    all_unmasked_sequences_decoded = []
    all_unmasked_sequences = []
    all_logps = []

    while len(all_unmasked_sequences_decoded) < num_samples:
        print(len(all_unmasked_sequences_decoded))

        masked_sequences = parent_sequence.clone().repeat(num_to_generate_per_mask, 1)
        masked_sequences[:, mask_indices] = mask_token_sequence

        sequence_id = torch.ones((num_to_generate_per_mask, sequence_length), device=device).long() * 1

        structure_tokens = torch.ones((num_to_generate_per_mask, sequence_length), device=device).long() * structure_token_info["mask"]
        structure_tokens[:, 0] = structure_token_info["bos"]
        structure_tokens[:, -1] = structure_token_info["eos"]

        coords = torch.inf * torch.ones((num_to_generate_per_mask, sequence_length, 3, 3), device=device)

        average_plddt = torch.ones((num_to_generate_per_mask), device=device)

        per_res_plddt = torch.zeros((num_to_generate_per_mask, sequence_length), device=device)
        ss8_tokens = torch.zeros((num_to_generate_per_mask, sequence_length), device=device).long()
        sasa_tokens = torch.zeros((num_to_generate_per_mask, sequence_length), device=device).long()

        function_tokens = torch.zeros((num_to_generate_per_mask, sequence_length, 8), device=device).long()
        residue_annotation_tokens = torch.zeros((num_to_generate_per_mask, sequence_length, 16), device=device).long()

        with torch.no_grad():
            unmasked_sequences = sample_components_from_bidirectional_transformer(transformer_model=esm_model,
                                                                                  masked_sequence_tokens=masked_sequences,
                                                                                  structure_tokens=structure_tokens,
                                                                                  average_plddt=average_plddt,
                                                                                  per_res_plddt=per_res_plddt,
                                                                                  ss8_tokens=ss8_tokens,
                                                                                  sasa_tokens=sasa_tokens,
                                                                                  function_tokens=function_tokens,
                                                                                  residue_annotation_tokens=residue_annotation_tokens,
                                                                                  bb_coords=coords,
                                                                                  sequence_id=sequence_id,
                                                                                  mask_token_sequence=mask_token_sequence,
                                                                                  bos_token_sequence=bos_token_sequence,
                                                                                  eos_token_sequence=eos_token_sequence,
                                                                                  pad_token_sequence=pad_token_sequence,
                                                                                  inference_batch_size=infer_config.inference_batch_size)

            masked_indices = (masked_sequences == mask_token_sequence).float()
            logits = esm_model.nn(sequence_tokens=masked_sequences,
                                  structure_tokens=structure_tokens,
                                  average_plddt=average_plddt,
                                  per_res_plddt=per_res_plddt,
                                  ss8_tokens=ss8_tokens,
                                  sasa_tokens=sasa_tokens,
                                  function_tokens=function_tokens,
                                  residue_annotation_tokens=residue_annotation_tokens,
                                  sequence_id=sequence_id,
                                  bb_coords=coords)["sequence_logits"].detach()
            logps = torch.nn.functional.log_softmax(logits / sampling_temperature, dim=-1)
            logps = torch.gather(logps, dim=-1, index=unmasked_sequences.unsqueeze(-1)).squeeze(-1)
            logps = (logps * masked_indices).sum(-1).detach()

            decoded_seqs = [sequence.replace(" ", "")
                            for sequence in sequence_tokenizer.batch_decode(unmasked_sequences[:, 1:-1])]
            for seq, logp, masked_seq, unmasked_seq in zip(decoded_seqs, logps, masked_sequences, unmasked_sequences):
                if seq in all_unmasked_sequences_decoded:
                    continue
                all_unmasked_sequences_decoded.append(seq)
                all_logps.append(logp)
                all_masked_sequences.append(masked_seq)
                all_unmasked_sequences.append(unmasked_seq)

    all_unmasked_sequences_decoded = all_unmasked_sequences_decoded[:num_samples]
    all_masked_sequences = all_masked_sequences[:num_samples]
    all_unmasked_sequences = all_unmasked_sequences[:num_samples]
    all_logps = all_logps[:num_samples]

    all_masked_sequences = torch.stack(all_masked_sequences, dim=0)
    all_unmasked_sequences = torch.stack(all_unmasked_sequences, dim=0)
    all_logps = torch.stack(all_logps, dim=0)

    to_save = {"parent_sequence": parent_sequence,
               "all_masked_sequences": all_masked_sequences,
               "all_unmasked_sequences": all_unmasked_sequences,
               "all_unmasked_sequences_decoded": all_unmasked_sequences_decoded,
               "all_logps": all_logps}
    torch.save(to_save, f"{save_folder_name}/{infer_config.output_filename}")


if __name__ == "__main__":
    main()
