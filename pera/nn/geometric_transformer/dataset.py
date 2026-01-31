import torch
import numpy as np
from torch.utils.data import Dataset


def geometric_transformer_sft_collate_fn(data):
    (masked_sequence_tokens,
     structure_tokens,
     average_plddt,
     per_res_plddt,
     ss8_tokens,
     sasa_tokens,
     function_tokens,
     residue_annotation_tokens,
     sequence_id,
     bb_coords,
     unmasked_sequence_tokens,
     loss_mask) = zip(*data)

    masked_sequence_tokens = torch.stack(masked_sequence_tokens, axis=0)
    structure_tokens = torch.stack(structure_tokens, axis=0)
    average_plddt = torch.stack(average_plddt, axis=0)
    per_res_plddt = torch.stack(per_res_plddt, axis=0)
    ss8_tokens = torch.stack(ss8_tokens, axis=0)
    sasa_tokens = torch.stack(sasa_tokens, axis=0)
    function_tokens = torch.stack(function_tokens, axis=0)
    residue_annotation_tokens = torch.stack(residue_annotation_tokens, axis=0)
    sequence_id = torch.stack(sequence_id, axis=0)
    bb_coords = torch.stack(bb_coords, axis=0)
    unmasked_sequence_tokens = torch.stack(unmasked_sequence_tokens, axis=0)
    loss_mask = torch.stack(loss_mask, axis=0)

    return (masked_sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
            sequence_id,
            bb_coords,
            unmasked_sequence_tokens,
            loss_mask)


class GeometricTransformerSFTDataset(Dataset):
    def __init__(self, get_hdf5_data, data_in_memory=True, config=None):
        self.get_hdf5_data = get_hdf5_data
        self.data_in_memory = data_in_memory
        t_hdf5 = self.get_hdf5_data()
        self.length = len(t_hdf5['sequence_tokens'])
        self.mask_token_sequence = config["model_args"]["residue_token_info"]["mask"]
        self.bos_token_sequence = config["model_args"]["residue_token_info"]["bos"]
        self.eos_token_sequence = config["model_args"]["residue_token_info"]["eos"]
        self.pad_token_sequence = config["model_args"]["residue_token_info"]["pad"]

        del t_hdf5

    def __len__(self):
        return self.length

    def _sample_from_beta_linear30(self):
        if np.random.rand() < 0.8:
            # Draw from Beta(3, 9) distribution
            mask_rate = np.random.beta(3, 9)
        else:
            # Draw from a linear distribution
            mask_rate = np.random.uniform(0, 1)

        return mask_rate

    def _get_mask_indices(self, sequence_length):
        mask_rate = self._sample_from_beta_linear30()
        num_to_mask = int(mask_rate * sequence_length)
        num_to_mask = max(1, num_to_mask)
        mask_indices = np.random.choice(
            sequence_length, num_to_mask, replace=False)
        return mask_indices

    def open_hdf5(self):
        self.t_hdf5 = self.get_hdf5_data()

        self.sequence_tokens = self.t_hdf5['sequence_tokens']
        self.structural_tokens = self.t_hdf5['structural_tokens']
        self.average_plddt = self.t_hdf5['average_plddt']
        self.per_res_plddt = self.t_hdf5['per_res_plddt']
        self.ss8_tokens = self.t_hdf5['ss8_tokens']
        self.sasa_tokens = self.t_hdf5['sasa_tokens']
        self.function_tokens = self.t_hdf5['function_tokens']
        self.residue_annotation_tokens = self.t_hdf5['residue_annotation_tokens']
        self.sequence_id = self.t_hdf5['sequence_id']
        self.bb_coords = self.t_hdf5['bb_coords']
        self.mask_indices = self.t_hdf5['mask_indices']

        if self.data_in_memory:
            self.sequence_tokens = self.sequence_tokens[:]
            self.structural_tokens = self.structural_tokens[:]
            self.average_plddt = self.average_plddt[:]
            self.per_res_plddt = self.per_res_plddt[:]
            self.ss8_tokens = self.ss8_tokens[:]
            self.sasa_tokens = self.sasa_tokens[:]
            self.function_tokens = self.function_tokens[:]
            self.residue_annotation_tokens = self.residue_annotation_tokens[:]
            self.sequence_id = self.sequence_id[:]
            self.bb_coords = self.bb_coords[:]
            self.mask_indices = self.mask_indices[:]

    def __getitem__(self, idx):
        if not hasattr(self, 't_hdf5'):
            self.open_hdf5()

        unmasked_sequence_tokens = torch.tensor(self.sequence_tokens[idx],
                                                dtype=torch.long)

        sequence_id = torch.tensor(self.sequence_id[idx],
                                   dtype=torch.long)

        if self.t_hdf5.attrs["fixed_structural_tokens"]:
            structural_tokens = torch.tensor(self.structural_tokens[0],
                                             dtype=torch.long)
        else:
            structural_tokens = torch.tensor(self.structural_tokens[idx],
                                             dtype=torch.long)

        if self.t_hdf5.attrs["fixed_average_plddt"]:
            average_plddt = torch.tensor(self.average_plddt[0],
                                         dtype=torch.float)
        else:
            average_plddt = torch.tensor(self.average_plddt[idx],
                                         dtype=torch.float)

        if self.t_hdf5.attrs["fixed_per_res_plddt"]:
            per_res_plddt = torch.tensor(self.per_res_plddt[0],
                                         dtype=torch.float)
        else:
            per_res_plddt = torch.tensor(self.per_res_plddt[idx],
                                         dtype=torch.float)

        if self.t_hdf5.attrs["fixed_ss8_tokens"]:
            ss8_tokens = torch.tensor(self.ss8_tokens[0], dtype=torch.long)
        else:
            ss8_tokens = torch.tensor(self.ss8_tokens[idx], dtype=torch.long)

        if self.t_hdf5.attrs["fixed_sasa_tokens"]:
            sasa_tokens = torch.tensor(self.sasa_tokens[0], dtype=torch.long)
        else:
            sasa_tokens = torch.tensor(self.sasa_tokens[idx], dtype=torch.long)

        if self.t_hdf5.attrs["fixed_function_tokens"]:
            function_tokens = torch.tensor(self.function_tokens[0],
                                           dtype=torch.long)
        else:
            function_tokens = torch.tensor(self.function_tokens[idx],
                                           dtype=torch.long)

        if self.t_hdf5.attrs["fixed_residue_annotation_tokens"]:
            residue_annotation_tokens = torch.tensor(self.residue_annotation_tokens[0],
                                                     dtype=torch.long)
        else:
            residue_annotation_tokens = torch.tensor(self.residue_annotation_tokens[idx],
                                                     dtype=torch.long)

        if self.t_hdf5.attrs["fixed_bb_coords"]:
            bb_coords = torch.tensor(self.bb_coords[0], dtype=torch.float)
        else:
            bb_coords = torch.tensor(self.bb_coords[idx], dtype=torch.float)

        sequence_non_special_indices = torch.where((unmasked_sequence_tokens != self.bos_token_sequence)
                                                   & (unmasked_sequence_tokens != self.eos_token_sequence)
                                                   & (unmasked_sequence_tokens != self.pad_token_sequence))[0]
        sequence_length = sequence_non_special_indices.shape[0]
        
        if self.mask_indices is None:
            non_special_mask_indices = torch.tensor(self._get_mask_indices(sequence_length), dtype=torch.long)
            mask_indices = sequence_non_special_indices[non_special_mask_indices]

            masked_sequence_tokens = torch.clone(unmasked_sequence_tokens)
            masked_sequence_tokens[mask_indices] = self.mask_token_sequence
            loss_mask = (masked_sequence_tokens == self.mask_token_sequence)

        else:
            masked_sequence_tokens = torch.clone(unmasked_sequence_tokens)
            masked_sequence_tokens[self.mask_indices] = self.mask_token_sequence
            loss_mask = (masked_sequence_tokens == self.mask_token_sequence)


        return (masked_sequence_tokens,
                structural_tokens,
                average_plddt,
                per_res_plddt,
                ss8_tokens,
                sasa_tokens,
                function_tokens,
                residue_annotation_tokens,
                sequence_id,
                bb_coords,
                unmasked_sequence_tokens,
                loss_mask)




def geometric_transformer_era_collate_fn(data):
    (masked_sequence_tokens,
     structure_tokens,
     average_plddt,
     per_res_plddt,
     ss8_tokens,
     sasa_tokens,
     function_tokens,
     residue_annotation_tokens,
     sequence_id,
     bb_coords,
     unmasked_sequence_tokens,
     logp_mask,
     energies,
     ref_logps) = zip(*data)

    masked_sequence_tokens = torch.cat(masked_sequence_tokens, axis=0)
    structure_tokens = torch.cat(structure_tokens, axis=0)
    average_plddt = torch.cat(average_plddt, axis=0)
    per_res_plddt = torch.cat(per_res_plddt, axis=0)
    ss8_tokens = torch.cat(ss8_tokens, axis=0)
    sasa_tokens = torch.cat(sasa_tokens, axis=0)
    function_tokens = torch.cat(function_tokens, axis=0)
    residue_annotation_tokens = torch.cat(residue_annotation_tokens, axis=0)
    sequence_id = torch.cat(sequence_id, axis=0)
    bb_coords = torch.cat(bb_coords, axis=0)
    unmasked_sequence_tokens = torch.cat(unmasked_sequence_tokens, axis=0)
    logp_mask = torch.cat(logp_mask, axis=0)
    energies = torch.cat(energies, axis=0)
    ref_logps = torch.cat(ref_logps, axis=0)

    return (masked_sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
            sequence_id,
            bb_coords,
            unmasked_sequence_tokens,
            logp_mask,
            energies,
            ref_logps)


class GeometricTransformerERADataset(Dataset):
    def __init__(self, get_hdf5_data, data_in_memory=True, config=None):
        self.get_hdf5_data = get_hdf5_data
        self.data_in_memory = data_in_memory
        t_hdf5 = self.get_hdf5_data()
        self.length = len(t_hdf5['structural_tokens'])

        self.num_examples_per_prompt = t_hdf5.attrs['num_examples_per_prompt']
        self.num_pairs_per_prompt = (self.num_examples_per_prompt
                                     * (self.num_examples_per_prompt - 1) // 2)
        self.num_prompts = t_hdf5.attrs['num_prompts']


        self.mask_token_sequence = config["model_args"]["residue_token_info"]["mask"]
        self.bos_token_sequence = config["model_args"]["residue_token_info"]["bos"]
        self.eos_token_sequence = config["model_args"]["residue_token_info"]["eos"]
        self.pad_token_sequence = config["model_args"]["residue_token_info"]["pad"]

        self.mask_token_struc = config["model_args"]["struc_token_info"]["mask"]
        self.bos_token_struc = config["model_args"]["struc_token_info"]["bos"]
        self.eos_token_struc = config["model_args"]["struc_token_info"]["eos"]
        self.pad_token_struc = config["model_args"]["struc_token_info"]["pad"]

        assert (len(t_hdf5['masked_sequence_tokens'])
                == self.num_examples_per_prompt * self.num_prompts)
        del t_hdf5

        self.length = int(self.num_prompts * self.num_pairs_per_prompt)
        self.all_pairs = [[i, j] for i in range(self.num_examples_per_prompt)
                          for j in range(i + 1, self.num_examples_per_prompt)]

    def __len__(self):
        return self.length

    def open_hdf5(self):
        self.t_hdf5 = self.get_hdf5_data()

        self.masked_sequence_tokens = self.t_hdf5['masked_sequence_tokens']
        self.unmasked_sequence_tokens = self.t_hdf5['unmasked_sequence_tokens']
        
        self.structural_tokens = self.t_hdf5['structural_tokens']
        self.average_plddt = self.t_hdf5['average_plddt']
        self.per_res_plddt = self.t_hdf5['per_res_plddt']
        self.ss8_tokens = self.t_hdf5['ss8_tokens']
        self.sasa_tokens = self.t_hdf5['sasa_tokens']
        self.function_tokens = self.t_hdf5['function_tokens']
        self.residue_annotation_tokens = self.t_hdf5['residue_annotation_tokens']
        self.sequence_id = self.t_hdf5['sequence_id']
        self.bb_coords = self.t_hdf5['bb_coords']
        self.energies = self.t_hdf5['energies']
        self.ref_logps = self.t_hdf5['ref_logps']

        if self.data_in_memory:
            self.sequence_tokens = self.sequence_tokens[:]
            self.structural_tokens = self.structural_tokens[:]
            self.average_plddt = self.average_plddt[:]
            self.per_res_plddt = self.per_res_plddt[:]
            self.ss8_tokens = self.ss8_tokens[:]
            self.sasa_tokens = self.sasa_tokens[:]
            self.function_tokens = self.function_tokens[:]
            self.residue_annotation_tokens = self.residue_annotation_tokens[:]
            self.sequence_id = self.sequence_id[:]
            self.bb_coords = self.bb_coords[:]

    def __getitem__(self, idx):
        if not hasattr(self, 't_hdf5'):
            self.open_hdf5()

        prompt_index = idx // self.num_pairs_per_prompt
        pair_index = idx % self.num_pairs_per_prompt
        pair = self.all_pairs[pair_index]
        dataset_idx_1 = prompt_index * self.num_examples_per_prompt + pair[0]
        dataset_idx_2 = prompt_index * self.num_examples_per_prompt + pair[1]
        assert dataset_idx_1 != dataset_idx_2

        masked_sequence_tokens = torch.tensor(self.masked_sequence_tokens[[dataset_idx_1, dataset_idx_2]],
                                              dtype=torch.long)
        unmasked_sequence_tokens = torch.tensor(self.unmasked_sequence_tokens[[dataset_idx_1, dataset_idx_2]],
                                                dtype=torch.long)
        
        energies = torch.tensor(self.energies[[dataset_idx_1, dataset_idx_2]],
                                dtype=torch.float)
        ref_logps = torch.tensor(self.ref_logps[[dataset_idx_1, dataset_idx_2]],
                                 dtype=torch.float)

        sequence_id = torch.tensor(self.sequence_id[[dataset_idx_1, dataset_idx_2]],
                                   dtype=torch.long)

        if self.t_hdf5.attrs["fixed_structural_tokens"]:
            """Fixed is assumed to be all pad
            """
            structural_tokens = torch.tensor(self.structural_tokens[0:1],
                                             dtype=torch.long)
            structural_tokens = structural_tokens.repeat(2, 1)

            structural_tokens[masked_sequence_tokens !=
                              self.pad_token_sequence] = self.mask_token_struc

            structural_tokens[masked_sequence_tokens ==
                            self.bos_token_sequence] = self.bos_token_struc
            structural_tokens[masked_sequence_tokens ==
                            self.eos_token_sequence] = self.eos_token_struc
        else:
            structural_tokens = torch.tensor(self.structural_tokens[[dataset_idx_1, dataset_idx_2]],
                                             dtype=torch.long)
            

        

        if self.t_hdf5.attrs["fixed_average_plddt"]:
            average_plddt = torch.tensor(self.average_plddt[0:1],
                                         dtype=torch.float)
            average_plddt = average_plddt.repeat(2)
        else:
            average_plddt = torch.tensor(self.average_plddt[[dataset_idx_1, dataset_idx_2]],
                                         dtype=torch.float)

        if self.t_hdf5.attrs["fixed_per_res_plddt"]:
            per_res_plddt = torch.tensor(self.per_res_plddt[0:1],
                                         dtype=torch.float)
            per_res_plddt = per_res_plddt.repeat(2, 1)
        else:
            per_res_plddt = torch.tensor(self.per_res_plddt[[dataset_idx_1, dataset_idx_2]],
                                         dtype=torch.float)

        if self.t_hdf5.attrs["fixed_ss8_tokens"]:
            ss8_tokens = torch.tensor(self.ss8_tokens[0:1], dtype=torch.long)
            ss8_tokens = ss8_tokens.repeat(2, 1)
        else:
            ss8_tokens = torch.tensor(self.ss8_tokens[[dataset_idx_1, dataset_idx_2]], 
                                      dtype=torch.long)

        if self.t_hdf5.attrs["fixed_sasa_tokens"]:
            sasa_tokens = torch.tensor(self.sasa_tokens[0:1], dtype=torch.long)
            sasa_tokens = sasa_tokens.repeat(2, 1)
        else:
            sasa_tokens = torch.tensor(self.sasa_tokens[[dataset_idx_1, dataset_idx_2]], 
                                       dtype=torch.long)

        if self.t_hdf5.attrs["fixed_function_tokens"]:
            function_tokens = torch.tensor(self.function_tokens[0:1],
                                           dtype=torch.long)
            function_tokens = function_tokens.repeat(2, 1, 1)
        else:
            function_tokens = torch.tensor(self.function_tokens[[dataset_idx_1, dataset_idx_2]],
                                           dtype=torch.long)

        if self.t_hdf5.attrs["fixed_residue_annotation_tokens"]:
            residue_annotation_tokens = torch.tensor(self.residue_annotation_tokens[0:1],
                                                     dtype=torch.long)
            residue_annotation_tokens = residue_annotation_tokens.repeat(2, 1, 1)
        else:
            residue_annotation_tokens = torch.tensor(self.residue_annotation_tokens[[dataset_idx_1, dataset_idx_2]],
                                                     dtype=torch.long)

        if self.t_hdf5.attrs["fixed_bb_coords"]:
            bb_coords = torch.tensor(self.bb_coords[0:1], dtype=torch.float)
            bb_coords = bb_coords.repeat(2, 1, 1, 1)
        else:
            bb_coords = torch.tensor(self.bb_coords[[dataset_idx_1, dataset_idx_2]], 
                                     dtype=torch.float)


        logp_mask = (masked_sequence_tokens == self.mask_token_sequence)

        assert torch.all(logp_mask[0] == logp_mask[1])

        return (masked_sequence_tokens,
                structural_tokens,
                average_plddt,
                per_res_plddt,
                ss8_tokens,
                sasa_tokens,
                function_tokens,
                residue_annotation_tokens,
                sequence_id,
                bb_coords,
                unmasked_sequence_tokens,
                logp_mask,
                energies,
                ref_logps)


def geometric_transformer_era_pretrain_collate_fn(data):
    (masked_sequence_tokens,
     structure_tokens,
     average_plddt,
     per_res_plddt,
     ss8_tokens,
     sasa_tokens,
     function_tokens,
     residue_annotation_tokens,
     sequence_id,
     bb_coords,
     unmasked_sequence_tokens,
     logp_mask,
     energies,
     ref_logps) = zip(*data)

    masked_sequence_tokens = torch.cat(masked_sequence_tokens, axis=0)
    structure_tokens = torch.cat(structure_tokens, axis=0)
    average_plddt = torch.cat(average_plddt, axis=0)
    per_res_plddt = torch.cat(per_res_plddt, axis=0)
    ss8_tokens = torch.cat(ss8_tokens, axis=0)
    sasa_tokens = torch.cat(sasa_tokens, axis=0)
    function_tokens = torch.cat(function_tokens, axis=0)
    residue_annotation_tokens = torch.cat(residue_annotation_tokens, axis=0)
    sequence_id = torch.cat(sequence_id, axis=0)
    bb_coords = torch.cat(bb_coords, axis=0)
    unmasked_sequence_tokens = torch.cat(unmasked_sequence_tokens, axis=0)
    logp_mask = torch.cat(logp_mask, axis=0)
    energies = torch.cat(energies, axis=0)
    ref_logps = torch.cat(ref_logps, axis=0)

    return (masked_sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
            sequence_id,
            bb_coords,
            unmasked_sequence_tokens,
            logp_mask,
            energies,
            ref_logps)


class GeometricTransformerERADatasetPretrain(Dataset):
    def __init__(self, get_hdf5_data, data_in_memory=True, config=None):
        self.get_hdf5_data = get_hdf5_data
        self.data_in_memory = data_in_memory
        t_hdf5 = self.get_hdf5_data()
        self.length = len(t_hdf5['masked_sequence_tokens'])

        self.num_examples_per_prompt = t_hdf5['num_examples_per_prompt'][:]
        self.num_prompts = t_hdf5.attrs['num_prompts']

        assert self.num_prompts == len(self.num_examples_per_prompt), \
            f"num_prompts ({self.num_prompts}) does not match length of num_examples_per_prompt ({len(self.num_examples_per_prompt)})"

        total_examples = sum(self.num_examples_per_prompt)
        assert total_examples == len(t_hdf5['masked_sequence_tokens']), \
            f"Sum of num_examples_per_prompt ({total_examples}) does not match dataset length ({len(t_hdf5['masked_sequence_tokens'])})"

        self.mask_token_sequence = config["model_args"]["residue_token_info"]["mask"]
        self.bos_token_sequence = config["model_args"]["residue_token_info"]["bos"]
        self.eos_token_sequence = config["model_args"]["residue_token_info"]["eos"]
        self.pad_token_sequence = config["model_args"]["residue_token_info"]["pad"]

        self.mask_token_struc = config["model_args"]["struc_token_info"]["mask"]
        self.bos_token_struc = config["model_args"]["struc_token_info"]["bos"]
        self.eos_token_struc = config["model_args"]["struc_token_info"]["eos"]
        self.pad_token_struc = config["model_args"]["struc_token_info"]["pad"]

        self.prompt_start_indices = []
        self.valid_pairs_per_prompt = []
        current_idx = 0

        for prompt_idx in range(self.num_prompts):
            self.prompt_start_indices.append(current_idx)

            num_examples = int(self.num_examples_per_prompt[prompt_idx])
            
            stabilizing_mut = t_hdf5['Stabilizing_mut'][current_idx:current_idx + num_examples]

            valid_pairs = []
            for i in range(num_examples):
                for j in range(i + 1, num_examples):
                    if stabilizing_mut[i] != stabilizing_mut[j]:
                        valid_pairs.append([i, j])

            self.valid_pairs_per_prompt.append(valid_pairs)
            current_idx += num_examples

        self.length = sum(len(pairs) for pairs in self.valid_pairs_per_prompt)

        self.cumulative_pairs = [0]
        for pairs in self.valid_pairs_per_prompt:
            self.cumulative_pairs.append(self.cumulative_pairs[-1] + len(pairs))

        del t_hdf5

    def __len__(self):
        return self.length

    def open_hdf5(self):
        self.t_hdf5 = self.get_hdf5_data()

        self.masked_sequence_tokens = self.t_hdf5['masked_sequence_tokens']
        self.unmasked_sequence_tokens = self.t_hdf5['unmasked_sequence_tokens']

        self.structural_tokens = self.t_hdf5['structural_tokens']
        self.average_plddt = self.t_hdf5['average_plddt']
        self.per_res_plddt = self.t_hdf5['per_res_plddt']
        self.ss8_tokens = self.t_hdf5['ss8_tokens']
        self.sasa_tokens = self.t_hdf5['sasa_tokens']
        self.function_tokens = self.t_hdf5['function_tokens']
        self.residue_annotation_tokens = self.t_hdf5['residue_annotation_tokens']
        self.sequence_id = self.t_hdf5['sequence_id']
        self.bb_coords = self.t_hdf5['bb_coords']
        self.energies = self.t_hdf5['energies']
        self.ref_logps = self.t_hdf5['ref_logps']

        if self.data_in_memory:
            self.masked_sequence_tokens = self.masked_sequence_tokens[:]
            self.unmasked_sequence_tokens = self.unmasked_sequence_tokens[:]
            self.structural_tokens = self.structural_tokens[:]
            self.average_plddt = self.average_plddt[:]
            self.per_res_plddt = self.per_res_plddt[:]
            self.ss8_tokens = self.ss8_tokens[:]
            self.sasa_tokens = self.sasa_tokens[:]
            self.function_tokens = self.function_tokens[:]
            self.residue_annotation_tokens = self.residue_annotation_tokens[:]
            self.sequence_id = self.sequence_id[:]
            self.bb_coords = self.bb_coords[:]
            self.energies = self.energies[:]
            self.ref_logps = self.ref_logps[:]

    def __getitem__(self, idx):
        if not hasattr(self, 't_hdf5'):
            self.open_hdf5()

        prompt_index = 0
        for i in range(len(self.cumulative_pairs) - 1):
            if self.cumulative_pairs[i] <= idx < self.cumulative_pairs[i + 1]:
                prompt_index = i
                break

        pair_index = idx - self.cumulative_pairs[prompt_index]
        pair = self.valid_pairs_per_prompt[prompt_index][pair_index]

        dataset_idx_1 = self.prompt_start_indices[prompt_index] + pair[0]
        dataset_idx_2 = self.prompt_start_indices[prompt_index] + pair[1]
        assert dataset_idx_1 != dataset_idx_2

        masked_sequence_tokens = torch.tensor(self.masked_sequence_tokens[[dataset_idx_1, dataset_idx_2]],
                                              dtype=torch.long)
        unmasked_sequence_tokens = torch.tensor(self.unmasked_sequence_tokens[[dataset_idx_1, dataset_idx_2]],
                                                dtype=torch.long)

        energies = torch.tensor(self.energies[[dataset_idx_1, dataset_idx_2]],
                                dtype=torch.float)
        ref_logps = torch.tensor(self.ref_logps[[dataset_idx_1, dataset_idx_2]],
                                 dtype=torch.float)

        sequence_id = torch.tensor(self.sequence_id[[dataset_idx_1, dataset_idx_2]],
                                   dtype=torch.long)

        if self.t_hdf5.attrs["fixed_structural_tokens"]:
            """Fixed is assumed to be all pad
            """
            structural_tokens = torch.tensor(self.structural_tokens[0:1],
                                             dtype=torch.long)
            structural_tokens = structural_tokens.repeat(2, 1)

            structural_tokens[masked_sequence_tokens !=
                              self.pad_token_sequence] = self.mask_token_struc

            structural_tokens[masked_sequence_tokens ==
                            self.bos_token_sequence] = self.bos_token_struc
            structural_tokens[masked_sequence_tokens ==
                            self.eos_token_sequence] = self.eos_token_struc
        else:
            structural_tokens = torch.tensor(self.structural_tokens[[dataset_idx_1, dataset_idx_2]],
                                             dtype=torch.long)




        if self.t_hdf5.attrs["fixed_average_plddt"]:
            average_plddt = torch.tensor(self.average_plddt[0:1],
                                         dtype=torch.float)
            average_plddt = average_plddt.repeat(2)
        else:
            average_plddt = torch.tensor(self.average_plddt[[dataset_idx_1, dataset_idx_2]],
                                         dtype=torch.float)

        if self.t_hdf5.attrs["fixed_per_res_plddt"]:
            per_res_plddt = torch.tensor(self.per_res_plddt[0:1],
                                         dtype=torch.float)
            per_res_plddt = per_res_plddt.repeat(2, 1)
        else:
            per_res_plddt = torch.tensor(self.per_res_plddt[[dataset_idx_1, dataset_idx_2]],
                                         dtype=torch.float)

        if self.t_hdf5.attrs["fixed_ss8_tokens"]:
            ss8_tokens = torch.tensor(self.ss8_tokens[0:1], dtype=torch.long)
            ss8_tokens = ss8_tokens.repeat(2, 1)
        else:
            ss8_tokens = torch.tensor(self.ss8_tokens[[dataset_idx_1, dataset_idx_2]],
                                      dtype=torch.long)

        if self.t_hdf5.attrs["fixed_sasa_tokens"]:
            sasa_tokens = torch.tensor(self.sasa_tokens[0:1], dtype=torch.long)
            sasa_tokens = sasa_tokens.repeat(2, 1)
        else:
            sasa_tokens = torch.tensor(self.sasa_tokens[[dataset_idx_1, dataset_idx_2]],
                                       dtype=torch.long)

        if self.t_hdf5.attrs["fixed_function_tokens"]:
            function_tokens = torch.tensor(self.function_tokens[0:1],
                                           dtype=torch.long)
            function_tokens = function_tokens.repeat(2, 1, 1)
        else:
            function_tokens = torch.tensor(self.function_tokens[[dataset_idx_1, dataset_idx_2]],
                                           dtype=torch.long)

        if self.t_hdf5.attrs["fixed_residue_annotation_tokens"]:
            residue_annotation_tokens = torch.tensor(self.residue_annotation_tokens[0:1],
                                                     dtype=torch.long)
            residue_annotation_tokens = residue_annotation_tokens.repeat(2, 1, 1)
        else:
            residue_annotation_tokens = torch.tensor(self.residue_annotation_tokens[[dataset_idx_1, dataset_idx_2]],
                                                     dtype=torch.long)

        if self.t_hdf5.attrs["fixed_bb_coords"]:
            bb_coords = torch.tensor(self.bb_coords[0:1], dtype=torch.float)
            bb_coords = bb_coords.repeat(2, 1, 1, 1)
        else:
            bb_coords = torch.tensor(self.bb_coords[[dataset_idx_1, dataset_idx_2]],
                                     dtype=torch.float)


        logp_mask = (masked_sequence_tokens == self.mask_token_sequence)


        return (masked_sequence_tokens,
                structural_tokens,
                average_plddt,
                per_res_plddt,
                ss8_tokens,
                sasa_tokens,
                function_tokens,
                residue_annotation_tokens,
                sequence_id,
                bb_coords,
                unmasked_sequence_tokens,
                logp_mask,
                energies,
                ref_logps)

