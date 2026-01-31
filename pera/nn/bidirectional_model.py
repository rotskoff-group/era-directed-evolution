import pera.nn
import lightning as L
import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import pl_bolts


class BidirectionalModel(L.LightningModule):
    def __init__(self, nn_model_name, nn_model_args,
                 eval_type, beta, gamma,
                 better_energy,
                 sampling_temperature,
                 optimizer, optimizer_args,
                 lr_scheduler, lr_scheduler_args,
                 interval,
                 monitor, sync_dist, on_step
                 ):
        super().__init__()
        nn = getattr(pera.nn, nn_model_name)
        self.nn = nn(**nn_model_args)
        self._shared_eval = getattr(self, f"_shared_eval_{eval_type}")
        self.loss_fn = CrossEntropyLoss(reduction="none")
        self.save_hyperparameters()

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def on_test_model_eval(self, *args, **kwargs):
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def set_special_token_info(self, mask_token_sequence, bos_token_sequence, 
                               eos_token_sequence, pad_token_sequence):
        self.mask_token_sequence = mask_token_sequence
        self.bos_token_sequence = bos_token_sequence
        self.eos_token_sequence = eos_token_sequence
        self.pad_token_sequence = pad_token_sequence

    def _get_unmask_indices(self, masked_indices, sequence_length):
        num_masked = masked_indices.shape[0]
        if num_masked == 0:
            return np.array([])

        if np.random.rand() < 0.8:
            # Draw from Beta(3, 9) distribution
            unmask_rate = np.random.beta(3, 9)
        else:
            # Draw from a linear distribution
            unmask_rate = np.random.uniform(0, 1)


        unmask_rate = 1.0
        num_to_unmask = int(unmask_rate * sequence_length)
        num_to_unmask = max(1, num_to_unmask)
        num_to_unmask = min(num_to_unmask, num_masked)
        unmask_indices = np.random.choice(masked_indices,
                                        num_to_unmask,
                                        replace=False)
        return unmask_indices

    def forward(self, batch, perturb_logits=False, noise_std=0.1):
        """Does not currently add padding
        """
        (masked_sequence_tokens_batch, structure_tokens_batch,
         average_plddt_batch, per_res_plddt_batch, ss8_tokens_batch,
         sasa_tokens_batch, function_tokens_batch, residue_annotation_tokens_batch,
         bb_coords_batch, sequence_id_batch) = batch

        device = structure_tokens_batch.device

        batch_size = masked_sequence_tokens_batch.shape[0]
        padded_sequence_length = masked_sequence_tokens_batch.shape[1]

        sequence_masked_indices = (masked_sequence_tokens_batch
                                   == self.mask_token_sequence)

        # True sequence length without padding, masking, BOS and EOS
        sequence_non_special_indices = ((masked_sequence_tokens_batch != self.bos_token_sequence)
                                        & (masked_sequence_tokens_batch != self.eos_token_sequence)
                                        & (masked_sequence_tokens_batch != self.pad_token_sequence))
        

        sequence_length = sequence_non_special_indices.sum(-1)

        masked_indices = [torch.where(sequence_masked_indices[i])[0].detach().cpu().numpy()
                          for i in range(batch_size)]

        unmasked_indices = [(self._get_unmask_indices(masked_indices[i],
                                                      sequence_length[i])
                             + padded_sequence_length*i)
                            for i in range(batch_size)]
        unmasked_indices = torch.tensor(np.concatenate(unmasked_indices),
                                        device=device).long()

        while unmasked_indices.shape[0] > 0:

            logits = self.nn(sequence_tokens=masked_sequence_tokens_batch,
                             structure_tokens=structure_tokens_batch,
                             average_plddt=average_plddt_batch,
                             per_res_plddt=per_res_plddt_batch,
                             ss8_tokens=ss8_tokens_batch,
                             sasa_tokens=sasa_tokens_batch,
                             function_tokens=function_tokens_batch,
                             residue_annotation_tokens=residue_annotation_tokens_batch,
                             sequence_id=sequence_id_batch,
                             bb_coords=bb_coords_batch)["sequence_logits"].detach()
            
            if perturb_logits:
                noise_std = noise_std  
                noise = torch.randn_like(logits) * noise_std
                logits = logits + noise

            masked_sequence_tokens_batch_flattened = masked_sequence_tokens_batch.flatten()
            logits_flattened = logits.view(-1, logits.shape[-1])


            mask_token_probs = nn.functional.softmax(logits_flattened[unmasked_indices]/self.hparams.sampling_temperature,
                                                     dim=-1)
            tokens = torch.multinomial(mask_token_probs, 1).squeeze(-1)

            masked_sequence_tokens_batch_flattened[unmasked_indices] = tokens

            masked_sequence_tokens_batch = masked_sequence_tokens_batch_flattened.view(-1,
                                                                                       masked_sequence_tokens_batch.shape[-1])

            rotamer_masked_indices = (masked_sequence_tokens_batch
                                      == self.mask_token_sequence)

            masked_indices = [torch.where(rotamer_masked_indices[i])[0].detach().cpu().numpy()
                              for i in range(batch_size)]
            unmasked_indices = [(self._get_unmask_indices(masked_indices[i],
                                                          sequence_length[i])
                                + padded_sequence_length*i)
                                for i in range(batch_size)]
            unmasked_indices = torch.tensor(np.concatenate(unmasked_indices),
                                            device=device).long()
        return masked_sequence_tokens_batch

    def _shared_eval_sft(self, batch, batch_idx, prefix):
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
         loss_mask) = batch
        
        print("IN SFT", prefix)

        pred_sequence = self.nn(sequence_tokens=masked_sequence_tokens,
                                structure_tokens=structure_tokens,
                                average_plddt=average_plddt,
                                per_res_plddt=per_res_plddt,
                                ss8_tokens=ss8_tokens,
                                sasa_tokens=sasa_tokens,
                                function_tokens=function_tokens,
                                residue_annotation_tokens=residue_annotation_tokens,
                                sequence_id=sequence_id,
                                bb_coords=bb_coords)["sequence_logits"]
        # (N, C, S) - pred_sequence
        pred_sequence = pred_sequence.transpose(1, 2)
        # (N, C)- rotamer_tokens
        # (N, C)
        loss = self.loss_fn(pred_sequence, unmasked_sequence_tokens)
        loss = (loss * loss_mask).sum(-1)/(loss_mask.sum(-1))
        loss = loss.mean()

        metrics = {f"{prefix}/CELoss": loss}

        self.log_dict(metrics, on_epoch=True,
                      on_step=self.hparams.on_step,
                      sync_dist=self.hparams.sync_dist,
                      batch_size=masked_sequence_tokens.shape[0])
        return loss

    def _shared_eval_era(self, batch, batch_idx, prefix):
        
        #Assumed to be (pair1_1, pair1_2, pair2_1, pair2_2, ...)
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
         ref_logps) = batch
        
        policy_logits = self.nn(sequence_tokens=masked_sequence_tokens,
                                structure_tokens=structure_tokens,
                                average_plddt=average_plddt,
                                per_res_plddt=per_res_plddt,
                                ss8_tokens=ss8_tokens,
                                sasa_tokens=sasa_tokens,
                                function_tokens=function_tokens,
                                residue_annotation_tokens=residue_annotation_tokens,
                                sequence_id=sequence_id,
                                bb_coords=bb_coords)["sequence_logits"]

        policy_logps = torch.nn.functional.log_softmax(policy_logits/self.hparams.sampling_temperature, dim=-1)
        policy_logps = torch.gather(policy_logps, dim=-1, 
                                    index=unmasked_sequence_tokens.unsqueeze(-1)).squeeze(-1)
        policy_logps = (policy_logps * logp_mask).sum(-1)

        beta_prime = (self.hparams.beta / (1 + self.hparams.gamma))
        gamma_prime = (self.hparams.gamma / (1 + self.hparams.gamma))

        energies = energies * beta_prime

        #Check this
        policy_logps_y1 = policy_logps.reshape(-1, 2)[:, 0]
        policy_logps_y2 = policy_logps.reshape(-1, 2)[:, 1]

        ref_logps_y1 = ref_logps.reshape(-1, 2)[:, 0]
        ref_logps_y2 = ref_logps.reshape(-1, 2)[:, 1]

        energies_y1 = energies.reshape(-1, 2)[:, 0]
        energies_y2 = energies.reshape(-1, 2)[:, 1]

        
        logp = nn.functional.logsigmoid(policy_logps_y2 - policy_logps_y1)
        logp_prime = nn.functional.logsigmoid(policy_logps_y1
                                              - policy_logps_y2)

        logp_star = nn.functional.logsigmoid(-(energies_y2 - energies_y1)
                                             + (gamma_prime * (ref_logps_y2 - ref_logps_y1)))
        logp_star_prime = nn.functional.logsigmoid(-(energies_y1 - energies_y2)
                                                   + (gamma_prime * (ref_logps_y1 - ref_logps_y2)))
        kl_loss = (torch.exp(logp_star) * (logp_star - logp)
                   + torch.exp(logp_star_prime) * (logp_star_prime - logp_prime))
        
        kl_loss = kl_loss.mean()

        metrics = {f"{prefix}/ERALoss": kl_loss}
        
        self.log_dict(metrics, on_epoch=True,
                      on_step=self.hparams.on_step,
                      sync_dist=self.hparams.sync_dist,
                      batch_size=masked_sequence_tokens.shape[0])

        return kl_loss 
    
    def _shared_eval_dpo(self, batch, batch_idx, prefix):
        
        #Assumed to be (pair1_1, pair1_2, pair2_1, pair2_2, ...)
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
         ref_logps) = batch
        
        policy_logits = self.nn(sequence_tokens=masked_sequence_tokens,
                                structure_tokens=structure_tokens,
                                average_plddt=average_plddt,
                                per_res_plddt=per_res_plddt,
                                ss8_tokens=ss8_tokens,
                                sasa_tokens=sasa_tokens,
                                function_tokens=function_tokens,
                                residue_annotation_tokens=residue_annotation_tokens,
                                sequence_id=sequence_id,
                                bb_coords=bb_coords)["sequence_logits"]

        policy_logps = torch.nn.functional.log_softmax(policy_logits/self.hparams.sampling_temperature, dim=-1)
        policy_logps = torch.gather(policy_logps, dim=-1, 
                                    index=unmasked_sequence_tokens.unsqueeze(-1)).squeeze(-1)
        policy_logps = (policy_logps * logp_mask).sum(-1)

        beta = self.hparams.beta


        policy_logps_y1 = policy_logps.reshape(-1, 2)[:, 0]
        policy_logps_y2 = policy_logps.reshape(-1, 2)[:, 1]

        ref_logps_y1 = ref_logps.reshape(-1, 2)[:, 0]
        ref_logps_y2 = ref_logps.reshape(-1, 2)[:, 1]

        energies_y1 = energies.reshape(-1, 2)[:, 0]
        energies_y2 = energies.reshape(-1, 2)[:, 1]

        
        # SI: want to be able to specify whether higher or lower energy is better
        if self.hparams.better_energy == "higher":
            y2_sign = (energies_y2 >= energies_y1).long()
        elif self.hparams.better_energy == "lower":
            y2_sign = (energies_y2 <= energies_y1).long()
        else:
            raise ValueError("better_energy must be 'higher' or 'lower'")
        y2_sign[y2_sign == 0] = -1
        y1_sign = -y2_sign

        pi_logratios = (y2_sign * policy_logps_y2
                        + y1_sign * policy_logps_y1)
        ref_logratios = (y2_sign * ref_logps_y2
                        + y1_sign * ref_logps_y1)
        
        loss = -nn.functional.logsigmoid(beta * (pi_logratios - ref_logratios))
        tot_loss = loss.mean()

        metrics = {f"{prefix}/DPOLoss": tot_loss}

        self.log_dict(
            metrics,
            on_epoch=True,
            on_step=self.hparams.on_step,
            sync_dist=self.hparams.sync_dist,
            batch_size=masked_sequence_tokens.shape[0],
        )

        return tot_loss

    def training_step(self, batch, batch_idx):
        assert self.nn.training
        loss = self._shared_eval(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        assert not self.nn.training
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        assert not self.nn.training
        self._shared_eval(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        assert not self.nn.training
        tgt_input = self.forward(batch)
        return tgt_input

    def configure_optimizers(self):
        u_optimizer = getattr(torch.optim,
                                self.hparams.optimizer)
        u_optimizer = u_optimizer(self.nn.parameters(),
                                **self.hparams.optimizer_args)
        if self.hparams.lr_scheduler is None:
            to_return = {"optimizer": u_optimizer}
        else:
            if self.hparams.lr_scheduler == "LinearWarmupCosineAnnealingLR":
                u_scheduler = getattr(pl_bolts.optimizers.lr_scheduler,
                                      self.hparams.lr_scheduler)
            else:
                u_scheduler = getattr(torch.optim.lr_scheduler,
                                      self.hparams.lr_scheduler)
            u_scheduler = u_scheduler(u_optimizer,
                                      **self.hparams.lr_scheduler_args)
            lr_scheduler_config = {"scheduler": u_scheduler,
                                   "interval": self.hparams.interval,
                                   "monitor": self.hparams.monitor}
            to_return = {"optimizer": u_optimizer,
                         "lr_scheduler": lr_scheduler_config}
        return to_return

    def load_model_from_ckpt(self, filename):
        print(f"Loading model from {filename}")
        model_weights = torch.load(filename, map_location=torch.device("cpu"))["state_dict"]
        model_weights = model_weights = {(k[3:] if k.startswith("nn.") else k): v
                                         for k, v in model_weights.items()}
        self.nn.load_state_dict(model_weights)



def sample_components_from_bidirectional_transformer(transformer_model,
                                                     masked_sequence_tokens, structure_tokens,
                                                     average_plddt, per_res_plddt, ss8_tokens,
                                                     sasa_tokens, function_tokens, residue_annotation_tokens,
                                                     bb_coords, sequence_id,
                                                     mask_token_sequence, bos_token_sequence,
                                                     eos_token_sequence, pad_token_sequence,
                                                     inference_batch_size=128):
    """Samples components from the transformer model
    """
    masked_sequence_tokens = masked_sequence_tokens.clone()
    transformer_model.nn.eval()
    assert not transformer_model.nn.training
    transformer_model.set_special_token_info(mask_token_sequence=mask_token_sequence,
                                             bos_token_sequence=bos_token_sequence,
                                             eos_token_sequence=eos_token_sequence,
                                             pad_token_sequence=pad_token_sequence)

    num_batches = ((masked_sequence_tokens.shape[0] + inference_batch_size - 1)
                   // inference_batch_size)

    unmasked_rotamer_tokens = []
    for batch in range(num_batches):
        masked_sequence_tokens_batch = masked_sequence_tokens[batch * inference_batch_size:
                                                              (batch+1)*inference_batch_size]
        structure_tokens_batch = structure_tokens[batch * inference_batch_size:
                                                  (batch+1)*inference_batch_size]
        average_plddt_batch = average_plddt[batch * inference_batch_size:
                                            (batch+1)*inference_batch_size]
        per_res_plddt_batch = per_res_plddt[batch * inference_batch_size:
                                            (batch+1)*inference_batch_size]
        ss8_tokens_batch = ss8_tokens[batch * inference_batch_size:
                                      (batch+1)*inference_batch_size]
        sasa_tokens_batch = sasa_tokens[batch * inference_batch_size:
                                        (batch+1)*inference_batch_size]
        function_tokens_batch = function_tokens[batch * inference_batch_size:
                                                (batch+1)*inference_batch_size]
        residue_annotation_tokens_batch = residue_annotation_tokens[batch * inference_batch_size:
                                                                    (batch+1)*inference_batch_size]
        bb_coords_batch = bb_coords[batch*inference_batch_size:
                                    (batch+1)*inference_batch_size]
        sequence_id_batch = sequence_id[batch*inference_batch_size:
                                        (batch+1)*inference_batch_size]

        batch = (masked_sequence_tokens_batch, structure_tokens_batch,
                 average_plddt_batch, per_res_plddt_batch, ss8_tokens_batch,
                 sasa_tokens_batch, function_tokens_batch, residue_annotation_tokens_batch,
                 bb_coords_batch, sequence_id_batch)

        unmasked_rotamer_tokens_batch = transformer_model.forward(batch)
        unmasked_rotamer_tokens.append(unmasked_rotamer_tokens_batch)
    unmasked_rotamer_tokens = torch.cat(unmasked_rotamer_tokens, dim=0)

    return unmasked_rotamer_tokens

def sample_perturbations(transformer_model, batch, noise_std=0.1, sampling_temperature=1.0):
    """
    Samples sequences by applying Gaussian perturbations to logits.
    
    Args:
        batch: The input batch containing the sequences and other features.
        noise_std: Standard deviation of the Gaussian perturbation.
        sampling_temperature: Temperature for sampling from the logits.
    
    Returns:
        Updated sequences after sampling from perturbed logits.
    """
    (masked_sequence_tokens_batch, structure_tokens_batch,
     average_plddt_batch, per_res_plddt_batch, ss8_tokens_batch,
     sasa_tokens_batch, function_tokens_batch, residue_annotation_tokens_batch,
     bb_coords_batch, sequence_id_batch, mask_token_sequence) = batch

    device = structure_tokens_batch.device

    logits = transformer_model.nn(sequence_tokens=masked_sequence_tokens_batch,
                     structure_tokens=structure_tokens_batch,
                     average_plddt=average_plddt_batch,
                     per_res_plddt=per_res_plddt_batch,
                     ss8_tokens=ss8_tokens_batch,
                     sasa_tokens=sasa_tokens_batch,
                     function_tokens=function_tokens_batch,
                     residue_annotation_tokens=residue_annotation_tokens_batch,
                     sequence_id=sequence_id_batch,
                     bb_coords=bb_coords_batch)["sequence_logits"].detach()

    noise = torch.randn_like(logits) * noise_std
    perturbed_logits = logits + noise

    masked_sequence_tokens_batch_flattened = masked_sequence_tokens_batch.flatten()
    logits_flattened = perturbed_logits.view(-1, perturbed_logits.shape[-1])
    
    sequence_masked_indices = (masked_sequence_tokens_batch
                               == mask_token_sequence)
    unmasked_indices = torch.where(sequence_masked_indices.flatten())[0]

    mask_token_probs = nn.functional.softmax(logits_flattened[unmasked_indices] / sampling_temperature, dim=-1)
    sampled_tokens = torch.multinomial(mask_token_probs, 1).squeeze(-1)

    masked_sequence_tokens_batch_flattened[unmasked_indices] = sampled_tokens
    updated_sequences = masked_sequence_tokens_batch_flattened.view(*masked_sequence_tokens_batch.shape)

    return updated_sequences, perturbed_logits

def sample_embedding_perturbations(transformer_model, batch, noise_std=0.1, sampling_temperature=1.0):
    """
    Samples sequences by applying Gaussian perturbations to logits.
    
    Args:
        batch: The input batch containing the sequences and other features.
        noise_std: Standard deviation of the Gaussian perturbation.
        sampling_temperature: Temperature for sampling from the logits.
    
    Returns:
        Updated sequences after sampling from perturbed logits.
    """
    (masked_sequence_tokens_batch, structure_tokens_batch,
     average_plddt_batch, per_res_plddt_batch, ss8_tokens_batch,
     sasa_tokens_batch, function_tokens_batch, residue_annotation_tokens_batch,
     bb_coords_batch, sequence_id_batch, mask_token_sequence) = batch

    device = structure_tokens_batch.device

    logits = transformer_model.nn(sequence_tokens=masked_sequence_tokens_batch,
                     structure_tokens=structure_tokens_batch,
                     average_plddt=average_plddt_batch,
                     per_res_plddt=per_res_plddt_batch,
                     ss8_tokens=ss8_tokens_batch,
                     sasa_tokens=sasa_tokens_batch,
                     function_tokens=function_tokens_batch,
                     residue_annotation_tokens=residue_annotation_tokens_batch,
                     sequence_id=sequence_id_batch,
                     bb_coords=bb_coords_batch, 
                     sigma=noise_std)["sequence_logits"].detach()


    masked_sequence_tokens_batch_flattened = masked_sequence_tokens_batch.flatten()
    logits_flattened = logits.view(-1, logits.shape[-1])
    
    sequence_masked_indices = (masked_sequence_tokens_batch
                               == mask_token_sequence)
    unmasked_indices = torch.where(sequence_masked_indices.flatten())[0]

    mask_token_probs = nn.functional.softmax(logits_flattened[unmasked_indices] / sampling_temperature, dim=-1)
    sampled_tokens = torch.multinomial(mask_token_probs, 1).squeeze(-1)

    masked_sequence_tokens_batch_flattened[unmasked_indices] = sampled_tokens
    updated_sequences = masked_sequence_tokens_batch_flattened.view(*masked_sequence_tokens_batch.shape)

    return updated_sequences, logits
