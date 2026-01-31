import einops
import torch
import torch.nn as nn
from functools import partial

from pera.nn.layers import (RegressionHead, TransformerStack)
from pera.nn.utils import build_affine3d_from_coordinates

def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(
        v_min, v_max, n_bins, device=values.device, dtype=values.dtype
    )
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-(z**2))


class EncodeInputs(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.sequence_embed = nn.Embedding(64, d_model)

        self.plddt_projection = nn.Linear(16, d_model)
        self.structure_per_res_plddt_projection = nn.Linear(16, d_model)

        self.structure_tokens_embed = nn.Embedding(4096 + 5, d_model)

        self.ss8_embed = nn.Embedding(8 + 3, d_model)
        self.sasa_embed = nn.Embedding(16 + 3, d_model)

        self.function_embed = nn.ModuleList([nn.Embedding(260, d_model // 8, padding_idx=0)
                                             for _ in range(8)])

        self.residue_embed = nn.EmbeddingBag(1478,
                                             d_model, mode="sum",
                                             padding_idx=0)

    def forward(self,
                sequence_tokens,
                structure_tokens,
                average_plddt,
                per_res_plddt,
                ss8_tokens,
                sasa_tokens,
                function_tokens,
                residue_annotation_tokens):

        sequence_embed = self.sequence_embed(sequence_tokens)

        rbf_16_fn = partial(rbf,
                            v_min=0.0,
                            v_max=1.0,
                            n_bins=16)

        plddt_embed = self.plddt_projection(rbf_16_fn(average_plddt))
        structure_per_res_plddt = self.structure_per_res_plddt_projection(
            rbf_16_fn(per_res_plddt))

        # Structure + "structural features" embeds
        structure_embed = self.structure_tokens_embed(structure_tokens)
        ss8_embed = self.ss8_embed(ss8_tokens)
        sasa_embed = self.sasa_embed(sasa_tokens)

        # "Functional" features embeds
        function_embed = torch.cat([embed_fn(funcs)
                                    for embed_fn, funcs in zip(self.function_embed, function_tokens.unbind(-1))],
                                   dim=-1)

        # Residue embeds
        B, L, N = residue_annotation_tokens.shape
        residue_embed = self.residue_embed(einops.rearrange(residue_annotation_tokens, 
                                                            "B L N -> (B L) N", B=B, L=L, N=N))
        residue_embed = einops.rearrange(residue_embed, 
                                         "(B L) D -> B L D", B=B, L=L)
        

        total_embed = (sequence_embed
                       + plddt_embed.unsqueeze(-2)
                       + structure_per_res_plddt
                       + structure_embed
                       + ss8_embed
                       + sasa_embed
                       + function_embed
                       + residue_embed)
        return total_embed


class OutputHeads(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.sequence_head = RegressionHead(d_model, 64)
        self.structure_head = RegressionHead(d_model, 4096)
        self.ss8_head = RegressionHead(d_model, 8 + 3)
        self.sasa_head = RegressionHead(d_model, 16 + 3)
        self.function_head = RegressionHead(d_model, 260 * 8)
        self.residue_head = RegressionHead(d_model, 1478)

    def forward(self, x: torch.Tensor, embed: torch.Tensor):
        sequence_logits = self.sequence_head(x)
        structure_logits = self.structure_head(x)
        secondary_structure_logits = self.ss8_head(x)
        sasa_logits = self.sasa_head(x)
        function_logits = self.function_head(x)
        function_logits = einops.rearrange(
            function_logits,
            "... (k v) -> ... k v",
            k=8,
        )

        residue_logits = self.residue_head(x)

        return dict(sequence_logits=sequence_logits,
                    structure_logits=structure_logits,
                    secondary_structure_logits=secondary_structure_logits,
                    sasa_logits=sasa_logits,
                    function_logits=function_logits,
                    residue_logits=residue_logits,
                    embeddings=embed)


class ESM3(nn.Module):
    """
    ESM3 model implementation.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads in the transformer layers.
        v_heads (int): The number of attention heads in the variational transformer layers.
        n_layers (int): The number of transformer layers.
    """

    def __init__(self,
                 d_model,
                 unified_transformer_args):
        super().__init__()
        self.encoder = EncodeInputs(d_model)
        self.transformer = TransformerStack(d_model,
                                            **unified_transformer_args)
        self.output_heads = OutputHeads(d_model)

    def forward(self,
                sequence_tokens,
                structure_tokens,
                average_plddt,
                per_res_plddt,
                ss8_tokens,
                sasa_tokens,
                function_tokens,
                residue_annotation_tokens,
                sequence_id,
                bb_coords,
                sigma=None):
        """
        Performs forward pass through the ESM3 model. Check utils to see how to tokenize inputs from raw data.

        Args:
            sequence_tokens (torch.Tensor, optional): The amino acid tokens.
            structure_tokens (torch.Tensor, optional): The structure tokens.
            ss8_tokens (torch.Tensor, optional): The secondary structure tokens.
            sasa_tokens (torch.Tensor, optional): The solvent accessible surface area tokens.
            function_tokens (torch.Tensor, optional): The function tokens.
            residue_annotation_tokens (torch.Tensor, optional): The residue annotation tokens.
            average_plddt (torch.Tensor, optional): The average plddt across the entire sequence.
            per_res_plddt (torch.Tensor, optional): The per residue plddt, if you want to specify exact plddts, use this,
                otherwise, use average_plddt.
            structure_coords (torch.Tensor, optional): The structure coordinates, in the form of (B, L, 3, 3).
            chain_id (torch.Tensor, optional): The chain ID
            sequence_id (torch.Tensor, optional): The sequence ID.

        Returns:
            ESMOutput: The output of the ESM3 model.

        Raises:
            ValueError: If at least one of the inputs is None.

        """
        affine, affine_mask = build_affine3d_from_coordinates(bb_coords)

        x = self.encoder(sequence_tokens,
                         structure_tokens,
                         average_plddt,
                         per_res_plddt,
                         ss8_tokens,
                         sasa_tokens,
                         function_tokens,
                         residue_annotation_tokens)
        
        if sigma is not None:
            noise = torch.randn_like(x) * sigma  
            x = x + noise
    
        x, embedding = self.transformer(x, sequence_id, affine, affine_mask)
        output = self.output_heads(x, embedding)
        return output