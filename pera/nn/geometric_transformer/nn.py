from pera.models.esm import ESM3

class GeometricTransformer(ESM3):
    def __init__(self, dim_model, unified_transformer_args,
                 struc_token_info, residue_token_info, sasa_token_info, 
                 sec_struct_token_info, res_annot_token_info):
        super().__init__(d_model=dim_model, 
                         unified_transformer_args=unified_transformer_args)
