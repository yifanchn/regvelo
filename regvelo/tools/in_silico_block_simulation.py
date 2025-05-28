import torch
from anndata import AnnData

from .._model import REGVELOVI

def in_silico_block_simulation(
        model : str,
        adata : AnnData,
        TF : str,
        effects : float = 0.0,
        cutoff : float = 1e-3,
        customized_GRN : torch.Tensor = None
        ) -> tuple[AnnData, REGVELOVI]:
    """ 
    Perform an in silico transcription factor (TF) regulon knock-out by modifying the gene regulatory network (GRN)
    in a trained RegVelo model and simulating its effect.

    Parameters
    ----------
    model : str
        Path to the saved RegVelo model.
    adata : AnnData
        Annotated data matrix.
    TF : str
        Transcription factor to be knocked out (its regulon will be silenced).
    effect : float, optional (default: 0.0)
        Value used to replace the weights of affected GRN links (e.g., 0 for knockout).
    cutoff : float, optional (default: 1e-3)
        Threshold to determine which links in the GRN are considered active and should be muted.
    customized_GRN : toch.Tensor, optional (default: None)
        A custom perturbed GRN weight matrix to directly replace the default GRN.

    Returns
    -------
    tuple
        - adata_target_perturb : AnnData
            AnnData object with simulated outputs after perturbation.
        - reg_vae_perturb : REGVELOVI
            The perturbed RegVelo model instance.
    """
    reg_vae_perturb = REGVELOVI.load(model,adata)
    perturb_GRN = reg_vae_perturb.module.v_encoder.fc1.weight.detach().clone()

    if customized_GRN is None:
        perturb_GRN[(perturb_GRN[:,[i == TF for i in adata.var.index]].abs()>cutoff).cpu().numpy().reshape(-1),[i == TF for i in adata.var.index]] = effects
        reg_vae_perturb.module.v_encoder.fc1.weight.data = perturb_GRN
    else:
        device = perturb_GRN.device
        customized_GRN =  customized_GRN.to(device)
        reg_vae_perturb.module.v_encoder.fc1.weight.data = customized_GRN
    
    adata_target_perturb = reg_vae_perturb.add_regvelo_outputs_to_adata(adata = adata)
    
    return adata_target_perturb, reg_vae_perturb