import torch
import pandas as pd
import numpy as np

from anndata import AnnData
from scvelo import logging as logg
import os, shutil

from .._model import REGVELOVI
from ._utils import get_list_name
from ._TFScanning_func import TFScanning_func


def TFscreening(
    adata: AnnData, 
    prior_graph: torch.Tensor,
    lam: int = 1,
    lam2: int = 0,
    soft_constraint: bool = True,
    TF_list: str | list[str] | dict[str, list[str]] | pd.Series = None,
    cluster_label : str | None = None,
    terminal_states : str | list[str] | dict[str, list[str]] | pd.Series = None,
    KO_list : str | list[str] | dict[str, list[str]] | pd.Series = None,
    n_states : int | list[int] = 8,
    cutoff : float = 1e-3,
    max_nruns : float = 5,
    method : str = "likelihood",
    dir : str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""Perform repeated in silico TF regulon knock-out screening.

    Parameters
    ----------
    adata
        Annotated data matrix.
    prior_graph
        A prior graph for RegVelo inference.
    lam
        Regularization parameter for controling the strengths of adding prior knowledge.
    lam2
        Regularization parameter for controling the strengths of L1 regularization to the Jacobian matrix.
    soft_constraint
        Apply soft constraint mode RegVelo.
    TF_list
        The TF list used for RegVelo inference.
    cluster_label
        Key in `adata.obs` to associate names and colors with :attr:`terminal_states`.
    terminal_states
        subset of :attr:`macrostates`.
    KO_list
        List of TF names or combinations (e.g., ["geneA", "geneB_geneC"]).
    n_states
        Number of macrostates to compute.
    cutoff
        Threshold to zero out TF-target links during knock-out.
    max_nruns
        Maximum number of runs, soft constrainted RegVelo model need to have repeat runs to get stable perturbation results.
        Set to 1 if ``soft_constraint=False``.
    method
        Method to quantify perturbation effects
    dir
        Directory to store intermediate model files and results.

    Returns
    -------
    
    - DataFrame containing the coefficients of the perturbation effects.
    - DataFrame containing the p-values of the perturbation effects.
    """

    if soft_constraint is not True:
        max_nruns = 1
    
    if dir is None:
        dir = os.getcwd()
    
    # Ensure the output directory exists
    output_dir = os.path.join(dir, "perturb_repeat_runs")
    os.makedirs(output_dir, exist_ok=True)

    for nrun in range(max_nruns):
        logg.info("training model...")
        REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
        reg_vae = REGVELOVI(adata, W=prior_graph, regulators=TF_list, soft_constraint=soft_constraint, lam=lam, lam2=lam2)
        reg_vae.train()
        
        logg.info("save model...")
        model_name = f"rgv_model_{nrun}"
        coef_name = f"coef_{nrun}.csv"
        pval_name = f"pval_{nrun}.csv"
        
        model = os.path.join(output_dir, model_name)
        coef_save = os.path.join(output_dir, coef_name)
        pval_save = os.path.join(output_dir, pval_name)
        
        reg_vae.save(model)

        logg.info("inferring perturbation...")
        while True:
            try:
                perturb_screening = TFScanning_func(model, adata, cluster_label, terminal_states, KO_list, n_states, cutoff, method)
                
                coef = pd.DataFrame(np.array(perturb_screening['coefficient']))
                coef.index = perturb_screening['TF']
                coef.columns = get_list_name(perturb_screening['coefficient'][0])

                pval = pd.DataFrame(np.array(perturb_screening['pvalue']))
                pval.index = perturb_screening['TF']
                pval.columns = get_list_name(perturb_screening['pvalue'][0])

                # Handle NaN rows
                rows_with_nan = coef.isna().any(axis=1)
                coef.loc[rows_with_nan, :] = np.nan
                pval.loc[rows_with_nan, :] = np.nan

                coef.to_csv(coef_save)
                pval.to_csv(pval_save)

                break
            except Exception as e:
                # Catch the exception and retry training
                print(f"Perturbation screening encountered an error: {e}, retraining model...")
                shutil.rmtree(model, ignore_errors=True)
                REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
                reg_vae = REGVELOVI(adata, W=prior_graph, regulators=TF_list, soft_constraint=soft_constraint, lam=lam, lam2=lam2)
                reg_vae.train()

                logg.info("save model...")
                reg_vae.save(model)

    # Read all repeat run results
    pert_coef = []
    pert_p = []
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if filename.startswith("coef_") and filename.endswith(".csv"):
            df = pd.read_csv(file_path, index_col=0)
            pert_coef.append(df)
        elif filename.startswith("pval_") and filename.endswith(".csv"):
            df = pd.read_csv(file_path, index_col=0)
            pert_p.append(df)
    
    # Combine DataFrames
    coef = pd.concat(pert_coef).groupby(level=0).mean()
    pval = pd.concat(pert_p).groupby(level=0).mean()

    return coef, pval

