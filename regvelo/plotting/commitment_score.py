import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from typing import Any

from ..metrics._utils import calculate_entropy


def commitment_score(
        adata : AnnData, 
        lineage_key : str = "lineages_fwd",
        **kwargs : Any
        ) -> None:
    """
    Compute and plot cell fate commitment scores based on fate probabilities. 
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing cell fate probabilities in `adata.obsm`.
    lineage_key : str, optional, default: "lineages_fwd"
        Key in `adata.obsm` that stores the cell fate probabilities (default: "lineages_fwd").
    **kwargs : Any
        Additional keyword arguments passed to `scanpy.pl.umap`.
    Raises
    ------
    KeyError
        If the specified `lineage_key` is not found in `adata.obsm`.
    Returns
    -------
    None
        Modifies `adata.obs["commitment_score"]` and plots a UMAP.
    """
    if lineage_key not in adata.obsm:
        raise KeyError(f"'{lineage_key}' not found in `adata.obsm`.")

    p = pd.DataFrame(adata.obsm[lineage_key], columns=adata.obsm[lineage_key].names.tolist())
    score = calculate_entropy(p)
    adata.obs["commitment_score"] = np.array(score)

    sc.pl.umap(
        adata,
        color="commitment_score",
        **kwargs
    )

