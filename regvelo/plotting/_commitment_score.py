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
    """Compute and plot cell fate commitment scores based on fate probabilities. 
    
    Parameters
    ----------
    adata
        Dataset containing fate probabilities. Original dataset or perturbed dataset.
    lineage_key
        The key in .obsm that stores the fate probabilities.
    **kwargs
        Additional keyword arguments passed to `scanpy.pl.umap`.
        
    Returns
    -------
    None
        Modifies `adata.obs["commitment_score"]` and plots a UMAP.
    """

    if lineage_key not in adata.obsm:
        raise KeyError(f"Key '{lineage_key}' not found in `adata.obsm`.")

    p = pd.DataFrame(adata.obsm[lineage_key], columns=adata.obsm[lineage_key].names.tolist())
    score = calculate_entropy(p)
    adata.obs["commitment_score"] = np.array(score)

    sc.pl.umap(
        adata,
        color="commitment_score",
        **kwargs
    )

