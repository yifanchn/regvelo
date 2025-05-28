# the folloing code is adapted from velovi
from typing import Optional

import numpy as np
import scvelo as scv
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(
    adata: AnnData,
    spliced_layer: Optional[str] = "Ms",
    unspliced_layer: Optional[str] = "Mu",
    min_max_scale: bool = True,
    filter_on_r2: bool = True,
    ) -> AnnData:
    """
    Preprocess an AnnData object.

    This function optionally applies min-max scaling to the spliced and unspliced layers,
    and filters genes based on the velocity regression results (velocity_r2 and gamma).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    spliced_layer : str, optional
        Key in `adata.layers` corresponding to the spliced layer (default: "Ms").
    unspliced_layer : str, optional
        Key in `adata.layers` corresponding to the unspliced layer (default: "Mu").
    min_max_scale : bool, optional
        Whether to apply min-max scaling to the spliced and unspliced layers (default: True).
    filter_on_r2 : bool, optional
        Whether to filter genes based on the velocity regression results (default: True).

    Returns
    -------
    AnnData
        Preprocessed AnnData object.
    """
    if min_max_scale:
        scaler = MinMaxScaler()
        adata.layers[spliced_layer] = scaler.fit_transform(adata.layers[spliced_layer])

        scaler = MinMaxScaler()
        adata.layers[unspliced_layer] = scaler.fit_transform(
            adata.layers[unspliced_layer]
        )

    if filter_on_r2:
        scv.tl.velocity(adata, mode="deterministic")

        adata = adata[
            :, np.logical_and(adata.var.velocity_r2 > 0, adata.var.velocity_gamma > 0)
        ].copy()
        adata = adata[:, adata.var.velocity_genes].copy()

    return adata