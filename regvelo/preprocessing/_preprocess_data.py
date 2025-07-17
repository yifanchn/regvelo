# the folloing code is adapted from velovi
import numpy as np
import scvelo as scv
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(
    adata: AnnData,
    spliced_layer: str = "Ms",
    unspliced_layer: str = "Mu",
    min_max_scale: bool = True,
    filter_on_r2: bool = True,
) -> AnnData:
    r"""Preprocess an AnnData object.

    This function optionally applies min-max scaling to the spliced and unspliced layers,
    and filters genes based on the velocity regression results (velocity_r2 and gamma).

    Parameters
    ----------
    adata
        Annotated data object.
    spliced_layer
        Name of the spliced layer.
    unspliced_layer
        Name of the unspliced layer.
    min_max_scale
        Whether to apply min-max scaling to the spliced and unspliced layers.
    filter_on_r2
        Whether to filter genes based on the velocity regression results.

    Returns
    -------
    Preprocessed annotated data object.
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