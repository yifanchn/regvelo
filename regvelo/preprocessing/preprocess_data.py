
from pathlib import Path
from typing import Optional, Union
from urllib.request import urlretrieve

import torch
import numpy as np
import pandas as pd
import scvelo as scv
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

def preprocess_data(
    adata : AnnData,
    spliced_layer : Optional[str] = "Ms",
    unspliced_layer : Optional[str] = "Mu",
    ) -> AnnData:
    """
    Preprocess data.

    This function removes poorly detected genes and minmax scales the data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    spliced_layer
        Name of the spliced layer.
    unspliced_layer
        Name of the unspliced layer.
    min_max_scale
        Min-max scale spliced and unspliced
    filter_on_r2
        Filter out genes according to linear regression fit

    Returns
    -------
    Preprocessed adata.
    """
    scaler = MinMaxScaler()
    adata.layers[spliced_layer] = scaler.fit_transform(adata.layers[spliced_layer])

    scaler = MinMaxScaler()
    adata.layers[unspliced_layer] = scaler.fit_transform(
        adata.layers[unspliced_layer]
    )

    return adata