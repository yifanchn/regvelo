from pathlib import Path
from typing import Optional, Union

import pandas as pd

from scanpy import read

from scvelo.core import cleanup
from scvelo.read_load import load

url_adata = "https://drive.google.com/uc?id=1Nzq1F6dGw-nR9lhRLfZdHOG7dcYq7P0i&export=download"
url_grn = "https://drive.google.com/uc?id=1ci_gCwdgGlZ0xSn6gSa_-LlIl9-aDa1c&export=download/"
url_adata_murine_processed = "https://drive.usercontent.google.com/download?id=19bNQfW3jMKEEjpjNdUkVd7KDTjJfqxa5&export=download&authuser=1&confirm=t&uuid=4fdf3051-229b-4ce2-b644-cb390424570a&at=APcmpoxgcuZ5r6m6Fb6N_2Og6tEO:1745354679573"
url_adata_murine_normalized = "https://drive.usercontent.google.com/download?id=1xy2FNYi6Y2o_DzXjRmmCtARjoZ97Ro_w&export=download&authuser=1&confirm=t&uuid=12cf5d23-f549-48d9-b7ec-95411a58589f&at=APcmpoyexgouf243lNygF9yRUkmi:1745997349046"
url_adata_murine_velocyto = "https://drive.usercontent.google.com/download?id=18Bhtb7ruoUxpNt8WMYSaJ1RyoiHOCEjd&export=download&authuser=1&confirm=t&uuid=ecc42202-bc82-4ab1-b2c3-bfc31c99f0df&at=APcmpozsh6tBzkv8NSIZW0VipDJa:1745997422108"


def zebrafish_nc(file_path: Union[str, Path] = "data/zebrafish_nc/adata_zebrafish_preprocessed.h5ad"):
    """Zebrafish neural crest cells.

    Single cell RNA-seq datasets of zebrafish neural crest cell development across 
    seven distinct time points using ultra-deep Smart-seq3 technique.

    There are four distinct phases of NC cell development: 1) specification at the NPB, 2) epithelial-to-mesenchymal
    transition (EMT) from the neural tube, 3) migration throughout the periphery, 4) differentiation into distinct cell types

    Arguments:
    ---------
    file_path
        Path where to save dataset and read it from.

    Returns
    -------
    Returns `adata` object
    """
    adata = read(file_path, backup_url=url_adata, sparse=True, cache=True)
    return adata

def zebrafish_grn(file_path: Union[str, Path] = "data/zebrafish_nc/prior_GRN.csv"):
    """Zebrafish neural crest cells.

    Single cell RNA-seq datasets of zebrafish neural crest cell development across 
    seven distinct time points using ultra-deep Smart-seq3 technique.

    There are four distinct phases of NC cell development: 1) specification at the NPB, 2) epithelial-to-mesenchymal
    transition (EMT) from the neural tube, 3) migration throughout the periphery, 4) differentiation into distinct cell types

    Arguments:
    ---------
    file_path
        Path where to save dataset and read it from.

    Returns
    -------
    Returns `adata` object
    """
    grn = pd.read_csv(url_grn, index_col = 0)
    grn.to_csv(file_path)
    return grn

def murine_nc(data_type: str = "preprocessed"):
    """
    Mouse neural crest cells.

    Single-cell RNA-seq datasets of mouse neural crest cell development, 
    subset from Qiu, Chengxiang et al.

    The gene regulatory network (GRN) is saved in `adata.uns["skeleton"]`, 
    which is learned via pySCENIC.

    Parameters
    ----------
    data_type : str
        Which version of the dataset to load. Must be one of:
        - "preprocessed"
        - "normalized"
        - "velocyto"

    Returns
    -------
    AnnData
        Annotated data matrix (an `AnnData` object).
    """
    valid_types = ["preprocessed", "normalized", "velocyto"]
    if data_type not in valid_types:
        raise ValueError(f"Invalid data_type: '{data_type}'. Must be one of {valid_types}.")

    file_path = ["data/murine_nc/adata_preprocessed.h5ad","data/murine_nc/adata_gex_normalized.h5ad","data/murine_nc/adata_velocity.h5ad"]

    if data_type == "preprocessed":
        adata = read(file_path[0], backup_url=url_adata_murine_processed, sparse=True, cache=True)
    elif data_type == "normalized":
        adata = read(file_path[1], backup_url=url_adata_murine_normalized, sparse=True, cache=True)
    elif data_type == "velocyto":
        adata = read(file_path[2], backup_url=url_adata_murine_velocyto, sparse=True, cache=True)

    return adata