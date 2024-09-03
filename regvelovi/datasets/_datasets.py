from pathlib import Path
from typing import Optional, Union

import pandas as pd

from scanpy import read

from scvelo.core import cleanup
from scvelo.read_load import load

url_adata = "https://drive.google.com/uc?id=1Nzq1F6dGw-nR9lhRLfZdHOG7dcYq7P0i&export=download"
url_grn = "https://drive.google.com/uc?id=1ci_gCwdgGlZ0xSn6gSa_-LlIl9-aDa1c&export=download/"

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