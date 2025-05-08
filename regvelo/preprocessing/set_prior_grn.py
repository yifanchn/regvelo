
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

def set_prior_grn(adata: AnnData, gt_net: pd.DataFrame, keep_dim: bool = False) -> AnnData:
    """Adds prior gene regulatory network (GRN) information to an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with gene expression data.
    gt_net : pd.DataFrame
        Prior gene regulatory network (targets as rows, regulators as columns).
    keep_dim : bool, optional
        If True, output AnnData retains original dimensions. Default is False.

    Returns
    -------
    AnnData
        Updated AnnData object with GRN stored in .uns["skeleton"].
    """
    # Identify regulators and targets present in adata
    regulator_mask = adata.var_names.isin(gt_net.columns)
    target_mask = adata.var_names.isin(gt_net.index)
    regulators = adata.var_names[regulator_mask]
    targets = adata.var_names[target_mask]

    if keep_dim:
        skeleton = pd.DataFrame(0, index=adata.var_names, columns=adata.var_names, dtype=float)
        common_targets = list(set(adata.var_names).intersection(gt_net.index))
        common_regulators = list(set(adata.var_names).intersection(gt_net.columns))
        skeleton.loc[common_targets, common_regulators] = gt_net.loc[common_targets, common_regulators]
        gt_net = skeleton.copy()

    # Compute correlation matrix based on gene expression layer "Ms"
    gex = adata.layers["Ms"]
    correlation = 1 - cdist(gex.T, gex.T, metric="correlation")
    #correlation = torch.tensor(correlation).float()
    correlation = correlation[np.ix_(target_mask, regulator_mask)]
    correlation[np.isnan(correlation)] = 0

    # Align and combine ground-truth GRN with expression correlation
    filtered_gt = gt_net.loc[targets, regulators]
    grn = filtered_gt * correlation

    # Binarize the GRN
    grn = (np.abs(grn) >= 0.01).astype(int)
    np.fill_diagonal(grn.values, 0)  # Remove self-loops

    if keep_dim:
        skeleton = pd.DataFrame(0, index=adata.var_names, columns=adata.var_names, dtype=int)
        skeleton.loc[grn.columns, grn.index] = grn.T
    else:
        # Prune genes with no edges
        grn = grn.loc[grn.sum(axis=1) > 0, grn.sum(axis=0) > 0]
        genes = grn.index.union(grn.columns)
        skeleton = pd.DataFrame(0, index=genes, columns=genes, dtype=int)
        skeleton.loc[grn.columns, grn.index] = grn.T

    # Subset the adata to GRN genes and store in .uns
    adata = adata[:, skeleton.index].copy()
    skeleton = skeleton.loc[adata.var_names, adata.var_names]

    adata.uns["regulators"] = adata.var_names.to_numpy()
    adata.uns["targets"] = adata.var_names.to_numpy()
    adata.uns["skeleton"] = skeleton
    adata.uns["network"] = skeleton.copy()

    return adata
