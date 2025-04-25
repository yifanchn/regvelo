
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

def set_prior_grn(
    adata : AnnData, 
    gt_net : pd.DataFrame
    ) -> AnnData:
    
    """
    Adding prior GRN information

    This function helps to adapt prior GRN into the anndata

    Parameters
    ----------
    adata
        Annotated data matrix.
    gt_net
        prior gene regulatory graph, row indicating targets, columns indicating regulators

    Returns
    -------
    adata object with prior GRN saved in adata.uns["skeleton"]
    """
    
    regulator_index = [i in gt_net.columns for i in adata.var.index.values]
    target_index = [i in gt_net.index for i in adata.var.index.values]
    
    gex = adata.layers["Ms"]
    corr_m = 1 - cdist(gex.T, gex.T, metric='correlation')
    corr_m = torch.tensor(corr_m)
    corr_m = corr_m[target_index,]
    corr_m = corr_m[:,regulator_index]
    corr_m = corr_m.float()
    
    corr_m = pd.DataFrame(corr_m, index = adata.var.index.values[target_index], columns = adata.var.index.values[regulator_index])
    gt_net = gt_net.loc[corr_m.index,corr_m.columns]
    GRN_final = gt_net * corr_m
    GRN_final[abs(GRN_final)<0.01] = 0
    GRN_final[GRN_final!=0] = 1
    GRN_final = GRN_final.iloc[(GRN_final.sum(1) > 0).tolist(),(GRN_final.sum(0) > 0).tolist()]
    for i in GRN_final.columns:
        GRN_final.loc[i,i] = 0
    GRN_final = GRN_final.iloc[(GRN_final.sum(1) > 0).tolist(),(GRN_final.sum(0) > 0).tolist()]
    ### use the regulatory genes to perform velocity analysis
    genes = np.unique(GRN_final.index.tolist()+GRN_final.columns.tolist())
    W = torch.zeros((len(genes),len(genes)))
    W = pd.DataFrame(W,index = genes,columns = genes)
    W.loc[GRN_final.index.tolist(),GRN_final.columns.tolist()] = GRN_final
    
    adata = adata[:,genes].copy()
    W = W.loc[adata.var.index.values.tolist(),adata.var.index.values.tolist()]
       
    adata.uns["regulators"] = adata.var.index.values
    adata.uns["targets"] = adata.var.index.values
    adata.uns["skeleton"] = np.array(W)
    adata.uns["network"] = np.array(W)
    
    return adata