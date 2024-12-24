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

def get_permutation_scores(save_path: Union[str, Path] = Path("data/")) -> pd.DataFrame:
    """Get the reference permutation scores on positive and negative controls.

    Parameters
    ----------
    save_path
        path to save the csv file

    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if not (save_path / "permutation_scores.csv").is_file():
        URL = "https://figshare.com/ndownloader/files/36658185"
        urlretrieve(url=URL, filename=save_path / "permutation_scores.csv")

    return pd.read_csv(save_path / "permutation_scores.csv")


def preprocess_data(
    adata: AnnData,
    spliced_layer: Optional[str] = "Ms",
    unspliced_layer: Optional[str] = "Mu",
) -> AnnData:
    """Preprocess data.

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


def set_prior_grn(
    adata: AnnData, 
    gt_net: pd.DataFrame
) -> AnnData:
    
    """Adding prior GRN information

    This function help to adapt prior GRN into the anndata

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


def sanity_check(
       adata: AnnData,
    ) -> AnnData:
    
    """Sanity check

    This function help to ensure each gene will have at least one regulator.

    Parameters
    ----------
    adata
        Annotated data matrix.
    """
    
    gene_name = adata.var.index.tolist()
    full_name = adata.uns["regulators"]
    index = [i in gene_name for i in full_name]
    full_name = full_name[index]
    adata = adata[:,full_name].copy()

    W = adata.uns["skeleton"]
    W = W[index,:]
    W = W[:,index]

    adata.uns["skeleton"] = W 
    W = adata.uns["network"]
    W = W[index,:]
    W = W[:,index]
    #csgn = csgn[index,:,:]
    #csgn = csgn[:,index,:]
    adata.uns["network"] = W

    ###
    for i in range(1000):
        if adata.uns["skeleton"].sum(0).min()>0:
            break
        else:
            W = np.array(adata.uns["skeleton"])
            gene_name = adata.var.index.tolist()

            indicator = W.sum(0) > 0 ## every gene would need to have a upstream regulators
            regulators = [gene for gene, boolean in zip(gene_name, indicator) if boolean]
            targets = [gene for gene, boolean in zip(gene_name, indicator) if boolean]
            print("num regulators: "+str(len(regulators)))
            print("num targets: "+str(len(targets)))
            W = np.array(adata.uns["skeleton"])
            W = W[indicator,:]
            W = W[:,indicator]
            adata.uns["skeleton"] = W

            W = np.array(adata.uns["network"])
            W = W[indicator,:]
            W = W[:,indicator]
            adata.uns["network"] = W

            #csgn = csgn[indicator,:,:]
            #csgn = csgn[:,indicator,:]
            #adata.uns["csgn"] = csgn

            adata.uns["regulators"] = regulators
            adata.uns["targets"] = targets

            W = pd.DataFrame(adata.uns["skeleton"],index = adata.uns["regulators"],columns = adata.uns["targets"])
            W = W.loc[regulators,targets]
            adata.uns["skeleton"] = W
            W = pd.DataFrame(adata.uns["network"],index = adata.uns["regulators"],columns = adata.uns["targets"])
            W = W.loc[regulators,targets]
            adata.uns["network"] = W
            adata = adata[:,indicator].copy()

    return adata