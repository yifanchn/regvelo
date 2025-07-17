
import numpy as np
import pandas as pd
from anndata import AnnData


def sanity_check(adata : AnnData) -> AnnData:
    r"""Ensure that all genes in the AnnData object have a least one regulator.

    Parameters
    ----------
    adata
        Annotated data object.

    Returns
    -------
    Updated AnnData object with filtered genes and refined regulatory matrices.
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