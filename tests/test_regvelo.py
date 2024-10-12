## test RegVelo
import numpy as np
import pandas as pd
from scvi.data import synthetic_iid
from regvelo import REGVELOVI
import torch

def test_regvelo():
    adata = synthetic_iid()
    adata.layers["spliced"] = adata.X.copy()
    adata.layers["unspliced"] = adata.X.copy()
    adata.var_names = "Gene" + adata.var_names
    n_gene = len(adata.var_names)
    ## create W
    grn_matrix = np.random.choice([0, 1], size=(n_gene,n_gene), p=[0.8, 0.2]).T
    W = pd.DataFrame(grn_matrix, index=adata.var_names, columns=adata.var_names)
    adata.uns["skeleton"] = W
    TF_list = adata.var_names.tolist()

    ## training process
    W = adata.uns["skeleton"].copy()
    W = torch.tensor(np.array(W))
    REGVELOVI.setup_anndata(adata, spliced_layer="spliced", unspliced_layer="unspliced")

    ## Training the model
    # hard constraint
    reg_vae = REGVELOVI(adata,W=W.T,regulators = TF_list,soft_constraint = False)
    reg_vae.train()
    # soft constraint
    reg_vae = REGVELOVI(adata,W=W.T,regulators = TF_list,soft_constraint = True)
    reg_vae.train()

    reg_vae.get_latent_representation()
    reg_vae.get_velocity()
    reg_vae.get_latent_time()
    
    reg_vae.history
    print(reg_vae)