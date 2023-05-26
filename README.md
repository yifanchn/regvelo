# RegVelo
inferring regulatory cellular dynamics

# installation

```
git clone 
cd ./RegVelo-global
pip install .
```

# use RegVelo
```
## reg_adata needs to contain the "regulators","targets","skeleton" and "network" instance in 'uns'
## assume we have no prior knowledge of the network, then we could set all one skeleton matrix
reg_adata.uns["regulators"] = reg_adata.var.index.values
reg_adata.uns["targets"] = reg_adata.var.index.values
reg_adata.uns["skeleton"] = np.ones((len(reg_adata.var.index),len(reg_adata.var.index)))
reg_adata.uns["network"] = np.ones((len(reg_adata.var.index),len(reg_adata.var.index)))

# skeleton
W = reg_adata.uns["skeleton"].copy()
W = torch.tensor(np.array(W)).int()
##

from regvelovi import preprocess_data,organize_multiview_anndata,sanity_check
reg_adata = sanity_check(reg_adata,network_mode = "GENIE3")
rgv_adata = organize_multiview_anndata(reg_adata) ## merge 'spliced', 'unspliced' and 'accessibility'(alternative) layer into one layer 'readout'
REGVELOVI.setup_anndata(rgv_adata, readout_layer = "readout")

## lam represents the GRN L1 penalty coefficients
reg_vae = REGVELOVI(rgv_adata,W=W.T,lam=1,interpolator = "GP",velocity_mode = "global",SVD_transform = False)
reg_vae.train(max_epochs=800,lr=0.01,optimizer = "AdamW",weight_decay = 1e-5,early_stopping = False)
```
