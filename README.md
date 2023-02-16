# RegVelo
inferring regulatory cellular dynamics

# installation

```
git clone 
cd ./RegVelo
pip install .
```

# use RegVelo
```
import RegVelo as rgv
```
1. build a RegVelo model and training the model
```
rgv_m = rgv.train.Trainer(adata, loss_mode='mse',W=W.T, early_stopping = False, nepoch = 400, n_latent = 20)
rgv_m.train()
```
2. get latent time and velocity vector
```
adata.obs["ptime"] = rgv_m.get_time()
velocity = rgv_m.get_vector_field(T=adata.obs['ptime'].values)[:,int(rgv_m.n_int/2):rgv_m.n_int]
```
