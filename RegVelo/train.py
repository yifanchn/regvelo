import torch
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from typing import Optional, Union
from typing_extensions import Literal
import numpy as np
from anndata import AnnData
from scipy import sparse
from scipy.sparse import spmatrix
from tqdm import tqdm
import os
from collections import defaultdict
from scipy.spatial.distance import cdist

from .model import TNODE
from ._utils import get_step_size, _softplus_inverse
from .data import split_data, MakeDataset, BatchSampler


##reverse time
def reverse_time(t):
    return 1 - t

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Trainer:
    """
    Class for implementing the scTour training process.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    percent
        The percentage of cells used for model training. Default to 0.2 when the cell number > 10,000 and to 0.9 otherwise.
    n_latent
        The dimensionality of the latent space.
        (Default: 5)
    n_ode_hidden
        The dimensionality of the hidden layer for the latent ODE function.
        (Default: 25)
    n_vae_hidden
        The dimensionality of the hidden layer for the VAE.
        (Default: 128)
    batch_norm
        Whether to include a `BatchNorm` layer.
        (Default: `False`)
    ode_method
        The solver for integration.
        (Default: `'euler'`)
    step_size
        The step size during integration.
    alpha_recon_lec
        The scaling factor for the reconstruction loss from encoder-derived latent space.
        (Default: 0.5)
    alpha_recon_lode
        The scaling factor for the reconstruction loss from ODE-solver latent space.
        (Default: 0.5)
    alpha_kl
        The scaling factor for the KL divergence.
        (Default: 1.0)
    loss_mode
        The mode for calculating the reconstruction error.
        (Default: `'nb'`)
        Three modes are included
        ``'mse'``: mean squared error
        ``'nb'``: negative binomial conditioned likelihood
        ``'zinb'``: zero-inflated negative binomial conditioned likelihood
    nepoch
        Number of epochs.
    batch_size
        The batch size during training.
        (Default: 1024)
    drop_last
        Whether or not drop the last batch when its size is smaller than the batch_size
        (Default: `False`)
    lr
        The learning rate.
        (Default: 1e-3)
    wt_decay
        The weight decay for Adam optimizer.
        (Default: 1e-6)
    eps
        The eps for Adam optimizer.
        (Default: 0.01)
    random_state
        The seed for generating random numbers.
        (Default: 0)
    val_frac
        The percentage of data used for validation.
        (Default: 0.1)
    use_gpu
        Whether to use GPU when available
        (Default: True)
    """

    def __init__(
        self,
        adata: AnnData,
        percent: Optional[float] = None,
        n_latent: int = 20,
        n_ode_hidden: int = 25,
        n_vae_hidden: int = 128,
        batch_norm: bool = False,
        ode_method: str = 'euler',
        step_size: Optional[int] = None,
        alpha_recon_lec: float = 0.5,
        alpha_recon_lode: float = 0.5,
        alpha_z_div: float = 1.,
        alpha_kl: float = 1.,
        loss_mode: Literal['mse', 'nb', 'zinb'] = 'mse',
        nepoch: Optional[int] = None,
        batch_size: int = 1024,
        drop_last: bool = False,
        solver: str = "RMSprop",
        lr: float = 1e-3,
        momentum: float = 0.9,
        scheduler_lr: bool = True,
        scheduler: str = "Cosine",
        T_max: int = 100,
        wt_decay: float = 1e-6,
        eps: float = 0.01,
        random_state: int = 0,
        val_frac: float = 0.1,
        use_gpu: bool = True,
        W: torch.Tensor = None,
        corr_int: bool = True,
        ratio: float = 0.8,
        early_stopping: bool = True,
        patience: int = 45,
        min_delta: float = 0.1,
    ):
        self.loss_mode = loss_mode
        if self.loss_mode not in ['mse', 'nb', 'zinb']:
            raise ValueError(f"`loss_mode` must be one of ['mse', 'nb', 'zinb'], but input was '{self.loss_mode}'.")

        if (alpha_recon_lec < 0) or (alpha_recon_lec > 1):
            raise ValueError('`alpha_recon_lec` must be between 0 and 1.')
        if (alpha_recon_lode < 0) or (alpha_recon_lode > 1):
            raise ValueError('`alpha_recon_lode` must be between 0 and 1.')
        if alpha_recon_lec + alpha_recon_lode != 1:
            raise ValueError('The sum of `alpha_recon_lec` and `alpha_recon_lode` must be 1.')

        self.adata = adata
        if 'n_genes_by_counts' not in self.adata.obs:
            raise KeyError("`n_genes_by_counts` not found in `.obs` of the AnnData. Please run `scanpy.pp.calculate_qc_metrics` first to calculate the number of genes detected in each cell.")
        if loss_mode == 'mse':
            if (self.adata.X.min() < 0) or (self.adata.X.max() > np.log1p(1e6)):
                raise ValueError("Invalid expression matrix in `.X`. `mse` mode expects log1p(normalized expression) in `.X` of the AnnData.")
        else:
            X = self.adata.X.data if sparse.issparse(self.adata.X) else self.adata.X
            if (X.min() < 0) or np.any(~np.equal(np.mod(X, 1), 0)):
                raise ValueError(f"Invalid expression matrix in `.X`. `{self.loss_mode}` mode expects raw UMI counts in `.X` of the AnnData.")

        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta

        self.n_cells = adata.n_obs
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.percent = percent
        if self.percent is None:
            if self.n_cells > 10000:
                self.percent = .2
            else:
                self.percent = .9
        else:
            if (self.percent < 0) or (self.percent > 1):
                raise ValueError("`percent` must be between 0 and 1.")
        self.val_frac = val_frac
        if (self.val_frac < 0) or (self.val_frac > 1):
            raise ValueError('`val_frac` must be between 0 and 1.')

        if nepoch is None:
            ncells = round(self.n_cells * self.percent)
            self.nepoch = np.min([round((10000 / ncells) * 400), 400])
        else:
            self.nepoch = nepoch

        self.solver = solver
        self.scheduler_lr = scheduler_lr
        self.scheduler_model = scheduler
        self.T_max = T_max
        self.lr = lr
        self.momentum = momentum
        self.wt_decay = wt_decay
        self.eps = eps
        self.time_reverse = None

        self.random_state = random_state
        np.random.seed(random_state)
#       random.seed(random_state)
        torch.manual_seed(random_state)
#       torch.backends.cudnn.benchmark = False
#       torch.use_deterministic_algorithms(True)

        ## update alpha initialization
        spliced = adata.layers["Ms"]
        unspliced = adata.layers["Mu"]

        sorted_unspliced = np.argsort(unspliced, axis=0)
        ind = int(adata.n_obs * 0.99)
        us_upper_ind = sorted_unspliced[ind:, :]

        us_upper = []
        ms_upper = []
        for i in range(len(us_upper_ind)):
            row = us_upper_ind[i]
            us_upper += [unspliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
            ms_upper += [spliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
        us_upper = np.median(np.concatenate(us_upper, axis=0), axis=0)
        ms_upper = np.median(np.concatenate(ms_upper, axis=0), axis=0)

        alpha_unconstr_init = _softplus_inverse(us_upper)

        gpu = torch.cuda.is_available() and use_gpu
        if gpu:
            torch.cuda.manual_seed(random_state)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.n_int = adata.n_vars*2

        ## initialization
        if corr_int:
            corr_m = 1 - cdist(self.adata.X.todense().T, self.adata.X.todense().T, metric='correlation')
            corr_m = torch.tensor(corr_m).to(self.device)
        else:
            corr_m = None

        self.model_kwargs = dict(
            device = self.device,
            n_int = self.n_int,
            n_latent = n_latent,
            n_ode_hidden = n_ode_hidden,
            n_vae_hidden = n_vae_hidden,
            batch_norm = batch_norm,
            ode_method = ode_method,
            step_size = step_size,
            alpha_recon_lec = alpha_recon_lec,
            alpha_recon_lode = alpha_recon_lode,
            alpha_z_div = alpha_z_div,
            alpha_kl = alpha_kl,
            loss_mode = loss_mode,
            W = W,
            corr_m = corr_m,
            alpha_unconstr_init = alpha_unconstr_init,
            ratio = ratio,
        )
        self.model = TNODE(**self.model_kwargs)
        self.log = defaultdict(list)


    def _get_data_loaders(self) -> None:
        """
        Generate Data Loaders for training and validation datasets.
        """

        train_data, val_data = split_data(self.adata, self.percent, self.val_frac)
        self.train_dataset = MakeDataset(train_data, self.loss_mode)
        self.val_dataset = MakeDataset(val_data, self.loss_mode)

#        sampler = BatchSampler(train_data.n_obs, self.batch_size, self.drop_last)
#        self.train_dl = DataLoader(self.train_dataset, batch_sampler = sampler)
        self.train_dl = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        self.val_dl = DataLoader(self.val_dataset, batch_size = self.batch_size)


    def train(self):
        self._get_data_loaders()

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.solver == "AdamW":
            self.optimizer = torch.optim.AdamW(params, lr = self.lr, weight_decay = self.wt_decay, eps = self.eps)
        if self.solver == "Adam":
            self.optimizer = torch.optim.Adam(params, lr = self.lr, weight_decay = self.wt_decay, eps = self.eps)
        ## Don't use ADAM but try RMSprop
        if self.solver == "RMSprop":
            self.optimizer = torch.optim.RMSprop(params, lr=self.lr, momentum=self.momentum)
        if self.scheduler_lr:
            if self.scheduler_model == "Cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.T_max, eta_min=0, last_epoch=-1, verbose=False)
            if self.scheduler_model == "CyclicLR":
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr*0.1, max_lr=self.lr, mode='triangular', gamma=1.0,cycle_momentum=False)

        if self.early_stopping:
            early_stopper = EarlyStopper(patience = self.patience, min_delta = self.min_delta)
        with tqdm(total=self.nepoch, unit='epoch') as t:
            for tepoch in range(t.total):
                if self.scheduler_lr:
                    self.scheduler.step()
                train_loss, train_recon_loss_ec, train_recon_loss_ode, train_kl_div, train_z_div = self._on_epoch_train(self.train_dl)
                val_loss, val_recon_loss_ec, val_recon_loss_ode, val_kl_div, val_z_div = self._on_epoch_val(self.val_dl)
                if self.early_stopping:
                    if early_stopper.early_stop(val_loss):             
                        break
                self.log['train_loss'].append(train_loss)
                self.log['validation_loss'].append(val_loss)
                self.log['train_recon_loss_ec'].append(train_recon_loss_ec)
                self.log['val_recon_loss_ec'].append(val_recon_loss_ec)
                self.log['train_recon_loss_ode'].append(train_recon_loss_ode)
                self.log['val_recon_loss_ode'].append(val_recon_loss_ode)
                self.log['train_kl_div'].append(train_kl_div)
                self.log['val_kl_div'].append(val_kl_div)
                self.log['train_z_div'].append(train_z_div)
                self.log['val_z_div'].append(val_z_div)
                t.set_description(f"Epoch {tepoch + 1}")
                t.set_postfix({'train_loss': train_loss, 'val_loss': val_loss}, refresh=False)
                t.update()


    def _on_epoch_train(self, DL) -> float:
        """
        Go through the model and update the model parameters.

        Parameters
        ----------
        DL
            DataLoader for training dataset.

        Returns
        ----------
        float
            Training loss for the current epoch.
        """

        self.model.train()
        total_loss = total_recon_loss_ec = total_recon_loss_ode = total_kl_div = total_z_div = .0
        ss = 0
        for X, Y in DL:
            self.optimizer.zero_grad()
            X = X.to(self.device)
            Y = Y.to(self.device)
            loss, recon_loss_ec, recon_loss_ode, kl_div, z_div = self.model(X, Y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X.size(0)
            total_recon_loss_ec += recon_loss_ec.item() * X.size(0)
            total_recon_loss_ode += recon_loss_ode.item() * X.size(0)
            total_kl_div += kl_div.item() * X.size(0)
            total_z_div += z_div.item() * X.size(0)

            ss += X.size(0)

        train_loss = total_loss/ss
        train_recon_loss_ec = total_recon_loss_ec/ss
        train_recon_loss_ode = total_recon_loss_ode/ss
        train_kl_div = total_kl_div/ss
        train_z_div = total_z_div/ss
        return train_loss, train_recon_loss_ec, train_recon_loss_ode, train_kl_div, train_z_div


    @torch.no_grad()
    def _on_epoch_val(self, DL) -> float:
        """
        Validate using validation dataset.

        Parameters
        ----------
        DL
            DataLoader for validation dataset.

        Returns
        ----------
        float
            Validation loss for the current epoch.
        """

        self.model.eval()
        total_loss = total_recon_loss_ec = total_recon_loss_ode = total_kl_div = total_z_div = .0
        ss = 0
        for X, Y in DL:
            X = X.to(self.device)
            Y = Y.to(self.device)
            loss, recon_loss_ec, recon_loss_ode, kl_div, z_div = self.model(X, Y)
            total_loss += loss.item() * X.size(0)
            ss += X.size(0)

            total_recon_loss_ec += recon_loss_ec.item() * X.size(0)
            total_recon_loss_ode += recon_loss_ode.item() * X.size(0)
            total_kl_div += kl_div.item() * X.size(0)
            total_z_div += z_div.item() * X.size(0)

        val_loss = total_loss/ss
        val_recon_loss_ec = total_recon_loss_ec/ss
        val_recon_loss_ode = total_recon_loss_ode/ss
        val_kl_div = total_kl_div/ss
        val_z_div = total_z_div/ss
        return val_loss, val_recon_loss_ec, val_recon_loss_ode, val_kl_div, val_z_div


    @torch.no_grad()
    def get_time(self) -> np.ndarray:
        """
        Get the developmental pseudotime for all cells.

        Returns
        ----------
        :class:`~numpy.ndarray`
            The estimated pseudotime of cells.
        """

        self.model.eval()
        X = np.concatenate([self.adata.layers["Mu"],self.adata.layers["Ms"]],axis = 1)
        X = torch.tensor(X).to(self.device)
        #ts, _, _ = self.model.encoder(X)
        qz_m, _ = self.model.encoder(X)
        ts = self.model.t_encoder(qz_m).sigmoid() * 20
        ts = ts.ravel()

        ## The model might return reversed time. Check this based on number of genes expressed in cells
        #if self.time_reverse is None:
        #    n_genes = torch.tensor(self.adata.obs['n_genes_by_counts'].values).float().log1p().to(self.device)
        #    m_ts = ts.mean()
        #    m_ngenes = n_genes.mean()
        #    beta_direction = (ts * n_genes).sum() - len(ts) * m_ts * m_ngenes
        #    if beta_direction > 0:
        #        self.time_reverse = True
        #    else:
        #        self.time_reverse = False
        #if self.time_reverse:
        #    ts = 1 - ts

        return ts.cpu().detach().numpy()


    @torch.no_grad()
    def get_vector_field(
        self,
        T: np.ndarray,
        model: Optional[str] = None,
        sim_dynamics: bool = True,
    ) -> np.ndarray:
        """
        Get the vector field.

        Parameters
        ----------
        T
            The estimated pseudotime for each cell.
        Z
            The latent representation for each cell.
        model
            The model used to get the vector field. Only provided when using the saved model.

        Returns
        ----------
        :class:`~numpy.ndarray`
            The estimated vector field.
        """

        model = self._get_model(model)
        model.eval()
        Z = np.concatenate([self.adata.layers["Mu"],self.adata.layers["Ms"]],axis = 1)
        if not (isinstance(T, np.ndarray) and isinstance(Z, np.ndarray)):
            raise TypeError('The inputs must be numpy arrays.')
        Z = torch.tensor(Z).to(self.device)
        T = torch.tensor(T).to(self.device)
        #index = torch.argsort(T)
        #T = T[index]
        #Z = Z[index]
        
        #if self.time_reverse is None:
        #    raise RuntimeError('It seems you did not run get_time() function first. Please run get_time() before you run get_vector_field().')
        #direction = 1
        #if self.time_reverse:
        #    direction = -1
        velo = model.lode_func(T, Z).cpu().detach().numpy()

        if sim_dynamics:
            id = adata.obs.index.values
            ## simulate the system to generate the dynamics of all cells
            T = T.ravel()  ## odeint requires 1-D Tensor for time
            index = torch.argsort(T)
            T = T[index]
            Z = Z[index]
            id = id[index]
            index2 = (T[:-1] != T[1:])
            index2 = torch.cat((index2, torch.tensor([True]).to(index2.device))) ## index2 is used to get unique time points as odeint requires strictly increasing/decreasing time points
            T = T[index2]   
            Z = Z[index2]
            id = id[index2]

            ## generate dynamics through ODE solver
            Z0 = Z[0]
            options = get_step_size(model.step_size, T[0], T[-1], len(T))
            pred_x = odeint(model.lode_func, Z0.to(model.device), T.to(model.device), method = model.ode_method, options = options).view(-1, model.n_int)
            pred_x = pred_x.to(model.device)
            
            ## insert into the velocity matrix
            insert = [adata.obs.index.tolist().index(i) for i in id]
            for i in insert:
                velo[i,:] = pred_x[i,:]
        
        return velo

    def save_model(
        self,
        save_dir: str,
        save_prefix: str,
    ) -> None:
        """
        Save the trained model.

        Parameters
        ----------
        save_dir
            The directory where the model will be saved.
        save_prefix
            The prefix for model name.
        """

        save_path = os.path.abspath(os.path.join(save_dir, f'{save_prefix}.pth'))
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'var_names': self.adata.var_names,
                'nepoch': self.nepoch,
                'random_state': self.random_state,
                'percent': self.percent,
                'time_reverse': self.time_reverse,
                'model_kwargs': self.model_kwargs,
                'log': self.log,
            },
            save_path
        )

    @torch.no_grad()
    def get_latentsp(
        self,
        X: Optional[Union[np.ndarray, spmatrix]] = None,
        alpha_z: float = .5,
        alpha_predz: float = .5,
        step_size: Optional[int] = None,
        step_wise: bool = False,
        batch_size: Optional[int] = None,
        model: Optional[str] = None,
    ):
        """
        Get the latent representations of cells.

        Parameters
        ----------
        X
            The data matrix. Only provided when you want to get the latent representations for data not used for training.
        alpha_z
            Scaling factor for encoder-derived latent space.
            (Default: 0.5)
        alpha_predz
            Scaling factor for ODE-solver-derived latent space.
            (Default: 0.5)
        step_size
            Step size during integration.
        step_wise
            Whether to perform step-wise integration by providing just two time points at a time.
            (Default: `False`)
        batch_size
            Batch size during getting the latent space. The default is no mini-batching.
        model
            The model used to get the latent space. Only provided when you use the saved model.

        Returns
        ----------
        tuple
            3-tuple of mixed latent space, encoder-derived latent space, and ODE-solver-derived latent space.
        """

        model = self._get_model(model)
        model.eval()

        if (alpha_z < 0) or (alpha_z > 1):
            raise ValueError('`alpha_z` must be between 0 and 1.')
        if (alpha_predz < 0) or (alpha_predz > 1):
            raise ValueError('`alpha_predz` must be between 0 and 1.')
        if alpha_z + alpha_predz != 1:
            raise ValueError('The sum of `alpha_z` and `alpha_predz` must be 1.')

        if X is None:
            X = self.adata.X
            if model.loss_mode in ['nb', 'zinb']:
                X = np.log1p(X)
        if sparse.issparse(X):
            X = X.A
        X = torch.tensor(X).to(self.device)
        T, qz_mean, qz_logvar = model.encoder(X)
        T = T.ravel().cpu()
        epsilon = torch.randn(qz_mean.size())
        zs = epsilon * torch.exp(.5 * qz_logvar.cpu()) + qz_mean.cpu()

        sort_T, sort_idx, sort_ridx = np.unique(T, return_index=True, return_inverse=True)
        sort_T = torch.tensor(sort_T)
        sort_zs = zs[sort_idx]

        pred_zs = []
        if batch_size is None:
            batch_size = len(sort_T)
        times = int(np.ceil(len(sort_T) / batch_size))
        for i in range(times):
            idx1 = i * batch_size
            idx2 = np.min([(i + 1)*batch_size, len(sort_T)])
            t = sort_T[idx1:idx2]
            z = sort_zs[idx1:idx2]
            z0 = z[0]

            if not step_wise:
                options = get_step_size(step_size, t[0], t[-1], len(t))
                pred_z = odeint(
                                model.lode_func,
                                z0,
                                t,
                                method = model.ode_method,
                                options = options
                                ).view(-1, model.n_latent)
            else:
                pred_z = torch.empty((len(t), z.size(1)))
                pred_z[0] = z0
                for j in range(len(t) - 1):
                    t2 = t[j:(j + 2)]
                    options = get_step_size(step_size, t2[0], t2[-1], len(t2))
                    pred_z[j + 1] = odeint(
                                            model.lode_func,
                                            z0,
                                            t2,
                                            method = model.ode_method,
                                            options = options
                                    )[1]
                    z0 = z[j + 1]

            pred_zs += [pred_z]

        pred_zs = torch.cat(pred_zs)
        pred_zs = pred_zs[sort_ridx]
        mix_zs = alpha_z * zs + alpha_predz * pred_zs

        return mix_zs.numpy(), zs.numpy(), pred_zs.numpy()


    @torch.no_grad()
    def predict_time(
        self,
        adata: Optional[AnnData] = None,
        reverse: bool = False,
        get_ltsp: bool = True,
        mode: Literal['coarse', 'fine'] = 'fine',
        alpha_z: float = .5,
        alpha_predz: float = .5,
        step_size: Optional[int] = None,
        step_wise: bool = False,
        batch_size: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Union[np.ndarray, tuple]:
        """
        Predict the pseudotime of cells given their transcriptomes, as well as their latent representations when get_ltsp is set to True.

        Parameters
        ----------
        adata
            An :class:`~anndata.AnnData` object from the dataset that will be predicted.
        reverse
            Whether to reverse the predicted pseudotime. When the pseudotime returned by get_time() function was in reverse order and you used the post-inference adjustment (reverse_time() function), please set this parameter to `True`.
            (Default: `False`)
        get_ltsp
            Whether to get the latent space as well.
            (Default: `False`)
        mode
            The method for getting the latent space.
            Two modes are included
            ``'fine'``: take the training dataset into consideration when predicting the latent space.
            ``'coarse'``: derive the latent space of the given dataset directly without involving the training data.
        alpha_z
            Scaling factor for encoder-derived latent space.
            (Default: 0.5)
        alpha_predz
            Scaling factor for ODE-solver-derived latent space.
            (Default: 0.5)
        step_size
            The step size during integration.
        step_wise
            Whether to perform step-wise integration by providing just two time points at a time.
            (Default: `False`)
        batch_size
            Batch size during getting the latent space. The default is no mini-batching.
        model
            The model used to predict the pseudotime. Only provided when using the saved model.

        Returns
        ----------
        The predicted pseudotime and (if `get_ltsp = True`) the latent space.
        """

        model = self._get_model(model)
        model.eval()

        if self.time_reverse is None:
            raise RuntimeError('It seems you did not run get_time() function first. Please run get_time() using training data before you run predict_time() using test data.')

        if adata is None:
            X = self.adata.X
            if model.loss_mode in ['nb', 'zinb']:
                X = np.log1p(X)
        else:
            if len(adata.var_names.intersection(self.adata.var_names)) != self.adata.n_vars:
                raise ValueError("The given AnnData must contain all the genes that are used for model training from the training dataset.")

            adata = adata[:, self.adata.var_names]
            X = adata.X
            if model.loss_mode == 'mse':
                if (X.min() < 0) or (X.max() > np.log1p(1e6)):
                    raise ValueError("Invalid expression matrix in `.X`. Model trained from `mse` mode expects log1p(normalized expression) in `.X` of the AnnData.")
            else:
                data = X.data if sparse.issparse(X) else X
                if (data.min() < 0) or np.any(~np.equal(np.mod(data, 1), 0)):
                    raise ValueError(f"Invalid expression matrix in `.X`. Model trained from `{model.loss_mode}` mode expects raw UMI counts in `.X` of the AnnData.")
                else:
                    X = np.log1p(X)

        if sparse.issparse(X):
            X = X.A
        X = torch.tensor(X).to(self.device)
        ts, _, _ = model.encoder(X)
        ts = ts.ravel()
        if self.time_reverse:
            ts = 1 - ts
        if reverse:
            ts = 1 - ts

        if get_ltsp:
            if mode == 'coarse':
                mix_zs, zs, pred_zs = self.get_latentsp(
                                        X = X.cpu().numpy(),
                                        alpha_z = alpha_z,
                                        alpha_predz = alpha_predz,
                                        step_size = step_size,
                                        step_wise = step_wise,
                                        batch_size = batch_size,
                                        model = model,
                                        )
            if mode == 'fine':
                X2 = self.adata.X
                if model.loss_mode in ['nb', 'zinb']:
                    X2 = np.log1p(X2)
                if sparse.issparse(X2):
                    X2 = X2.A
                mix_zs, zs, pred_zs = self.get_latentsp(
                                        X = np.vstack((X.cpu().numpy(), X2)),
                                        alpha_z = alpha_z,
                                        alpha_predz = alpha_predz,
                                        step_size = step_size,
                                        step_wise = step_wise,
                                        batch_size = batch_size,
                                        model = model,
                                        )
                mix_zs = mix_zs[:len(X)]
                zs = zs[:len(X)]
                pred_zs = pred_zs[:len(X)]

        if not get_ltsp:
            return ts.cpu().numpy()
        else:
            return ts.cpu().numpy(), mix_zs, zs, pred_zs


    @torch.no_grad()
    def predict_ltsp_from_time(
        self,
        T: np.ndarray,
        reverse: bool = False,
        step_wise: bool = True,
        step_size: Optional[int] = None,
        alpha_z: float = 0.5,
        alpha_predz: float = 0.5,
        k: int = 20,
        model: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict the transcriptomic latent space for unobserved time intervals.

        Parameters
        ----------
        T
            A 1D numpy array containing the time points (unobserved time interval) with values between 0 and 1.
        reverse
            Whether to reverse the reference pseudotime from training data. When the pseudotime returned by get_time() function was in reverse order and you used the post-inference adjustment (reverse_time() function), please set this parameter to `True`.
            (Default: `False`)
        step_wise
            Whether to perform step-wise integration by providing just two time points at a time when inferring the reference latent space from training data.
            (Default: `True`)
        step_size
            The step size during integration.
        alpha_z
            Scaling factor for encoder-derived latent space when inferring the reference latent space from training data.
            (Default: 0.5)
        alpha_predz
            Scaling factor for ODE-solver-derived latent space when inferring the reference latent space from training data.
            (Default: 0.5)
        k
            The k nearest neighbors in the time space used to predict the latent space for each unobserved time point.
            (Default: 20)
        model
            The model used to predict the transcriptomic latent space. Only provided when using the saved model.

        Returns
        ----------
        :class:`~numpy.ndarray`
            Predicted latent space for the unobserved time interval.
        """

        model = self._get_model(model)
        model.eval()

        if not isinstance(T, np.ndarray):
            raise TypeError("The input time interval must be a numpy array.")
        if len(T.shape) > 1:
            raise TypeError("The input time interval must be a 1D numpy array.")
        if np.any(T < 0) or np.any(T > 1):
            raise ValueError("The provided time points must be between 0 and 1.")

        ridx = np.random.permutation(len(T))
        rT = torch.tensor(T[ridx])
        ## get the reference time and latent space from the training data
        mix_zs, zs, pred_zs = self.get_latentsp(step_wise = step_wise, step_size = step_size, model = model, alpha_z = alpha_z, alpha_predz = alpha_predz)
        ts = self.predict_time(reverse = reverse, get_ltsp = False, model = model)
#       zs = torch.tensor(zs).to(self.device)
        zs = torch.tensor(mix_zs)
        ts = torch.tensor(ts)

        pred_T_zs = torch.empty((len(rT), model.n_latent))
        for i, t in enumerate(rT):
            diff = torch.abs(t - ts)
            idxs = torch.argsort(diff)
#            n = (diff == 0).sum()
#            idxs = idxs[n:(k + n)]
            if (diff == 0).any():
                pred_T_zs[i] = zs[idxs[0]]
            else:
                idxs = idxs[:k]
                k_zs = torch.empty((k, model.n_latent))
                for j, idx in enumerate(idxs):
                    z0 = zs[idx]
                    t0 = ts[idx]
                    pred_t = torch.stack((t0, t))
                    if pred_t[0] < pred_t[1]:
                        options = get_step_size(step_size, pred_t[0], pred_t[-1], len(pred_t))
                    else:
                        options = get_step_size(step_size, pred_t[-1], pred_t[0], len(pred_t))
                    k_zs[j] = odeint(
                                    model.lode_func,
                                    z0,
                                    pred_t,
                                    method = model.ode_method,
                                    options = options
                                )[1]
                k_zs = torch.mean(k_zs, dim = 0)
                pred_T_zs[i] = k_zs
                ts = torch.cat((ts, t.unsqueeze(0)))
                zs = torch.cat((zs, k_zs.unsqueeze(0)))

        pred_T_zs = pred_T_zs[np.argsort(ridx)]
#       return pred_T_zs.numpy(), pred_zs
        return pred_T_zs.numpy()


    def _get_model(self, model):
        """
        Get the model for inference/prediction (for internal use).
        """

        if isinstance(model, str):
            checkpoint = torch.load(model, map_location=self.device)
            model = TNODE(**checkpoint['model_kwargs'])
            model.load_state_dict(checkpoint['model_state_dict'])
            self.time_reverse = checkpoint['time_reverse']
        if model is None:
            model = self.model
        return model
