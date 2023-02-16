import torch
import torch.nn as nn
from typing_extensions import Literal
import torch.nn.functional as F
from typing import Optional
import numpy as np

class DecoderODEfunc(nn.Module):
    """
    A class modelling the latent state derivatives with respect to time.

    Parameters
    ----------
    n_int
        The number of genes: ng
        (Default: 5)
    w
        The binary matrix indicate the regulatory relationship: ng*ng
    """

    def __init__(
        self,
        n_int: int = 5,
        alpha_unconstr_init: Optional[np.ndarray] = None,
        W: torch.Tensor = (torch.FloatTensor(5, 5).uniform_() > 0.5).int(),
    ):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(n_int, n_int)
        self.mask_m = W
        self.n_int =  n_int
        self.alpha_unconstr_init = alpha_unconstr_init

        ## define the trainable gamma parameter
        self.beta_mean_unconstr = torch.nn.Parameter(0.5 * torch.ones(n_int))
        self.gamma_mean_unconstr = torch.nn.Parameter(-1 * torch.ones(n_int))

        #def _set_mask_entries(self):
        #self.hooks = []
        
        ## set mask
    def _set_mask_grad(self):
        self.hooks = []
        #mask_m = self.mask_m
        
        def _hook_mask_no_regulator(grad):
            return grad * self.mask_m

        w = self.fc1.weight.register_hook(_hook_mask_no_regulator)
        self.hooks.append(w)
        
    def _set_mask_initialize(self):
        if self.alpha_unconstr_init is None:
            self.mask_m = self.mask_m.to(self.fc1.weight.device)
            int_m = self.fc1.weight.data.normal_() * self.mask_m
            self.fc1.weight.data = int_m
        else:
            self.mask_m = self.mask_m.to(self.fc1.weight.device)
            int_m = self.fc1.weight.data.normal_() * 0
            self.fc1.weight.data = int_m
            self.fc1.bias.data = torch.nn.Parameter(
                torch.from_numpy(self.alpha_unconstr_init).to(self.fc1.weight.device)
            )

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """
        Compute the gradient at a given time t and a given state x.


        Parameters
        ----------
        t
            A given time point.
        x
            A given latent state.

        Returns
        ----------
        :class:`torch.Tensor`
            A tensor
        """
        ## decompose x into unspliced and spliced readout
        if len(x.shape) == 1:
            u = x[0:self.n_int]
            s = x[self.n_int:(self.n_int*2)]
        else:
            u = x[:,0:self.n_int]
            s = x[:,self.n_int:(self.n_int*2)]

        ## generate kinetic rate
        beta = torch.clamp(F.softplus(self.beta_mean_unconstr), 0, 50)
        gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr), 0, 50)
        alpha_unconstr = self.fc1(u)
        alpha_unconstr = self.relu(alpha_unconstr)
        alpha = torch.clamp(F.softplus(alpha_unconstr),0,50)

        ## predict the gradient
        du = alpha - beta*u
        ds = beta*u - gamma*s
        
        if len(du.shape) == 1:
            out = torch.concatenate([du,ds])
        else:
            out = torch.concatenate([du,ds],axis = 1)
            
        return out


class Encoder(nn.Module):
    """
    Encoder class generating the time and latent space.
    Parameters

    Latent time is relate to the cell representation
    In VeloVI, the cell representation also used to generate the readouts
    Learning an encoder to generate suitable cell representation would also help to give a better latent time estimation
    
    the loss could be represented as:
    p(x|z) + ||x - x_t|| + ||z - z_t|| - D_KL(q(z|x)||p(z))
    ----------
    n_int
        The dimensionality of the input.
    n_latent
        The dimensionality of the latent space.
        (Default: 5)
    n_hidden
        The dimensionality of the hidden layer.
        (Default: 128)
    batch_norm
        Whether to include `BatchNorm` layer or not.
        (Default: `False`)
    """

    def __init__(
        self,
        n_int: int,
        n_latent: int = 5,
        n_hidden: int = 128,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.fc = nn.Sequential()
        self.fc.add_module('L1', nn.Linear(n_int, n_hidden))
        if batch_norm:
            self.fc.add_module('N1', nn.BatchNorm1d(n_hidden))
        self.fc.add_module('A1', nn.ReLU())
        self.fc2 = nn.Linear(n_hidden, n_latent*2)
        #self.fc3 = nn.Linear(n_hidden, 1)

    def forward(self, x:torch.Tensor) -> tuple:
        x = self.fc(x)
        out = self.fc2(x)
        qz_mean, qz_logvar = out[:, :self.n_latent], out[:, self.n_latent:]
        #t = self.fc3(x).sigmoid() * 20
        #t = torch.clamp(F.softplus(self.fc3(x)),0,20)
        return qz_mean, qz_logvar


class Decoder(nn.Module):
    """
    Decoder class to reconstruct the original input based on its latent space.

    Parameters
    ----------
    n_latent
        The dimensionality of the latent space.
        (Default: 5)
    n_int
        The dimensionality of the original input.
    n_hidden
        The dimensionality of the hidden layer.
        (Default: 128)
    batch_norm
        Whether to include `BatchNorm` layer or not.
        (Default: `False`)
    loss_mode
        The mode for reconstructing the original data.
        (Default: `'nb'`)
    """

    def __init__(
        self,
        n_int: int,
        n_latent: int = 5,
        n_hidden: int = 128,
        batch_norm: bool = False,
        loss_mode: Literal['mse', 'nb', 'zinb'] = 'nb',
    ):
        super().__init__()
        self.loss_mode = loss_mode
        if loss_mode in ['nb', 'zinb']:
            self.disp = nn.Parameter(torch.randn(n_int))

        self.fc = nn.Sequential()
        self.fc.add_module('L1', nn.Linear(n_latent, n_hidden))
        if batch_norm:
            self.fc.add_module('N1', nn.BatchNorm1d(n_hidden))
        self.fc.add_module('A1', nn.ReLU())

        if loss_mode == 'mse':
            self.fc2 = nn.Linear(n_hidden, n_int)
        if loss_mode in ['nb', 'zinb']:
            self.fc2 = nn.Sequential(nn.Linear(n_hidden, n_int), nn.Softmax(dim = -1))
        if loss_mode == 'zinb':
            self.fc3 = nn.Linear(n_hidden, n_int)

    def forward(self, z: torch.Tensor):
        out = self.fc(z)
        recon_x = self.fc2(out)
        if self.loss_mode == 'zinb':
            disp = self.fc3(out)
            return recon_x, disp
        else:
            return recon_x
