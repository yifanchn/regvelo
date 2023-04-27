###############
import torch
import torch.nn as nn
from typing_extensions import Literal
import torch.nn.functional as F
from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist
from scvi.nn import Encoder, FCLayers


class alpha_encoder(nn.Module):
    """ 
    encode the time dependent alpha (f)
    time dependent transcription rate is determined by upstream emulator

    """                 
    def __init__(
        self,
        n_int: int = 5,
        alpha_unconstr_init: torch.Tensor = None,
        W: torch.Tensor = (torch.FloatTensor(5, 5).uniform_() > 0.5).int(),
        W_int: torch.Tensor = None,
        log_h_int: torch.Tensor = None,
        global_time: bool = False,
    ):
        device = W.device
        super().__init__()
        self.n_int = n_int
        self.device = device
        if global_time:
            self.log_h = torch.nn.Parameter(torch.randn(W.shape[1]))
            self.log_phi = torch.nn.Parameter(torch.randn(W.shape[1]))
            self.tau = torch.nn.Parameter(torch.randn(W.shape[1]))
            self.o = torch.nn.Parameter(torch.randn(W.shape[1]))
        else:
            self.log_h = torch.nn.Parameter(log_h_int.repeat(W.shape[0],1)*W)
            self.log_phi = torch.nn.Parameter(torch.ones(W.shape).to(device)*W)
            self.tau = torch.nn.Parameter(torch.ones(W.shape).to(device)*W*10)
            self.o = torch.nn.Parameter(torch.ones(W.shape).to(device)*W)

        self.mask_m = W
        self.global_time = global_time

        ## initialize grn
        self.grn = torch.nn.Parameter(W_int*self.mask_m)
        
        ## initilize gamma and beta
        self.beta_mean_unconstr = torch.nn.Parameter(0.5 * torch.ones(n_int))
        self.gamma_mean_unconstr = torch.nn.Parameter(-1 * torch.ones(n_int))
        self.alpha_unconstr_bias = torch.nn.Parameter(torch.zeros(n_int))
        self.alpha_unconstr_max = torch.nn.Parameter(torch.randn(n_int))
        # calculating emulating matrix
    
    ### define hook to froze the parameters
    def _set_mask_grad(self):
        self.hooks_grn = []
        if not self.global_time:
            self.hooks_log_h = []
            self.hooks_log_phi = []
            self.hooks_tau = []
            self.hooks_o = []
        #mask_m = self.mask_m
        
        def _hook_mask_no_regulator(grad):
            return grad * self.mask_m

        w_grn = self.grn.register_hook(_hook_mask_no_regulator)
        self.hooks_grn.append(w_grn)
        if not self.global_time:
            w_log_h = self.log_h.register_hook(_hook_mask_no_regulator)
            w_log_phi = self.log_phi.register_hook(_hook_mask_no_regulator)
            w_tau = self.tau.register_hook(_hook_mask_no_regulator)
            w_o = self.o.register_hook(_hook_mask_no_regulator)

            self.hooks_log_h.append(w_log_h)
            self.hooks_log_phi.append(w_log_phi)
            self.hooks_tau.append(w_tau)
            self.hooks_o.append(w_o)

    def emulator(t,log_h_v,log_phi_v,tau_v,o_v):
        pre = torch.exp(log_h_v)*torch.exp(-torch.exp(phi_v)*(t-tau_v)**2)+o_v
        
        return pre

    def emulation_all(self,t: torch.Tensor = None):
        if self.global_time:
            # broadcasting the time t
            t = t.repeat((self.mask_m.shape[0],1))

        emulate_m = torch.zeros([self.mask_m.shape[0], self.mask_m.shape[1], t.shape[1]])

        h = torch.exp(self.log_h)
        phi = torch.exp(self.log_phi)
        for i in range(t.shape[1]):
            # for each time stamps, predict the emulator predict value
            tt = t[:,i]
            emu = h * torch.exp(-phi*(tt.reshape((len(tt),1))-self.tau)**2) + self.o
            emulate_m[:,:,i] = emu

        return emulate_m


    def forward(self,t,g):
        ## output f_g(t) predictions

        if self.global_time:
            u = u[locate]
            s = s[locate]
            ## when use global time, t is a single value
            T = t.repeat((dim,1))

            ## calculate emulator vector
            h = torch.exp(self.log_h)
            phi = torch.exp(self.log_phi)
            emu = h[locate,:] * torch.exp(-phi[locate,:]*(T.reshape((dim,1))-self.tau[locate,:])**2) + self.o[locate,:]
            
            ## Use the Emulator matrix to predict alpha
            emu = emu * self.grn[locate,:]
            alpha_unconstr = emu.sum(dim=1)
            alpha_unconstr = alpha_unconstr + self.alpha_unconstr[locate]
            
            ## Generate kinetic rate
            beta = torch.clamp(F.softplus(self.beta_mean_unconstr[locate]), 0, 50)
            gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr[locate]), 0, 50)
            alpha = torch.clamp(F.softplus(alpha_unconstr),0,50)

            ## Predict velocity
            du = alpha - beta*u
            ds = beta*u - gamma*s

            du = du.reshape((dim,1))
            ds = ds.reshape((dim,1))

            v = torch.concatenate([du,ds],axis = 1)

        else:
            ## calculate emulator value
            ## output the f_g(t)
            
            ## Build Emulator matrix for gene g
            
            h = torch.exp(self.log_h)[g,:].view(-1)
            phi = torch.exp(self.log_phi)[g,:].view(-1)
            tau = self.tau[g,:].view(-1)
            o = self.o[g,:].view(-1)
            w = self.grn[g,:].view(-1)
            bias = self.alpha_unconstr_bias[g]

            #emu = h[locate,:] * torch.exp(-phi[locate,:]*(T.reshape((dim,1))-self.tau[locate,:])**2) + self.o[locate,:]
            emu = h * torch.exp(-phi*(t - tau)**2) + o

            ## Use the Emulator matrix to predict alpha
            #emu = emu * self.grn[locate,:]
            emu = emu * w
            
            alpha_unconstr = emu.sum()
            #alpha_unconstr = alpha_unconstr + self.alpha_unconstr_bias[locate]
            alpha_unconstr = alpha_unconstr + bias

            ## Generate transcription kinetic rate for time t
            alpha = torch.clamp(alpha_unconstr,0,)
            alpha = F.softsign(alpha)

        return alpha
            
class Decoder(nn.Module):
    """
    Decoder global or gene specific latent time
    Using VeloVI way to encode gene expression into latent time
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        global_time: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.Decoder = FCLayers(
            n_in = n_input,
            n_out = n_hidden,
            n_hidden = n_hidden,
            n_layers = n_layers,
            use_batch_norm = "both",
            use_layer_norm = "both",
            activation_fn=torch.nn.ReLU,
            dropout_rate = dropout_rate,
        )

        if global_time:
            self.t_encoder = nn.Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())
        else:
            self.t_encoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())
            

    def forward(self, z:torch.Tensor) -> tuple:
        rho_first = self.Decoder(z)
        px_t = self.t_encoder(rho_first)

        return px_t