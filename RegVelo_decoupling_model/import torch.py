import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torchdiffeq import odeint
from typing import Optional
from typing_extensions import Literal
import numpy as np
from .module import velocity_encoder, Decoder
from ._utils import get_step_size, normal_kl, log_zinb, log_nb
from torch.distributions import Normal
from scvi.nn import Encoder
import torchode as to

class regODE(nn.Module):
    """
    Class to automatically infer cellular dynamics using VAE and neural ODE.

    Parameters
    ----------
    device
        The torch device.
    n_int
        The dimensionality of the input.
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
        Whether to include `BatchNorm` layer.
        (Default: `False`)
    ode_method
        Solver for integration.
        (Default: `'euler'`)
    step_size
        The step size during integration.
    alpha_recon_lec
        Scaling factor for reconstruction loss from encoder-derived latent space.
        (Default: 0.5)
    alpha_recon_lode
        Scaling factor for reconstruction loss from ODE-solver latent space.
        (Default: 0.5)
    alpha_kl
        Scaling factor for KL divergence.
        (Default: 1.0)
    loss_mode
        The mode for calculating the reconstruction error.
        (Default: `'nb'`)
    """

    def __init__(
        self,
        device,
        regulator_index,
        target_index,
        n_int: int,
        n_latent: int = 10,
        n_vae_hidden: int = 128,
        global_time: bool = False,
        ode_method: str = 'euler',
        alpha_recon_reg: float = 0.5,
        alpha_recon_target: float = 0.5,
        alpha_kl: float = 1.,
        W: torch.Tensor = None,
        corr_m: torch.Tensor = None,
        alpha_unconstr_init: Optional[np.ndarray] = None,
        log_h_int: torch.Tensor = None,
        ratio: float = 0.8,
    ):
        super().__init__()
        n_genes = int(n_int/2)
        self.n_genes = n_genes
        self.n_int = n_int
        self.n_latent = n_latent
        self.global_time = global_time
        self.n_vae_hidden = n_vae_hidden
        self.ode_method = ode_method
        self.regulator_index = regulator_index
        self.target_index = target_index
        self.n_regulators = int(sum(regulator_index))
        self.n_targets = int(sum(target_index))

        self.alpha_recon_reg = alpha_recon_reg
        self.alpha_recon_target = alpha_recon_target
        self.alpha_kl = alpha_kl
        self.device = device
    
        if W is None:
            W = (torch.FloatTensor(int(n_int/2), int(n_int/2)).uniform_() > ratio).int().to(device)
        else:
            W = W.to(device)

        self.z_encoder = Encoder(
            n_int,
            n_latent,
            n_layers=1,
            n_hidden=n_vae_hidden,
            dropout_rate=0.1,
            distribution="normal",
            use_batch_norm=True,
            use_layer_norm=True,
            var_activation=torch.nn.Softplus(),
            activation_fn=torch.nn.ReLU,
        ).to(self.device)
        self.decoder = Decoder(n_input = n_latent, n_output = self.n_targets, n_hidden = n_vae_hidden,n_layers = 1,global_time = global_time).to(self.device)

        ## generate emulator and velocity
        self.v_encoder = velocity_encoder(n_int = self.n_targets, alpha_unconstr_init = alpha_unconstr_init, log_h_int = log_h_int, global_time = global_time, W = W, W_int = corr_m).to(self.device)
        self.v_encoder._set_mask_grad()

    def forward(self, x: torch.Tensor, n_samples=1) -> tuple:
        """
        Given the transcriptomes of cells, this function derives the time and latent space of the cells, as well as reconstructs the transcriptomes.

        Parameters
        ----------
        x
            The input data.

        Returns
        ----------
        5-tuple of :class:`torch.Tensor`
            Tensors for loss, including:
            1) total loss,
            2) reconstruction loss from encoder-derived latent space,
            3) reconstruction loss from ODE-solver latent space,
            4) KL divergence,
            5) divergence between encoder-derived latent space and ODE-solver latent space
        """

        ## get the time and latent space through Encoder
        qz_m, qz_v, z = self.z_encoder(x)
        qz_logvar = torch.log(qz_v)

        ## split x into spliced and unspliced
        unspliced = x[:,0:int(x.shape[1]/2)].T
        spliced = x[:,int(x.shape[1]/2):int(x.shape[1])].T

        ## split spliced matrix into regulator spliced and target spliced
        regulator_spliced = spliced[self.regulator_index,:]
        target_spliced = spliced[self.target_index,:]
        target_unspliced = unspliced[self.target_index,:]
        
        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)
        
        if self.global_time:
            NotImplementedError

        else:
            T = self.decoder(z) * 20
            T = T.T

            ## T is a N*G matrix (G:number of target gene, N:number of cells)
            ## ordering each raw of T
            indices = torch.argsort(T, dim=1)

            t_eval = torch.gather(T, dim = 1, index = indices)
            target_spliced = torch.gather(target_spliced, dim = 1, index = indices)
            target_unspliced = torch.gather(target_unspliced, dim = 1, index = indices)
            
            index2 = (t_eval[:,:-1] != t_eval[:,1:])
            ## estimate the shift value
            subtraction_values,_ = torch.where((t_eval[:,1:] - t_eval[:,:-1])>0, (t_eval[:,1:] - t_eval[:,:-1]), torch.inf).min(axis=1)
            ## replace inf value = 0
            subtraction_values[subtraction_values == float("Inf")] = 0

            true_tensor = torch.ones((t_eval.shape[0],1), dtype=torch.bool)
            index2 = torch.cat((index2, true_tensor.to(index2.device)),dim=1) ## index2 is used to get unique time points as odeint requires strictly increasing/decreasing time points
            
            subtraction_values = subtraction_values[None, :].repeat(index2.shape[1], 1).T
            t_eval[index2 == False] -= subtraction_values[index2 == False]*0.1
            ## extract initial target gene expression value
            x0 = torch.cat((target_unspliced[:,0].reshape((target_unspliced.shape[0],1)),target_spliced[:,0].reshape((target_spliced.shape[0],1))),dim = 1)
            #x0 = x0.double()
            ## set up G batches, Each G represent a module (a target gene centerred regulon)
            ## infer the observe gene expression through ODE solver based on x0, t, and velocity_encoder
            indices = torch.arange(0, x0.shape[0]).reshape((x0.shape[0],1))

            term = to.ODETerm(self.v_encoder)
            step_method = to.Dopri5(term=term)
            #step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
            step_size_controller = to.FixedStepController()
            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            #jit_solver = torch.jit.script(solver)
            dt0 = torch.full((x0.shape[0],), 1)
            sol = solver.solve(to.InitialValueProblem(y0=x0, t_eval=t_eval),dt0=dt0)

        # the solved results are saved in sol.ys [the number of subsystems, time_stamps, [u,s]]
        pred_u = sol.ys[:,:,0]
        pred_s = sol.ys[:,:,1]
        
        # generate the emulator predict value (for regulator)
        emulation = self.v_encoder.emulation_all(T)
        emulation = emulation.mean(dim=0)

        ## reconstruct the input through Decoder and compute reconstruction loss
        recon_loss_regulator = F.mse_loss(regulator_spliced, emulation, reduction='none').sum(-1).mean()
        recon_loss_target = F.mse_loss(target_unspliced, pred_u, reduction='none').sum(-1).mean() + F.mse_loss(target_spliced, pred_s, reduction='none').sum(-1).mean()
        ## compute KL divergence and z divergence
        pz_mean = torch.zeros_like(qz_m)
        pz_logvar = torch.zeros_like(qz_m)
        kl_div = normal_kl(qz_m, qz_logvar, pz_mean, pz_logvar).sum(-1).mean()

        loss = self.alpha_recon_reg * recon_loss_regulator + self.alpha_recon_target * recon_loss_target + self.alpha_kl * kl_div
        #loss = recon_loss_ode + self.alpha_kl * kl_div

        return loss, recon_loss_regulator, recon_loss_target, kl_div





def SolveInitialValueProblem(self, x0, t0, t_eval=t_eval):
    
    ## generate the prediction of unspliced/spliced readout at time t for every gene
    ## use torchquad integral, different with torchode, the t_eval no longer need to be ordered
    
    ## get the kinetic parameters
    beta = torch.clamp(F.softplus(self.alpha_encoder.beta_mean_unconstr), 0, 50)
    gamma = torch.clamp(F.softplus(self.alpha_encoder.gamma_mean_unconstr), 0, 50)
    alpha_max = torch.clamp(F.softplus(self.alpha_encoder.alpha_unconstr_max),0,50)

    ## define integral function
    def integral_alpha_beta(self, t_eval, t0, beta, g):
        f_i = lamba t: self.f_t(t,g)*torch.exp(beta[g]*t)
        integration_doman = [[t0[g],t_eval]]
        result = simp.integrate(f_i, dim=1, N=101, integration_domain=integration_domain)
        return result
        
    def integral_alpha_gamma(self, t_eval,t0, gamma, g):
        f_i = lamba t: self.f_t(t,g)*torch.exp(gamma[g]*t)
        integration_domain = [[t0[g],t_eval]]
        result = simp.integrate(f_i, dim=1, N=101, integration_domain=integration_domain)
        return result

    ## get the initial condition (i.e. u,s = 0,0)
    u0 = x0[:,0].view(-1)
    s0 = x0[:,1].view(-1)
    pre_u = torch.zeros(t_eval.shape)
    pre_s = torch.zeros(t_eval.shape)

    ## build for loop to generate readout for each targets
    for g, t in enumerate(t_eval):
        u0g = u0[g]
        s0g = s0[g]
        ## calculate integral for gene g
        integral_tensor_alpha_beta = torch.tensor(list(map(lambda t_eval: integral_alpha_beta(t_eval,t0=t0,beta = beta,g = g), t)))
        integral_tensor_alpha_gamma = torch.tensor(list(map(lambda t_eval: integral_alpha_gamma(t_eval,t0=t0,gamma = gamma,g = g), t)))
        
        ug = u0g*torch.exp(-beta[g]*t) + alpha_max[g]*torch.exp(-beta[g]*t)*integral_tensor_alpha_beta
        us = s0g*torch.exp(-gamma[g]*t) + 
            ( (alpha_max[g]*beta[g])/(gamma[g] - beta[g]) )*(torch.exp(-beta[g]*t)*integral_tensor_alpha_beta - torch.exp(-gamma[g]*t)*integral_tensor_alpha_gamma) +
            ( (beta[g]*u0g)/(gamma[g] - beta[g]) )*(torch.exp(-beta[g]*t) - torch.exp(-gamma[g]*t))
        
        pre_u[g,:] = ug
        pre_s[s,:] = sg
    
    return pre_u, pre_s


        
        
        
        
    