import pandas as pd
import numpy as np

import cellrank as cr
from anndata import AnnData
from scvelo import logging as logg
from typing import Literal

from .._model import REGVELOVI
from ._utils import split_elements, combine_elements
from ..metrics._abundance_test import abundance_test


def TFScanning_func(
    model: str, 
    adata: AnnData, 
    cluster_label: str = None,
    terminal_states: str | list[str] | dict[str, list[str]] | pd.Series = None,
    KO_list: str | list[str] | dict[str, list[str]] | pd.Series = None,
    n_states: int | list[int] = None,
    cutoff: float | list[float] = 1e-3,
    method: Literal["likelihood", "t-statistics"] = "likelihood",
    combined_kernel: bool = False,
    ) -> dict[str, float | pd.DataFrame]:
    r"""Perform in silico TF regulon knock-out screening

    Parameters
    ----------
    model
        Path to the saved RegVelo model.
    adata
        Annotated data matrix.
    cluster_label
        Key in ``adata.obs`` to associate names and colors with ``terminal_states``.
    terminal_states
        Subset of ``macrostates``.
    KO_list
        List of TF names or combinations (e.g., ["geneA", "geneB_geneC"]).
    n_states
        Number of macrostates to compute.
    cutoff
        Threshold for determing which links need to be muted,
    method
        Method for quantifying perturbation effect.
    combined_kernel
        Whether to use a combined kernel (0.8*VelocityKernel + 0.2*ConnectivityKernel)

    Returns
    -------
    Dictionary with keys ``TF``, ``coefficient``, and ``pvalue`` summarizing KO effects.
    """

    reg_vae = REGVELOVI.load(model, adata)
    adata = reg_vae.add_regvelo_outputs_to_adata(adata = adata)
    raw_GRN = reg_vae.module.v_encoder.fc1.weight.detach().clone()
    perturb_GRN = reg_vae.module.v_encoder.fc1.weight.detach().clone()

    ## define kernel matrix
    vk = cr.kernels.VelocityKernel(adata)
    vk.compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
    
    if combined_kernel:
        g2 = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)
    else:
        g2 = cr.estimators.GPCCA(vk)

    ## evaluate the fate prob on original space
    g2.compute_macrostates(n_states=n_states, n_cells = 30, cluster_key=cluster_label)
    # set a high number of states, and merge some of them and rename
    if terminal_states is None:
        g2.predict_terminal_states()
        terminal_states = g2.terminal_states.cat.categories.tolist()
    g2.set_terminal_states(
        terminal_states
    )        
    g2.compute_fate_probabilities(solver="direct")
    fate_prob = g2.fate_probabilities
    sampleID = adata.obs.index.tolist()
    fate_name = fate_prob.names.tolist()
    fate_prob = pd.DataFrame(fate_prob,index= sampleID,columns=fate_name)
    fate_prob_original = fate_prob.copy()

    ## create dictionary
    terminal_id = terminal_states.copy()
    terminal_type = terminal_states.copy()
    for i in terminal_states:
        for j in [1,2,3,4,5,6,7,8,9,10]:
            terminal_id.append(i+"_"+str(j))
            terminal_type.append(i)
    terminal_dict = dict(zip(terminal_id, terminal_type))
    n_states = len(g2.macrostates.cat.categories.tolist())
    
    coef = []
    pvalue = []
    for tf in split_elements(KO_list):
        perturb_GRN = raw_GRN.clone()
        vec = perturb_GRN[:,[i in tf for i in adata.var.index.tolist()]].clone()
        vec[vec.abs() > cutoff] = 0
        perturb_GRN[:,[i in tf for i in adata.var.index.tolist()]]= vec
        reg_vae_perturb = REGVELOVI.load(model, adata)
        reg_vae_perturb.module.v_encoder.fc1.weight.data = perturb_GRN
            
        adata_target = reg_vae_perturb.add_regvelo_outputs_to_adata(adata = adata)
        ## perturb the regulations
        vk = cr.kernels.VelocityKernel(adata_target)
        vk.compute_transition_matrix()
        ck = cr.kernels.ConnectivityKernel(adata_target).compute_transition_matrix()
        
        if combined_kernel:
            g2 = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)
        else:
            g2 = cr.estimators.GPCCA(vk)
        ## evaluate the fate prob on original space
        n_states_perturb = n_states
        while True:
            try:
                # Perform some computation in f(a)
                g2.compute_macrostates(n_states=n_states_perturb, n_cells = 30, cluster_key=cluster_label)
                break
            except:
                # If an error is raised, increment a and try again, and need to recompute double knock-out reults
                n_states_perturb += 1
                vk = cr.kernels.VelocityKernel(adata)
                vk.compute_transition_matrix()
                ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
                if combined_kernel:
                    g = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)
                else:
                    g = cr.estimators.GPCCA(vk)
                ## evaluate the fate prob on original space
                g.compute_macrostates(n_states=n_states_perturb, n_cells = 30,cluster_key=cluster_label)
                ## set a high number of states, and merge some of them and rename
                if terminal_states is None:
                    g.predict_terminal_states()
                    terminal_states = g.terminal_states.cat.categories.tolist()
                g.set_terminal_states(
                    terminal_states
                )
                g.compute_fate_probabilities(solver="direct")
                fate_prob = g.fate_probabilities
                sampleID = adata.obs.index.tolist()
                fate_name = fate_prob.names.tolist()
                fate_prob = pd.DataFrame(fate_prob,index= sampleID,columns=fate_name)
                
        ## intersection the states
        terminal_states_perturb = g2.macrostates.cat.categories.tolist()
        terminal_states_perturb = list(set(terminal_states_perturb).intersection(terminal_states))
        
        g2.set_terminal_states(
            terminal_states_perturb
        )
        g2.compute_fate_probabilities(solver="direct")
        fb = g2.fate_probabilities
        sampleID = adata.obs.index.tolist()
        fate_name = fb.names.tolist()
        fb = pd.DataFrame(fb,index= sampleID,columns=fate_name)
        fate_prob2 = pd.DataFrame(columns= terminal_states, index=sampleID)   
        
        for i in terminal_states_perturb:
            fate_prob2.loc[:,i] = fb.loc[:,i]

        fate_prob2 = fate_prob2.fillna(0)
        arr = np.array(fate_prob2.sum(0))
        arr[arr!=0] = 1
        fate_prob = fate_prob * arr
        
        y = [0] * fate_prob.shape[0] + [1] * fate_prob2.shape[0]
        fate_prob2.index = [i + "_perturb" for i in fate_prob2.index]
        test_result = abundance_test(fate_prob, fate_prob2, method)
        coef.append(test_result.loc[:, "coefficient"])
        pvalue.append(test_result.loc[:, "FDR adjusted p-value"]) 
        logg.info("Done "+ combine_elements([tf])[0])
        fate_prob = fate_prob_original.copy()

    d = {'TF': KO_list, 'coefficient': coef, 'pvalue': pvalue}   
    return d