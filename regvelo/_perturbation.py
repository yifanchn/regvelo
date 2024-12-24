import torch
import pandas as pd
import numpy as np

from scipy.stats import ranksums, ttest_ind
from sklearn.metrics import roc_auc_score

import cellrank as cr
from anndata import AnnData
from scvelo import logging as logg
import os,shutil
from typing import Dict, Optional, Sequence, Tuple, Union

from ._model import REGVELOVI


def split_elements(character_list):
    """split elements."""
    result_list = []
    for element in character_list:
        if '_' in element:
            parts = element.split('_')
            result_list.append(parts)
        else:
            result_list.append([element])
    return result_list

def combine_elements(split_list):
    """combine elements."""
    result_list = []
    for parts in split_list:
        combined_element = "_".join(parts)
        result_list.append(combined_element)
    return result_list

def get_list_name(lst):
    names = []
    for name, obj in lst.items():
        names.append(name)
    return names

def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def in_silico_block_simulation(
        model: str,
        adata: AnnData,
        TF:str,
        effects: int = 0,
        cutoff: int = 1e-3,
        customized_GRN: torch.Tensor = None):
    """ Perform in silico TF regulon knock-out

    Parameters
    ----------
    model
        The saved address for the RegVelo model.
    adata
        Anndata objects.
    TF
        The candidate TF, need to knockout its regulon.
    effect
        The coefficient for replacing the weights in GRN
    cutoff
        The threshold for determing which links need to be muted,
    customized_GRN
        The customized perturbed GRN
    """

    reg_vae_perturb = REGVELOVI.load(model,adata)

    perturb_GRN = reg_vae_perturb.module.v_encoder.fc1.weight.detach().clone()

    if customized_GRN is None:
        perturb_GRN[(perturb_GRN[:,[i == TF for i in adata.var.index]].abs()>cutoff).cpu().numpy().reshape(-1),[i == TF for i in adata.var.index]] = effects
        reg_vae_perturb.module.v_encoder.fc1.weight.data = perturb_GRN
    else:
        device = perturb_GRN.device
        customized_GRN =  customized_GRN.to(device)
        reg_vae_perturb.module.v_encoder.fc1.weight.data = customized_GRN
    
    adata_target_perturb = reg_vae_perturb.add_regvelo_outputs_to_adata(adata = adata)
    
    return adata_target_perturb,reg_vae_perturb

def abundance_test(prob_raw: pd.DataFrame, prob_pert: pd.DataFrame, method: str = "likelihood") -> pd.DataFrame:
    """Perform an abundance test between two probability datasets.

    Parameters
    ----------
    prob_raw : pd.DataFrame
        Raw probabilities dataset.
    prob_pert : pd.DataFrame
        Perturbed probabilities dataset.
    method : str, optional (default="likelihood")
        Method to calculate scores: "likelihood" or "t-statistics".

    Returns
    -------
    pd.DataFrame
        Dataframe with coefficients, p-values, and FDR adjusted p-values.
    """
    y = [1] * prob_raw.shape[0] + [0] * prob_pert.shape[0]
    X = pd.concat([prob_raw, prob_pert], axis=0)

    table = []
    for i in range(prob_raw.shape[1]):
        pred = np.array(X.iloc[:, i])
        if np.sum(pred) == 0:
            score, pval = np.nan, np.nan
        else:
            pval = ranksums(pred[np.array(y) == 0], pred[np.array(y) == 1])[1]
            if method == "t-statistics":
                score = ttest_ind(pred[np.array(y) == 0], pred[np.array(y) == 1])[0]
            elif method == "likelihood":
                score = roc_auc_score(y, pred)
            else:
                raise NotImplementedError("Supported methods are 't-statistics' and 'likelihood'.")

        table.append(np.expand_dims(np.array([score, pval]), 0))

    table = np.concatenate(table, axis=0)
    table = pd.DataFrame(table, index=prob_raw.columns, columns=["coefficient", "p-value"])
    table["FDR adjusted p-value"] = p_adjust_bh(table["p-value"].tolist())
    return table

def TFScanning_func(
    model:str, 
    adata:AnnData, 
    cluster_label: Optional[str] = None,
    terminal_states: Optional[Union[str, Sequence[str], Dict[str, Sequence[str]], pd.Series]] = None,
    KO_list: Optional[Union[str, Sequence[str], Dict[str, Sequence[str]], pd.Series]] = None,
    n_states: Optional[Union[int, Sequence[int]]] = None,
    cutoff: Optional[Union[int, Sequence[int]]] = 1e-3,
    method: Optional[str] = "likelihood",
    combined_kernel: Optional[bool] = False,
):

    """ Perform in silico TF regulon knock-out screening

    Parameters
    ----------
    model
        The saved address for the RegVelo model.
    adata
        Anndata objects.
    cluster_label:
        Key in :attr:`~anndata.AnnData.obs` to associate names and colors with :attr:`terminal_states`.
    terminal_states:
        subset of :attr:`macrostates`.
    KO_list
        List of TF combinations to simulate knock-out (KO) effects
        Can be single TF e.g. geneA
        or double TFs e.g. geneB_geneC
        example input: ["geneA","geneB_geneC"]
    n_states
        Number of macrostates to compute.
    cutoff
        The threshold for determing which links need to be muted,
    method
        Quantify perturbation effects via `likelihood` or `t-statistics`
    combined_kernel
        Use combined kernel (0.8*VelocityKernel + 0.2*ConnectivityKernel)
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


def TFscreening(
    adata:AnnData, 
    prior_graph:torch.Tensor,
    lam: Optional[int] = 1,
    lam2: Optional[int] = 0,
    soft_constraint:Optional[bool] = True,
    TF_list: Optional[Union[str, Sequence[str], Dict[str, Sequence[str]], pd.Series]] = None,
    cluster_label: Optional[str] = None,
    terminal_states: Optional[Union[str, Sequence[str], Dict[str, Sequence[str]], pd.Series]] = None,
    KO_list: Optional[Union[str, Sequence[str], Dict[str, Sequence[str]], pd.Series]] = None,
    n_states: Optional[Union[int, Sequence[int]]] = 8,
    cutoff: Optional[float] = 1e-3,
    max_nruns: Optional[float] = 5,
    method: Optional[str] = "likelihood",
    dir:Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Perform in silico TF regulon knock-out screening

    Parameters
    ----------
    adata
        Anndata objects.
    prior_graph
        A prior graph for RegVelo inference
    lam
        Regularization parameter for controling the strengths of adding prior knowledge.
    lam2
        Regularization parameter for controling the strengths of L1 regularization to the Jacobian matrix.
    soft_constraint
        Apply soft constraint mode RegVelo.
    TF_list
        The TF list used for RegVelo inference.
    cluster_label:
        Key in :attr:`~anndata.AnnData.obs` to associate names and colors with :attr:`terminal_states`.
    terminal_states:
        subset of :attr:`macrostates`.
    KO_list
        List of TF combinations to simulate knock-out (KO) effects
        can be single TF e.g. geneA
        or double TFs e.g. geneB_geneC
        example input: ["geneA","geneB_geneC"]
    n_states
        Number of macrostates to compute.
    cutoff
        The threshold for determing which links need to be muted (<cutoff).
    max_nruns
        maximum number of runs, soft constrainted RegVelo model need to have repeat runs to get stable perturbation results.
    dir
        the location to save temporary datasets.
    method
        Use either `likelihood` or `t-statistics` to quantify perturbation effects
    """

    if soft_constraint is not True:
        max_nruns = 1
    
    if dir is None:
        dir = os.getcwd()
    
    # Ensure the output directory exists
    output_dir = os.path.join(dir, "perturb_repeat_runs")
    os.makedirs(output_dir, exist_ok=True)

    for nrun in range(max_nruns):
        logg.info("training model...")
        REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
        reg_vae = REGVELOVI(adata, W=prior_graph, regulators=TF_list, soft_constraint=soft_constraint, lam=lam, lam2=lam2)
        reg_vae.train()
        
        logg.info("save model...")
        model_name = f"rgv_model_{nrun}"
        coef_name = f"coef_{nrun}.csv"
        pval_name = f"pval_{nrun}.csv"
        
        model = os.path.join(output_dir, model_name)
        coef_save = os.path.join(output_dir, coef_name)
        pval_save = os.path.join(output_dir, pval_name)
        
        reg_vae.save(model)

        logg.info("inferring perturbation...")
        while True:
            try:
                perturb_screening = TFScanning_func(model, adata, cluster_label, terminal_states, KO_list, n_states, cutoff, method)
                
                coef = pd.DataFrame(np.array(perturb_screening['coefficient']))
                coef.index = perturb_screening['TF']
                coef.columns = get_list_name(perturb_screening['coefficient'][0])

                pval = pd.DataFrame(np.array(perturb_screening['pvalue']))
                pval.index = perturb_screening['TF']
                pval.columns = get_list_name(perturb_screening['pvalue'][0])

                # Handle NaN rows
                rows_with_nan = coef.isna().any(axis=1)
                coef.loc[rows_with_nan, :] = np.nan
                pval.loc[rows_with_nan, :] = np.nan

                coef.to_csv(coef_save)
                pval.to_csv(pval_save)

                break
            except Exception as e:
                # Catch the exception and retry training
                print(f"Perturbation screening encountered an error: {e}, retraining model...")
                shutil.rmtree(model, ignore_errors=True)
                REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
                reg_vae = REGVELOVI(adata, W=prior_graph, regulators=TF_list, soft_constraint=soft_constraint, lam=lam, lam2=lam2)
                reg_vae.train()

                logg.info("save model...")
                reg_vae.save(model)

    # Read all repeat run results
    pert_coef = []
    pert_p = []
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if filename.startswith("coef_") and filename.endswith(".csv"):
            df = pd.read_csv(file_path, index_col=0)
            pert_coef.append(df)
        elif filename.startswith("pval_") and filename.endswith(".csv"):
            df = pd.read_csv(file_path, index_col=0)
            pert_p.append(df)
    
    # Combine DataFrames
    coef = pd.concat(pert_coef).groupby(level=0).mean()
    pval = pd.concat(pert_p).groupby(level=0).mean()

    return coef, pval



    
