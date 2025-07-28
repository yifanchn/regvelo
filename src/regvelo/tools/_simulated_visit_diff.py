import numpy as np

from anndata import AnnData
from scipy.stats import ttest_rel

def simulated_visit_diff(
    adata: AnnData,
    adata_perturb: AnnData,
    terminal_states: list[str]
    ) -> tuple[list[float], list[float]]:
    r"""Compute difference scores and p-values for terminal states between baseline and perturbation simulations.

    Parameters
    -------
    adata
        Annotated data object (unperturbed) with ``obs["visits"]``.
    adata_perturb
        Annotated data object (perturbed) with ``obs["visits"]``.
    terminal_states
        Labels of terminal states corresponding to cells in ``adata.obs["term_states_fwd"]``.

    Returns
    -------
    
    - Mean density difference (perturbed - control) for each terminal state.
    - P-values from paired t-test for each terminal state.
    """

    if "visits" not in adata.obs:
        raise KeyError("Please run `rgv.tl.markov_density_simulation` for the unperturbed system first.")

    if "visits" not in adata_perturb.obs:
        raise KeyError("Please run `rgv.tl.markov_density_simulation` for the perturbed system first.")

    # extract visit counts per terminal state
    cont_sim = []
    pert_sim = []
    for ts in terminal_states:
        terminal_indices_sub = np.where(adata.obs["term_states_fwd"].isin([ts]))[0]
        arrivals = adata.obs["visits"].iloc[terminal_indices_sub]
        arrivals_p = adata_perturb.obs["visits"].iloc[terminal_indices_sub]
        
        cont_sim.append(arrivals)
        pert_sim.append(arrivals_p)

    dd_score = []
    dd_sig = []
    for i in range(len(terminal_states)):
        _, p_value = ttest_rel(pert_sim[i], cont_sim[i])
        dd_score.append((np.mean(pert_sim[i])) - (np.mean(cont_sim[i])))
        dd_sig.append(p_value)
    
    return dd_score, dd_sig