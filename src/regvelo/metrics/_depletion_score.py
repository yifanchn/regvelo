import pandas as pd
from anndata import AnnData
from typing import Sequence, Any
import cellrank as cr

from ._abundance_test import abundance_test

def depletion_score(
        perturbed : dict[str, AnnData],
        baseline : AnnData,
        terminal_state : str | Sequence[str],
        **kwargs : Any,
        ) -> tuple[pd.DataFrame, dict[str, AnnData]]:
    """Compute depletion scores.

    Parameters
    ----------
    perturbed
        Dictionary mapping TF candidates to their perturbed AnnData objects.
    baseline
        Annotated data matrix with precomputed fate probabilities (under key `'lineages_fwd'`).
    terminal_state
        One or more terminal states for which depletion scores are computed.
    **kwargs
        Additional keyword arguments passed to CellRank estimator methods:
        - For `compute_macrostates`: "n_states", "n_cells", "cluster_key", "method"
        - For `compute_fate_probabilities`: "solver", "tol"

    Returns
    -------
    tuple
        - pd.DataFrame: DataFrame summarizing depletion scores and significance statistics.
        - dict: Updated dictionary mapping TFs to perturbed AnnData objects with cell fate probabilities.
    """
    macro_kwargs = {k: kwargs[k] for k in ("n_states", "n_cells", "cluster_key", "method") if k in kwargs}
    compute_fate_probabilities_kwargs = {k: kwargs[k] for k in ("solver", "tol") if k in kwargs}

    if "lineages_fwd" not in baseline.obsm:
        raise KeyError("Lineages not found in baseline.obsm. Please compute lineages first.")

    if isinstance(terminal_state, str):
        terminal_state = [terminal_state]

    # selecting indices of cells that have reached a terminal state
    ct_indices = {
        ct: baseline.obs["term_states_fwd"][baseline.obs["term_states_fwd"] == ct].index.tolist()
        for ct in terminal_state
    }

    fate_prob_perturb = {}
    for TF, adata_target_perturb in perturbed.items():
        vk = cr.kernels.VelocityKernel(adata_target_perturb).compute_transition_matrix()
        vk.write_to_adata()
        estimator = cr.estimators.GPCCA(vk)

        estimator.compute_macrostates(**macro_kwargs)

        estimator.set_terminal_states(ct_indices)
        estimator.compute_fate_probabilities(**compute_fate_probabilities_kwargs)

        perturbed[TF] = adata_target_perturb

        perturbed_prob = pd.DataFrame(
            adata_target_perturb.obsm["lineages_fwd"], 
            columns=adata_target_perturb.obsm["lineages_fwd"].names.tolist()
            )[terminal_state]

        fate_prob_perturb[TF] = perturbed_prob

    fate_prob_raw = pd.DataFrame(
        baseline.obsm["lineages_fwd"], 
        columns=baseline.obsm["lineages_fwd"].names.tolist()
        )

    dfs = []
    for TF, perturbed_prob in fate_prob_perturb.items():
        stats = abundance_test(prob_raw=fate_prob_raw, prob_pert=perturbed_prob)
        df = pd.DataFrame(
            {
                "Depletion score": stats.iloc[:, 0].tolist(),
                "p-value": stats.iloc[:, 1].tolist(),
                "FDR adjusted p-value": stats.iloc[:, 2].tolist(),
                "Terminal state": stats.index.tolist(),
                "TF": [TF] * stats.shape[0],
            }
        )
        dfs.append(df)

    df = pd.concat(dfs)

    df["Depletion score"] = 2 * (0.5 - df["Depletion score"])

    return df, perturbed



