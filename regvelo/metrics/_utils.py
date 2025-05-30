import numpy as np
from typing import Sequence, Union

def p_adjust_bh(p : Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Perform Benjamini-Hochberg p-value correction for multiple hypothesis testing.

    Parameters
    ----------
    p
        Raw p-values to adjust.

    Returns
    -------
    np.ndarray
        Adjusted p-values using the Benjamini-Hochberg method.
    """
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def calculate_entropy(prob_matrix : np.ndarray) -> np.ndarray:
    """Calculate entropy for each row in a cell fate probability matrix.

    Parameters
    ----------
    prob_matrix
        A 2D NumPy array of shape (n_cells, n_lineages) where each row represents
        a cell's fate probabilities across different lineages. Each row is expected to sum to 1.

    Returns
    -------
    entropy
        A 1D NumPy array of shape (n_cells,) containing the entropy values for each cell.
    """
    log_probs = np.zeros_like(prob_matrix)
    mask = prob_matrix != 0
    np.log2(prob_matrix, where=mask, out=log_probs)

    entropy = -np.sum(prob_matrix * log_probs, axis=1)
    return entropy