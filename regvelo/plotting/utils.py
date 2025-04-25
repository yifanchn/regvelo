import numpy as np
import matplotlib.pyplot as plt
import scvelo as scv

from .._perturbation import in_silico_block_simulation

def set_plotting_style(style : str = "scvelo",
                       dpi_save : int = 400,
                       dpi : int =80, 
                       transpartent : bool = True,
                       color_map : str ="viridis", 
                       fontsize : int =14,
                       ) -> None:
    """
    Set plotting style for all figures.

    Parameters
    ----------
    style : str, optional
        The matplotlib style to use. Default is "scvelo".
    dpi_save : int, optional
        Dots per inch for saved figures. Default is 400.
    dpi : int, optional
        Dots per inch for displayed figures. Default is 80.
    transpartent : bool, optional
        Whether saved figures should have a transparent background. Default is True.
    color_map : str, optional
        The default color map to use. Default is "viridis".
    fontsize : int, optional
        Base font size for plots. Default is 14.
    """

    plt.rcParams["svg.fonttype"] = "none"
    scv.settings.set_figure_params(
        style=style,
        dpi_save=dpi_save,
        dpi=dpi,
        transparent=transpartent,
        fontsize=fontsize,
        color_map=color_map,
    )

def calculate_entropy(prob_matrix : np.array
                      ) -> np.array:
    """
    Calculate entropy for each row in a cell fate probability matrix.

    Parameters
    ----------
    prob_matrix : np.ndarray
        A 2D NumPy array of shape (n_cells, n_lineages) where each row represents
        a cell's fate probabilities across different lineages. Each row should sum to 1.

    Returns
    -------
    entropy : np.ndarray
        A 1D NumPy array of length n_cells containing the entropy values for each cell.
    """
    log_probs = np.zeros_like(prob_matrix)
    mask = prob_matrix != 0
    np.log2(prob_matrix, where=mask, out=log_probs)

    entropy = -np.sum(prob_matrix * log_probs, axis=1)
    return entropy

def calculate_depletion_score(depletion_likelihood : np.array
                              ) -> np.array:
    """
    Calculate depletion score by rescaling depletion likelihood.

    Parameters
    ----------
        depletion_likelihood: 1D NumPy array of depletion likelihood values.

    Returns
    -------
        depletion_score: 1D numpy array of depletion scores.
    """
    depletion_score = 2*(0.5 - depletion_likelihood)
    return depletion_score
