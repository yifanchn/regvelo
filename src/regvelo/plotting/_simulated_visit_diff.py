import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData
from typing import Any

from ._utils import delta_to_probability, smooth_score

def simulated_visit_diff(
    adata: AnnData,
    adata_perturb: AnnData,
    terminal_states: list[str],
    total_simulations: int,
    basis: str = None,
    **kwargs: Any
    ) -> None:
    r"""Assign and visualize smoothed absolute differences in visit counts between control and perturbation simulations.

    Parameters
    ----------
    adata
        Annotated data object of the control condition.
    adata_perturb
        Annotated data object of the perturbed condition.
    terminal_states
        List of terminal state labels.
    total_simulations
        Total number of simulations performed.
    basis
        Optional embedding basis. If given, temporarily overrides UMAP coordinates.
    **kwargs
        Optional keyword arguments:
        
        - ``key`` : Name of the observation key in :attr:`.obs` containing visit counts (default: ``"visits"``).
        - ``embedding`` : Key in :attr:`.obsm` for the embedding (default: ``"X_pca"``).
        - ``title`` : Title for the plot (default: ``"Density difference"``).
        - ``figsize`` : Size of the output plot (default: ``(4, 3.5)``).
        - ``n_neighbors`` : Number of neighbors for smoothing (default: ``10``).
        - ``color_map`` : Colormap used for plotting (default: ``"vlag"``).

    Returns
    -------
    Nothing, just plots the figure. Also updates ``adata`` with the following fields:

    - ``adata.obs[f"{key}_diff"]``: Absolute difference in visit counts, scaled, and passed through sigmoid function.
    - ``adata.obs[f"{key}_diff_smooth"]``: Smoothed version of the above for plotting.
    """

    key = kwargs.get("key", "visits")
    embedding = kwargs.get("embedding", "X_pca")
    title = kwargs.get("title", "Density difference")
    figsize = kwargs.get("figsize", (4, 3.5))
    n_neighbors = kwargs.get("n_neighbors", 10)
    color_map = kwargs.get("color_map", "vlag")

    # Validate required keys
    if key not in adata.obs or key not in adata_perturb.obs:
        raise KeyError(f"'{key}' must be present in both `adata.obs` and `adata_perturb.obs`. Please run `rgv.tl.markov_density_simulation` first.")

    if embedding not in adata.obsm:
        raise KeyError(f"Embedding '{embedding}' not found in `adata.obsm`.")

    # Temporarily replace UMAP if basis is provided
    umap_backup = None
    if basis is not None:
        key_basis = f"X_{basis}"
        if key_basis not in adata.obsm:
            raise KeyError(f"'{key_basis}' not found in `adata.obsm` for basis override.")
        umap_backup = adata.obsm["X_umap"].copy()
        adata.obsm["X_umap"] = adata.obsm[key_basis].copy()


    # Compute normalized differences and smooth
    adata.obs[f"{key}_diff"] = delta_to_probability(
        adata_perturb.obs[key] - adata.obs[key],
        k=1 / np.sqrt(total_simulations)
    )

    smooth_score(adata, key=f"{key}_diff", embedding=embedding, n_neighbors=n_neighbors)

    # Plot using Scanpy with dark background
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        sc.pl.umap(
            adata,
            color=f"{key}_diff_smooth",
            color_map=color_map,
            vcenter=0.5,
            na_color="black",
            s=300,
            colorbar_loc=None,
            outline_color=["white", "black"],
            outline_width=(0.2, 0.1),
            add_outline=True,
            show=False,
            ax=ax,
        )

        adata_subset = adata[~np.isnan(adata.obs[f"{key}_diff_smooth"])]
        sc.pl.umap(
            adata_subset,
            color=f"{key}_diff_smooth",
            color_map=color_map,
            vcenter=0.5,
            add_outline=False,
            s=300,
            show=False,
            frameon=False,
            title=title,
            colorbar_loc="bottom",
            ax=ax,
        )

        plt.tight_layout()
        plt.show()

    # Restore original UMAP if modified
    if umap_backup is not None:
        adata.obsm["X_umap"] = umap_backup