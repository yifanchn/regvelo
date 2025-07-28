
import pandas as pd
from anndata import AnnData
from typing import Any

import seaborn as sns
import matplotlib.pyplot as plt


def depletion_score(
        adata: AnnData,
        df: pd.DataFrame,
        color_label: str = "celltype_cluster",
        **kwargs: Any,
        ) -> None:
    """Plot depletion scores for transcription factors across terminal states.

    Parameters
    ----------
    adata
        Annotated data matrix of the unperturbed system, used to extract color palette info.
    df
        DataFrame containing columns "TF", "Depletion score", and "Terminal state".
    color_label
        Key in `adata.obs` used to extract categorical color palette from `adata.uns`.
    **kwargs 
        Additional keyword arguments, including:
        - fontsize : int, default 14
        - figsize : tuple, default (12, 6)
        - xlabel : str, default "TF"
        - ylabel : str, default "Depletion score"
        - legend_loc : str, default "center left"
        - legend_bbox : tuple, default (1.02, 0.5)

    Returns
    -------
    None
        Displays a searborn barplot.
    """

    fontsize = kwargs.get("fontsize", 14)
    figsize = kwargs.get("figsize", (12, 6))

    legend_loc = kwargs.get("legend_loc", "center left")
    legend_bbox = kwargs.get("legend_bbox", (1.02, 0.5))

    xlabel = kwargs.get("xlabel", "TF")
    ylabel = kwargs.get("ylabel", "Depletion score")

    plot_kwargs = {k: kwargs[k] for k in ("ax",) if k in kwargs}

    plt.figure(figsize=figsize)

    palette = dict(zip(adata.obs[color_label].cat.categories, adata.uns[f"{color_label}_colors"]))
    sns.barplot(x=xlabel, y=ylabel, hue='Terminal state', data=df, palette=palette, dodge=True, **plot_kwargs)

    for i in range(len(df['TF'].unique()) - 1):
        plt.axvline(x=i + 0.5, color='gray', linestyle='--')

    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)  
    plt.yticks(fontsize=fontsize)  

    plt.legend(
        title='Terminal state',
        bbox_to_anchor=legend_bbox, 
        loc=legend_loc,
        borderaxespad=0
    )


    plt.tight_layout()
    plt.show()
