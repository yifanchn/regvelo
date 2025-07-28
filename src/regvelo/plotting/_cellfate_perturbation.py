
import pandas as pd
from anndata import AnnData
from typing import Any

import seaborn as sns
import matplotlib.pyplot as plt


def cellfate_perturbation(
        adata: AnnData,
        df: pd.DataFrame,
        color_label: str = "celltype_cluster",
        **kwargs: Any,
        ) -> None:
    r"""Plot depletion likelihood or score for transcription factors across terminal states.

    Parameters
    ----------
    adata
        Annotated data matrix of the unperturbed system, used to extract color palette info.
    df
        DataFrame containing perturbation results. Must include columns:
        
        - ``"TF"``,
        - ``"Terminal state"``,
        - and either ``"Depletion likelihood"`` or ``"Depletion score"``.
    color_label
        Key in `adata.obs` used to extract categorical color palette from `adata.uns`.
    **kwargs 
        Additional optional parameters:

        - ``fontsize`` : Font size for axis labels and title (default: ``14``).
        - ``figsize`` : Size of the plot (default: ``(12, 6)``).
        - ``legend_loc`` : Location of the legend (default: ``"center left"``).
        - ``legend_bbox`` : Positioning of the legend bounding box (default: ``(1.02, 0.5)``).
        - ``xlabel`` : Label for the x-axis (default: ``"TF"``).
        - ``ylabel`` : Label for the y-axis (default is determined by the score type).

    Returns
    -------
    Nothing, just plots the figure. 
    """

    fontsize = kwargs.get("fontsize", 14)
    figsize = kwargs.get("figsize", (12, 6))
    legend_loc = kwargs.get("legend_loc", "center left")
    legend_bbox = kwargs.get("legend_bbox", (1.02, 0.5))

    if "Depletion likelihood" in df.columns:
        value_col = "Depletion likelihood"
        reference_line = 0.5
        ylabel = kwargs.get("ylabel", "Depletion likelihood")
        style_by_pval = True
    elif "Depletion score" in df.columns:
        value_col = "Depletion score"
        reference_line = 0
        ylabel = kwargs.get("ylabel", "Depletion score")
        style_by_pval = False
    else:
        raise ValueError("df must contain either 'Depletion likelihood' or 'Depletion score'.")

    xlabel = kwargs.get("xlabel", "TF")

    tf_order = df["TF"].unique().tolist()
    state_order = df["Terminal state"].unique().tolist()

    palette = dict(zip(adata.obs[color_label].cat.categories, adata.uns[f"{color_label}_colors"]))
    pv_table = df.pivot(index='TF', columns='Terminal state', values='p-value')

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    bars = sns.barplot(
        data=df,
        x="TF",
        y=value_col,
        hue="Terminal state",
        palette=palette,
        dodge=True,
        order=tf_order,
        hue_order=state_order,
        ax=ax
    )

    if style_by_pval:
        # Process bars based on p-value
        for i, container in enumerate(ax.containers):
            for j, bar in enumerate(container):
                tf = tf_order[j]
                state = state_order[i]
                try:
                    pval = pv_table.loc[tf, state]
                except KeyError:
                    continue
                    
                if pval >= 0.05:
                    # Style non-significant bars
                    bar.set_facecolor("#F0F0F0")
                    bar.set_edgecolor('k')
                    bar.set_linestyle((0, (3, 3)))  # Dashed pattern
                    bar.set_linewidth(1.5)
                else:
                    # Ensure significant bars have solid borders
                    bar.set_edgecolor(bar.get_facecolor())
                    bar.set_linewidth(0.5)

    # Add vertical separators between TF groups
    for i in range(len(tf_order) - 1):
        ax.axvline(x=i + 0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Add reference line
    ax.axhline(y=reference_line, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    # Label and font settings
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.grid(False)
    
    plt.legend(
        title='Terminal state',
        bbox_to_anchor=legend_bbox,
        loc=legend_loc,
        borderaxespad=0
    )
    plt.title("RegVelo perturbation predictions", fontsize=fontsize+2, pad=10)
    plt.tight_layout()
    plt.show()