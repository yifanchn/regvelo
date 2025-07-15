import pandas as pd
from anndata import AnnData

def filter_genes(adata: AnnData) -> AnnData:
    r"""Filter genes in an AnnData object to ensure each gene has at least one upstream regulator in the GRN.

    The function iteratively refines the skeleton matrix to maintain only genes with regulatory connections. Only used
    by ``soft_constraint=False`` RegVelo model.

    Parameters
    ----------
    adata
        Annotated data object containing:
        
        - gene expression data,
        - a prior regulatory network in ``adata.uns["skeleton"]``,
        - a list of regulators in ``adata.uns["regulators"]``.

    Returns
    -------
    Updates ``adata`` with the following field:

    - ``adata.uns["skeleton"]``.
    """

    skeleton = adata.uns["skeleton"].loc[adata.var_names.tolist(), adata.var_names.tolist()]
    adata.uns["skeleton"] = skeleton
    
    # Remove genes with no upstream regulators
    while adata.uns["skeleton"].sum(0).min() == 0:
        # Update filtering based on skeleton
        skeleton = adata.uns["skeleton"]
        mask = skeleton.sum(0) > 0

        genes = adata.var_names[mask].tolist()
        print(f"Number of genes: {len(genes)}")

        # Filter skeleton and update `adata`
        skeleton = skeleton.loc[genes, genes]
        adata.uns["skeleton"] = skeleton

        # Update adata with filtered genes
        adata = adata[:, mask].copy()
        adata.uns["regulators"] = genes
        adata.uns["targets"] = genes

        # Re-index skeleton with updated gene names
        skeleton_df = pd.DataFrame(
            adata.uns["skeleton"],
            index=adata.uns["regulators"],
            columns=adata.uns["targets"],
        )
        adata.uns["skeleton"] = skeleton_df

    return adata