import numpy as np
import pandas as pd
from scipy.stats import ranksums, ttest_ind
from sklearn.metrics import roc_auc_score
from typing import Literal

from ._utils import p_adjust_bh


def abundance_test(
    prob_raw : pd.DataFrame, 
    prob_pert : pd.DataFrame, 
    method : Literal["likelihood", "t-statistics"] = "likelihood"
    ) -> pd.DataFrame:
    r"""Perform an abundance test comparing cell fate probabilities between 
    raw and perturbed datasets.

    Parameters
    ----------
    prob_raw
        DataFrame containing fate probabilities from the original (unperturbed) data.
    prob_pert
        DataFrame containing fate probabilities from the perturbed data.
    method
        Scoring method to use: 

        - "t-statistics" (uses t-statistics),
        - "likelihood" (uses ROC AUC).

    Returns
    -------
    DataFrame containing: 

    - "coefficient" (test statistic or ROC AUC score), 
    - "p-value" (unadjusted p-value),
    - "FDR adjusted p-value" (Benjamini-Hochberg corrected p-value).
    """
    y = [1] * prob_raw.shape[0] + [0] * prob_pert.shape[0]
    X = pd.concat([prob_raw, prob_pert], axis=0)

    table = []
    for i in range(prob_raw.shape[1]):
        pred = np.array(X.iloc[:, i])
        if np.sum(pred) == 0:
            score, pval = np.nan, np.nan
        else:
            pval = ranksums(pred[np.array(y) == 0], pred[np.array(y) == 1], alternative = "less")[1]
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