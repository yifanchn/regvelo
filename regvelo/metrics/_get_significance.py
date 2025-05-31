

def get_significance(pvalue: float) -> str:
    """Return significance annotation for a p-value.

    Parameters
    ----------
    pvalue
        P-value to interpret.

    Returns
    -------
    str
        A string indicating the level of significance:
        "***" for p < 0.001,
        "**" for p < 0.01,
        "*" for p < 0.1,
        "n.s." (not significant) otherwise.
    """
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.1:
        return "*"
    else:
        return "n.s."