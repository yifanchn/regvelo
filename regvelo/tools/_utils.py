
from typing import List, Sequence, Dict

def split_elements(character_list : Sequence[str]) -> List[List[str]]:
    """
    Split underscore-separated TF combinations into lists of individual TFs.

    Parameters
    ----------
    character_list : Sequence[str]
        A list of TFs or TF combinations (e.g., ["TF1", "TF2_TF3"]).

    Returns
    -------
    List[List[str]]
        A list of lists, where each inner list contains individual TFs.
        For example, ["TF1", "TF2_TF3"] becomes [["TF1"], ["TF2", "TF3"]].
    """
    result_list = []
    for element in character_list:
        if '_' in element:
            parts = element.split('_')
            result_list.append(parts)
        else:
            result_list.append([element])
    return result_list

def combine_elements(split_list : Sequence[Sequence[str]]) -> List[str]:
    """
    Combine individual TF names into underscore-separated strings.

    Parameters
    ----------
    split_list : Sequence[Sequence[str]]
        A list of lists, where each inner list contains individual TFs.

    Returns
    -------
     List[str]
        A list of TF combinations as strings (e.g., ["TF1", "TF2_TF3"]).
    """
    result_list = []
    for parts in split_list:
        combined_element = "_".join(parts)
        result_list.append(combined_element)
    return result_list

def get_list_name(lst : Dict[str, object]) -> List[str]:
    """
    Extract keys from a dictionary.

    Parameters
    ----------
    lst : Dict[str, object]
        A dictionary from which to extract keys.

    Returns
    -------
    List[str]
        A list of keys from the dictionary.
    """
    names = []
    for name, obj in lst.items():
        names.append(name)
    return names



