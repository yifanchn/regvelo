import torch
import pandas as pd
import numpy as np

import cellrank as cr
from anndata import AnnData
from scvelo import logging as logg
import os,shutil
from typing import Dict, Optional, Sequence, Tuple, Union


def split_elements(character_list : Sequence[str]) -> Sequence[Sequence[str]]:
    """Split elements."""
    result_list = []
    for element in character_list:
        if '_' in element:
            parts = element.split('_')
            result_list.append(parts)
        else:
            result_list.append([element])
    return result_list

def combine_elements(split_list : Sequence[Sequence[str]]) -> Sequence[str]:
    """Combine elements."""
    result_list = []
    for parts in split_list:
        combined_element = "_".join(parts)
        result_list.append(combined_element)
    return result_list

def get_list_name(lst : Dict[str, Any]) -> Sequence[str]:
    """Get the names of the elements in a dictionary."""
    names = []
    for name, obj in lst.items():
        names.append(name)
    return names



