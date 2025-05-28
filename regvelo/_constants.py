from typing import NamedTuple

"""
Constants used for registering layers in the AnnData object.
"""

class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    U_KEY: str = "U"


REGISTRY_KEYS = _REGISTRY_KEYS_NT()
