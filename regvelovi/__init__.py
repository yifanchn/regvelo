"""regvelovi."""

import logging

from rich.console import Console
from rich.logging import RichHandler

from regvelovi import datasets
from ._constants import REGISTRY_KEYS
from ._model import REGVELOVI, VELOVAE
from ._utils import get_permutation_scores, preprocess_data, prior_GRN, sanity_check
from ._perturbation import in_silico_block_simulation,TFScanning_func,TFscreening,abundance_test

# https://github.com/python-poetry/poetry/pull/2366#issuecomment-652418094
# https://github.com/python-poetry/poetry/issues/144#issuecomment-623927302
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "regvelovi"
__version__ = importlib_metadata.version(package_name)

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("regvelovi: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False

__all__ = [
    "REGVELOVI",
    "VELOVAE",
    "REGISTRY_KEYS",
    "datasets",
    "get_permutation_scores",
    "preprocess_data",
    "prior_GRN",
    "sanity_check",
    "in_silico_block_simulation",
    "TFScanning_func",
    "TFscreening",
    "abundance_test"
]
