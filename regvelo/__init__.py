"""regvelo."""

import logging

from rich.console import Console
from rich.logging import RichHandler

from regvelo import datasets
from regvelo import tools as tl
from regvelo import plotting as pl
from regvelo import preprocessing as pp

from ._constants import REGISTRY_KEYS
from ._model import REGVELOVI, VELOVAE
from .ModelComparison import ModelComparison

import sys  # isort:skip

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tl", "pl", "pp"]})

# https://github.com/python-poetry/poetry/pull/2366#issuecomment-652418094
# https://github.com/python-poetry/poetry/issues/144#issuecomment-623927302
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "regvelo"
__version__ = importlib_metadata.version(package_name)

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("regvelo: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False

__all__ = [
    "REGVELOVI",
    "VELOVAE",
    "REGISTRY_KEYS",
    "datasets",
    "ModelComparison"
]
