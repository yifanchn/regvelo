"""regvelo"""

import logging

from rich.console import Console
from rich.logging import RichHandler

from . import datasets
from . import metrics as mt
from . import tools as tl
from . import plotting as pl
from . import preprocessing as pp

from ._constants import REGISTRY_KEYS
from ._model import REGVELOVI, VELOVAE
from .ModelComparison import ModelComparison

import sys  # isort:skip
sys.modules.update({
    f"{__name__}.metrics": mt,
    f"{__name__}.tools": tl,
    f"{__name__}.plotting": pl,
    f"{__name__}.preprocessing": pp
})


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
    "ModelComparison",
    "metrics", "mt",
    "tools", "tl",
    "plotting", "pl",
    "preprocessing", "pp"
]
