"""regvelo"""

import sys
import logging
from importlib import metadata

from rich.console import Console
from rich.logging import RichHandler

from . import datasets, metrics as mt, tools as tl, plotting as pl, preprocessing as pp
from ._constants import REGISTRY_KEYS
from ._model import REGVELOVI, VELOVAE
from .ModelComparison import ModelComparison


try:
    md = metadata.metadata(__name__)
    __version__ = md.get("version", "")
    __author__ = md.get("Author", "")
except ImportError:
    md = None

sys.modules.update({
    f"{__name__}.metrics": mt,
    f"{__name__}.tools": tl,
    f"{__name__}.plotting": pl,
    f"{__name__}.preprocessing": pp
})


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False

ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("regvelo: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False

__all__ = [
    "REGVELOVI",
    "VELOVAE",
    "REGISTRY_KEYS",
    "datasets",
    "metrics", "mt",
    "tools", "tl",
    "plotting", "pl",
    "preprocessing", "pp",
    "ModelComparison"
]

del metadata, md