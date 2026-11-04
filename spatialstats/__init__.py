from .correlation import *
from .generate import *
from .plotting import *
from .density import *
from .fitting import *

try:
    from ._version import __version__
except ImportError:
    # Fallback for development installations without setuptools_scm
    __version__ = "0.0.0.dev0"
