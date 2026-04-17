from .correlation import *
from .generate import *

try:
    from ._version import __version__
except ImportError:
    # Fallback for development installations without setuptools_scm
    __version__ = "0.0.0.dev0"
