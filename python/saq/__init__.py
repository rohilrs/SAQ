"""SAQ: Scalar Additive Quantization for approximate nearest neighbor search."""

import os as _os
import sys as _sys

# On Windows, add the package directory to DLL search path so that
# glogd.dll / fmtd.dll (copied during build) can be found.
if _sys.platform == "win32" and hasattr(_os, "add_dll_directory"):
    _os.add_dll_directory(_os.path.dirname(_os.path.abspath(__file__)))

from ._saq_core import (
    BaseQuantType,
    DistType,
    IVF,
    QuantizeConfig,
    QuantSingleConfig,
    SearcherConfig,
    load_fvecs,
    load_ivecs,
)

__all__ = [
    "BaseQuantType",
    "DistType",
    "IVF",
    "QuantizeConfig",
    "QuantSingleConfig",
    "SearcherConfig",
    "load_fvecs",
    "load_ivecs",
]
