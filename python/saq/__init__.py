"""SAQ: Scalar Additive Quantization for approximate nearest neighbor search."""

import os as _os
import sys as _sys

# On Windows, add DLL search paths so dependent DLLs can be found.
if _sys.platform == "win32" and hasattr(_os, "add_dll_directory"):
    _os.add_dll_directory(_os.path.dirname(_os.path.abspath(__file__)))
    # Add CUDA DLL directories for cublas/cudart if available.
    _cuda_path = _os.environ.get("CUDA_PATH", "")
    if _cuda_path:
        for _subdir in ("bin", _os.path.join("bin", "x64")):
            _cuda_dir = _os.path.join(_cuda_path, _subdir)
            if _os.path.isdir(_cuda_dir):
                _os.add_dll_directory(_cuda_dir)

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

# GPU support (optional — requires CUDA build)
try:
    from ._saq_gpu import GpuIVF
    __all__.append("GpuIVF")
except ImportError:
    pass
