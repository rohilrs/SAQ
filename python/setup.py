"""Setup script for SAQ Python bindings.

Two workflows:

1. Package pre-built bindings (fast, for distribution):
       cd python && pip wheel . --no-deps -w dist/
   Requires _saq_core*.so (and optionally _saq_gpu*.so) already built
   in saq/ via cmake.

2. Build from source (full):
       SAQ_BUILD_CUDA=1 pip install .
   Runs cmake to compile the C++ library and pybind11 bindings.
"""

import glob
import os
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Distribution that always forces a binary package (platform wheel)."""
    def has_ext_modules(self):
        return True


# Check if pre-built .so files exist
saq_dir = os.path.join(os.path.dirname(__file__), "saq")
prebuilt = glob.glob(os.path.join(saq_dir, "_saq_core*.so"))

if not prebuilt and "bdist_wheel" not in sys.argv:
    # No pre-built bindings — try to build via cmake
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    build_cuda = os.environ.get("SAQ_BUILD_CUDA", "0") == "1"
    use_faiss = os.environ.get("SAQ_USE_FAISS", "0") == "1"

    cmake_args = [
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        "-DSAQ_BUILD_PYTHON=ON",
        "-DSAQ_BUILD_SAMPLES=OFF",
        "-DSAQ_BUILD_TESTS=OFF",
        "-DSAQ_USE_OPENMP=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_FLAGS=-mfma",
    ]
    if build_cuda:
        cmake_args.append("-DSAQ_BUILD_CUDA=ON")
    if use_faiss:
        cmake_args.append("-DSAQ_USE_FAISS=ON")

    cc = os.environ.get("CC")
    cxx = os.environ.get("CXX")
    if cc:
        cmake_args.append(f"-DCMAKE_C_COMPILER={cc}")
    if cxx:
        cmake_args.append(f"-DCMAKE_CXX_COMPILER={cxx}")

    build_dir = os.path.join(project_dir, "build_wheel")
    os.makedirs(build_dir, exist_ok=True)

    subprocess.check_call(["cmake", project_dir] + cmake_args, cwd=build_dir)

    targets = ["_saq_core"]
    if build_cuda:
        targets.append("_saq_gpu")

    for target in targets:
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release",
             "-j", str(os.cpu_count() or 4), "--target", target],
            cwd=build_dir,
        )

# Collect .so files as package data
so_patterns = ["_saq_core*.so", "_saq_gpu*.so", "libfmt.so*", "libglog.so*"]
package_data_files = []
for pat in so_patterns:
    package_data_files.extend(
        os.path.basename(f) for f in glob.glob(os.path.join(saq_dir, pat))
    )

setup(
    name="saq",
    version="0.2.0",
    description="SAQ: Scalar Additive Quantization for ANN search",
    packages=find_packages(),
    package_data={"saq": package_data_files},
    distclass=BinaryDistribution,
    python_requires=">=3.8",
    install_requires=["numpy>=1.20.0"],
)
