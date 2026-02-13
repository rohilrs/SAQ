"""Setup script for SAQ Python bindings.

Build and install:
    pip install .

Development install (editable):
    pip install -e .
"""

import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}/saq",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DSAQ_BUILD_PYTHON=ON",
            "-DSAQ_BUILD_SAMPLES=OFF",
            "-DSAQ_BUILD_TESTS=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        build_dir = os.path.join(project_dir, "build_python")
        os.makedirs(build_dir, exist_ok=True)

        subprocess.check_call(
            ["cmake", project_dir] + cmake_args, cwd=build_dir
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release", "--target", "_saq_core"],
            cwd=build_dir,
        )


setup(
    name="saq",
    version="0.1.0",
    description="SAQ: Scalar Additive Quantization for ANN search",
    packages=find_packages(),
    ext_modules=[CMakeExtension("saq._saq_core")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    install_requires=["numpy>=1.20.0"],
)
