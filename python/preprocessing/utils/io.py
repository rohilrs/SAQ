"""
I/O utilities for fvecs/ivecs/fbin/ibin file formats.

Ported from reference SAQ repository (python/utils/io.py).
"""

import numpy as np
import struct


def read_ivecs(filename):
    """Read vectors from .ivecs format (int32)."""
    print(f"Reading File - {filename}")
    a = np.fromfile(filename, dtype="int32")
    d = a[0]
    print(f"\t{filename} read")
    return a.reshape(-1, d + 1)[:, 1:]


def read_fvecs(filename):
    """Read vectors from .fvecs format (float32)."""
    return read_ivecs(filename).view("float32")


def write_ivecs(filename, m):
    """Write vectors to .ivecs format (int32)."""
    print(f"Writing File - {filename}")
    n, d = m.shape
    myimt = "i" * d
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(struct.pack("i", d))
            bin = struct.pack(myimt, *m[i])
            f.write(bin)
    print(f"\t{filename} wrote")


def write_fvecs(filename, m):
    """Write vectors to .fvecs format (float32)."""
    m = m.astype("float32")
    write_ivecs(filename, m.view("int32"))


def read_ibin(filename):
    """Read vectors from .ibin format (int32 with n,d header)."""
    n, d = np.fromfile(filename, count=2, dtype="int32")
    a = np.fromfile(filename, dtype="int32")
    print(f"\t{filename} read")
    return a[2:].reshape(n, d)


def read_fbin(filename):
    """Read vectors from .fbin format (float32 with n,d header)."""
    return read_ibin(filename).view("float32")


def read_somefiles(filename):
    """Read vectors from a file, auto-detecting format by extension."""
    if filename.endswith(".fvecs"):
        return read_fvecs(filename)
    elif filename.endswith(".ivecs"):
        return read_ivecs(filename)
    elif filename.endswith(".fbin"):
        return read_fbin(filename)
    elif filename.endswith(".bin"):
        return read_ibin(filename)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
