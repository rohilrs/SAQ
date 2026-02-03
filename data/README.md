# SAQ Data Directory

This directory contains datasets for benchmarking and testing SAQ.

## Structure

```
data/
└── datasets/
    └── dbpedia_100k/       # Example dataset (downloaded)
        ├── vectors.fvecs   # Base vectors (99K × 1536)
        ├── queries.fvecs   # Query vectors (1K × 1536)
        ├── groundtruth.ivecs # Exact k-NN (1K × 100)
        └── metadata.txt    # Dataset info
```

## File Formats

### .fvecs (Float Vectors)
Standard ANN benchmark format. Each vector is stored as:
```
[dim: int32][v1: float32][v2: float32]...[vD: float32]
```

### .ivecs (Integer Vectors)
Same format but with int32 values (used for ground truth indices):
```
[k: int32][idx1: int32][idx2: int32]...[idxK: int32]
```

## Downloading Datasets

### DBpedia 100K (1536d OpenAI embeddings)
```bash
# Requires: pip install datasets numpy
python samples/download_dbpedia.py
```

### SIFT1M (128d SIFT descriptors)
```bash
# Download from http://corpus-texmex.irisa.fr/
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz -C data/datasets/
```

### GIST1M (960d GIST descriptors)
```bash
wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xzf gist.tar.gz -C data/datasets/
```

## Creating Custom Datasets

Use the Python utilities:

```python
import numpy as np
import struct

def write_fvecs(filename, vectors):
    """Write numpy array to .fvecs format."""
    n, d = vectors.shape
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('i', d))
            f.write(vectors[i].astype(np.float32).tobytes())

def read_fvecs(filename):
    """Read .fvecs file to numpy array."""
    with open(filename, 'rb') as f:
        d = struct.unpack('i', f.read(4))[0]
        f.seek(0, 2)  # End of file
        n = f.tell() // (4 + d * 4)
        f.seek(0)
        vectors = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            f.read(4)  # Skip dim
            vectors[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return vectors
```

## Notes

- Datasets are **not committed to git** (see `.gitignore`)
- Ground truth is computed as exact L2 k-NN
- For large datasets (1M+), consider using FAISS for ground truth computation
