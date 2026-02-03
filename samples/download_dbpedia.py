#!/usr/bin/env python3
"""
Download DBpedia OpenAI embeddings dataset from HuggingFace.

Dataset: Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K
- 100K vectors
- 1536 dimensions (OpenAI text-embedding-3-large)

Outputs:
- data/datasets/dbpedia_100k/vectors.fvecs  (base vectors)
- data/datasets/dbpedia_100k/queries.fvecs  (query vectors)
- data/datasets/dbpedia_100k/groundtruth.ivecs (ground truth k-NN)
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path

def download_dataset():
    """Download dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset
    
    print("Downloading DBpedia embeddings from HuggingFace...")
    dataset = load_dataset("Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K")
    return dataset

def write_fvecs(filename: str, vectors: np.ndarray):
    """Write vectors in .fvecs format (dimension + float32 data per vector)."""
    n, d = vectors.shape
    vectors = vectors.astype(np.float32)
    
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('i', d))
            f.write(vectors[i].tobytes())
    
    print(f"  Wrote {n} vectors ({d}d) to {filename}")

def write_ivecs(filename: str, indices: np.ndarray):
    """Write indices in .ivecs format (k + int32 indices per query)."""
    n, k = indices.shape
    indices = indices.astype(np.int32)
    
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('i', k))
            f.write(indices[i].tobytes())
    
    print(f"  Wrote {n} x {k} ground truth to {filename}")

def compute_ground_truth(base_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 100):
    """Compute exact k-NN ground truth using brute force."""
    print(f"Computing ground truth (k={k})...")
    n_queries = query_vectors.shape[0]
    ground_truth = np.zeros((n_queries, k), dtype=np.int32)
    
    # Compute in batches to avoid memory issues
    batch_size = 100
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        queries_batch = query_vectors[start:end]
        
        # Compute L2 distances: ||q - x||^2 = ||q||^2 + ||x||^2 - 2*q.x
        q_norms = np.sum(queries_batch ** 2, axis=1, keepdims=True)
        x_norms = np.sum(base_vectors ** 2, axis=1)
        dists = q_norms + x_norms - 2 * np.dot(queries_batch, base_vectors.T)
        
        # Get top-k indices
        indices = np.argsort(dists, axis=1)[:, :k]
        ground_truth[start:end] = indices
        
        print(f"  Processed queries {start+1}-{end}/{n_queries}")
    
    return ground_truth

def main():
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / "datasets" / "dbpedia_100k"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    dataset = download_dataset()
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    train_data = dataset['train']
    
    # Get all embeddings
    embeddings = np.array(train_data['text-embedding-3-large-1536-embedding'], dtype=np.float32)
    n_total, dim = embeddings.shape
    print(f"  Total vectors: {n_total}, Dimension: {dim}")
    
    # Split into base and query sets
    n_queries = 1000  # Use 1000 queries
    n_base = n_total - n_queries
    
    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(n_total)
    
    base_indices = indices[:n_base]
    query_indices = indices[n_base:]
    
    base_vectors = embeddings[base_indices]
    query_vectors = embeddings[query_indices]
    
    print(f"  Base vectors: {n_base}")
    print(f"  Query vectors: {n_queries}")
    
    # Write vectors
    print("\nWriting vector files...")
    write_fvecs(str(data_dir / "vectors.fvecs"), base_vectors)
    write_fvecs(str(data_dir / "queries.fvecs"), query_vectors)
    
    # Compute and write ground truth
    ground_truth = compute_ground_truth(base_vectors, query_vectors, k=100)
    write_ivecs(str(data_dir / "groundtruth.ivecs"), ground_truth)
    
    # Write metadata
    meta_file = data_dir / "metadata.txt"
    with open(meta_file, 'w') as f:
        f.write(f"dataset: dbpedia-entities-openai3-text-embedding-3-large-1536-100K\n")
        f.write(f"source: https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K\n")
        f.write(f"n_base: {n_base}\n")
        f.write(f"n_queries: {n_queries}\n")
        f.write(f"dimension: {dim}\n")
        f.write(f"ground_truth_k: 100\n")
    
    print(f"\nDataset prepared in {data_dir}")
    print("Files:")
    print(f"  - vectors.fvecs ({n_base} x {dim})")
    print(f"  - queries.fvecs ({n_queries} x {dim})")
    print(f"  - groundtruth.ivecs ({n_queries} x 100)")
    print(f"  - metadata.txt")

if __name__ == "__main__":
    main()
