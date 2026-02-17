# SAQ Codeflow Diagrams

## 1. High-Level File Architecture

This diagram shows the flow between major components from data input to search results.

```mermaid
flowchart TB
    subgraph Input["Input Data"]
        RAW["Raw Vectors<br/>(n x d float32)"]
    end

    subgraph Preprocessing["Preprocessing (Python/Faiss)"]
        PCA["pca.py<br/>---<br/>PCA full rotation via<br/>faiss.PCAMatrix.<br/>Outputs variances."]

        IVF_PY["ivf.py<br/>---<br/>K-means clustering via<br/>faiss.IndexIVFFlat.<br/>Outputs centroids + IDs."]

        GT["compute_gt.py<br/>---<br/>Brute-force exact kNN<br/>for ground truth."]
    end

    subgraph Training["Training Phase (C++)"]
        SAQDATA["quantization_plan.h<br/>---<br/>SaqDataMaker:<br/>variance → joint DP<br/>over 64-dim blocks →<br/>segment plan + rotators"]

        ROTATOR["rotator.h<br/>---<br/>Per-segment random<br/>orthogonal rotation<br/>(Householder QR)"]
    end

    subgraph Encoding["Encoding Phase (C++)"]
        QUANT["quantizer.h<br/>---<br/>QuantizerCluster:<br/>rotate → center →<br/>CAQEncoder per segment"]

        CAQENC["caq_encoder.h<br/>---<br/>CAQEncoder:<br/>scalar quantize →<br/>code adjustment →<br/>fac_rescale computation"]

        PACKER["cluster_packer.h<br/>---<br/>ClusterPacker:<br/>pack short codes (MSB)<br/>+ long codes + factors"]
    end

    subgraph Index["Index Construction"]
        IVFCPP["ivf_index.h<br/>---<br/>IVF: construct per-cluster<br/>SaqCluData, save/load."]

        CLUDATA["cluster_data.h<br/>---<br/>SaqCluData:<br/>multi-segment storage<br/>with packed layouts"]
    end

    subgraph Search["Search Phase (C++)"]
        INIT["initializer.h<br/>---<br/>FlatInitializer:<br/>find nprobe nearest<br/>centroids"]

        SEARCHER["saq_searcher.h<br/>---<br/>SAQSearcher (3-stage):<br/>1. Variance pruning<br/>2. 1-bit fastscan<br/>3. Accurate distance"]

        EST["caq_estimator.h<br/>---<br/>CaqCluEstimator:<br/>SIMD LUT-based distance<br/>+ saq_estimator.h"]

        FASTSCAN["fast_scan.h + lut.h<br/>---<br/>AVX-512/AVX2 shuffle<br/>for 32-vector blocks"]

        RESULT["Top-K Results<br/>(index, distance)"]
    end

    RAW --> PCA
    RAW --> IVF_PY
    RAW --> GT
    PCA --> SAQDATA
    SAQDATA --> ROTATOR

    PCA --> QUANT
    IVF_PY --> IVFCPP
    ROTATOR --> QUANT
    QUANT --> CAQENC
    CAQENC --> PACKER
    PACKER --> CLUDATA
    CLUDATA --> IVFCPP

    IVFCPP --> INIT
    INIT --> SEARCHER
    SEARCHER --> EST
    EST --> FASTSCAN
    SEARCHER --> RESULT

    style Input fill:#e1f5fe
    style Preprocessing fill:#fff9c4
    style Training fill:#fff3e0
    style Encoding fill:#f3e5f5
    style Index fill:#e8f5e9
    style Search fill:#e0f2f1
```

## 2. Detailed Function-Level Flow

### Index Construction: `IVF::construct()`

```mermaid
flowchart TB
    subgraph Construct["IVF::construct()"]
        direction TB
        C1["1. Load preprocessed data<br/>vectors_pca, centroids_pca,<br/>cluster_ids, variances"]

        C2["2. SaqDataMaker::make()<br/>---<br/>set_variance(variances)<br/>dynamic_programming(budget)<br/>→ quant_plan[(dim_len, bits)]<br/>gen_rotators() per segment"]

        C3["3. Per-cluster loop (OpenMP):<br/>SAQuantizer::quantize_cluster()<br/>---<br/>For each segment s:<br/>  rotate vectors<br/>  center on centroid<br/>  CAQEncoder::encode_and_fac()<br/>  ClusterPacker::store_and_pack()"]

        C4["4. ClusterPacker output:<br/>short_codes (MSB packed for fastscan)<br/>long_codes (remaining bits)<br/>factors (rescale, error)"]

        C5["5. Store SaqCluData per cluster<br/>+ global IDs → IVF index"]

        C1 --> C2
        C2 --> C3
        C3 --> C4
        C4 --> C5
    end

    style Construct fill:#e8f5e9
```

### Search: `IVF::search()`

```mermaid
flowchart TB
    subgraph SearchFlow["IVF::search()"]
        direction TB
        S1["1. FlatInitializer::centroids_distances()<br/>---<br/>L2 distance to all centroids<br/>Return top nprobe clusters"]

        S2["2. Per-cluster: SAQSearcher::searchCluster()<br/>---<br/>Prepare SaqCluEstimator<br/>(precompute LUTs per segment)"]

        S3["3. Block loop (32 vectors per block):"]

        S4["Stage 1: varsEstDist()<br/>---<br/>Variance-based lower bound<br/>Skip block if > current distk"]

        S5["Stage 2: compFastDist()<br/>---<br/>1-bit fastscan (MSB only)<br/>AVX-512 shuffle LUT lookup<br/>Skip if estimate > distk"]

        S6["Stage 3: compAccurateDist()<br/>---<br/>Full-bit codes, all segments<br/>Per-segment early termination<br/>Update top-K heap"]

        S7["4. Merge results across clusters<br/>Return top-K (index, distance)"]

        S1 --> S2
        S2 --> S3
        S3 --> S4
        S4 -->|"promising"| S5
        S4 -->|"skip"| S3
        S5 -->|"promising"| S6
        S5 -->|"skip"| S3
        S6 --> S3
        S3 -->|"done"| S7
    end

    style SearchFlow fill:#e0f2f1
```

### CAQEncoder: `encode_and_fac()`

```mermaid
flowchart TB
    subgraph Encode["CAQEncoder::encode_and_fac()"]
        direction TB
        E1["1. downUpSample(o):<br/>v_max = max|o_i|<br/>delta = 2*v_max / 2^B<br/>code[i] = floor((o_i + v_max) / delta)<br/>o_bar[i] = delta*(code[i]+0.5) - v_max"]

        E2["2. code_adjustment():<br/>For each dim, try code[i] ±1<br/>Accept if cosine(o, o_bar) improves<br/>Repeat for num_iter rounds"]

        E3["3. Compute factors:<br/>fac_rescale = |o|² / <o, o_a><br/>fac_err = |o - fac_rescale * o_a|²"]

        E4["4. rescale_vmx_to1():<br/>Normalize v_max to 1.0<br/>Absorb into fac_rescale"]

        E5["5. Output: CaqCode<br/>code[], v_max, v_min, delta,<br/>fac_rescale, fac_err,<br/>ip_o_oa, oa_l2sqr"]

        E1 --> E2
        E2 --> E3
        E3 --> E4
        E4 --> E5
    end

    style Encode fill:#f3e5f5
```

## Legend

| Phase | Description |
|-------|-------------|
| **Preprocessing** | PCA rotation, K-means clustering, ground truth (Python/Faiss) |
| **Training** | Joint DP segmentation + bit allocation, per-segment rotation matrices |
| **Encoding** | Per-cluster: rotate -> CAQ encode -> pack short/long codes |
| **Indexing** | Build IVF with SaqCluData per cluster |
| **Search** | 3-stage block-level search: variance pruning -> 1-bit fastscan -> accurate |

## Key Data Structures

```
SaqData (quantization plan)
├── QuantizeConfig          # bpd, dist_type, use_caq, etc.
├── quant_plan[]            # [(dim_len, bits)] per segment
├── BaseQuantizerData[]     # Per-segment: rotator, centroid, metadata
└── data_variance           # Per-dimension variance from PCA

SaqCluData (per-cluster storage)
├── CAQClusterData[]        # Per-segment:
│   ├── short_codes         # MSB-packed for fastscan (32-vector blocks)
│   ├── long_codes          # Remaining bits
│   ├── factors             # fac_rescale, fac_err per vector
│   └── metadata            # v_max, v_min, delta per vector
├── global_ids[]            # Original vector indices
└── num_vectors

IVF (inverted file index)
├── FlatInitializer         # Centroid storage for nprobe selection
├── SaqData                 # Shared quantization plan
├── SaqCluData[]            # Per-cluster encoded data
└── SearcherConfig          # nprobe, num_rerank, etc.
```

## Source Files

| File | Key Classes | Purpose |
|------|-------------|---------|
| `quantization_plan.h/.cpp` | SaqData, SaqDataMaker | DP segmentation, bit allocation, serialization |
| `caq_encoder.h/.cpp` | CAQEncoder, CaqCode | Scalar quantize + code adjustment + factors |
| `quantizer.h/.cpp` | QuantizerCluster, QuantizerSingle | Per-segment encode orchestration |
| `saq_quantizer.h/.cpp` | SAQuantizer, SAQuantizerSingle | Multi-segment cluster orchestration |
| `caq_estimator.h/.cpp` | CaqCluEstimator | 3-stage SIMD distance (variance, fastscan, accurate) |
| `saq_estimator.h/.cpp` | SaqCluEstimator | Multi-segment distance aggregation |
| `saq_searcher.h/.cpp` | SAQSearcher | Block-level 3-stage search with early termination |
| `cluster_data.h/.cpp` | CAQClusterData, SaqCluData | Per-segment and multi-segment cluster storage |
| `cluster_packer.h/.cpp` | ClusterPacker | Pack codes + factors into cluster format |
| `fast_scan.h/.cpp` | — | AVX-512/AVX2 LUT-based fastscan primitives |
| `lut.h/.cpp` | Lut | High-accuracy fast IP via lookup tables |
| `ivf_index.h/.cpp` | IVF | Index construct, search, save/load |
| `rotator.h/.cpp` | Rotator, PCARotator | Random orthogonal rotation (Householder QR) |
