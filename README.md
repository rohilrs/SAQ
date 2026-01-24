# SAQ Library (C/C++)

Minimal C/C++ library scaffold for SAQ-based vector quantization research.

## Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

## Directory Layout

- `include/saq/` public headers (to be added)
- `src/` core implementation files (listed in `src/README.md`)
- `sample/` minimal demos
- `python/` future bindings
- `third_party/` vendored dependencies (if any)

## Citation

If you use this codebase, please cite the SAQ paper:

```bibtex
@article{saq2025,
  title  = {SAQ},
  eprint = {2509.12086},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  year   = {2025},
  url    = {https://arxiv.org/abs/2509.12086}
}
```
