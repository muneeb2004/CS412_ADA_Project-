# Greedy Graph Spanner Benchmark (2k-1 Spanners)

## Project Description
This repository contains a Python implementation of the greedy weighted $(2k-1)$-spanner algorithm and an experimental benchmarking pipeline inspired by Chimani and Stutzenstein (ESA 2022). The project generates random connected weighted graphs, constructs spanners for selected $k$ values, evaluates structural and performance metrics, exports tabular results, and produces publication-ready plots.

The current implementation is designed for algorithm experimentation and reproducible academic reporting rather than production deployment.

## Installation Instructions

### Environment Requirements
- Python 3.10+ (tested in a modern Python 3.x setup)
- OS: Windows, Linux, or macOS

### Python Dependencies
Install required packages:

```bash
pip install networkx numpy matplotlib tsplib95
```

### Optional: Virtual Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install --upgrade pip
pip install networkx numpy matplotlib tsplib95
```

## Dataset Loaders

The repository includes three dataset loaders in [dataset_loaders.py](dataset_loaders.py):

- `load_tsplib_complete_graphs(tsplib_dir)`: loads every `.tsp` file with `tsplib95`, skips directed instances, builds a complete weighted NetworkX graph using `problem.get_weight(i, j)` for all node pairs, and skips instances if any edge weight is 0.
- `load_facebook_graph(path, seed=42)`: reads `facebook_combined.txt.gz` as an undirected edge list and assigns deterministic random integer weights in `[1, n]`.
- `load_gnutella_graph(path, seed=42)`: reads `p2p-Gnutella08.txt.gz`, converts to undirected, keeps the largest connected component, and assigns deterministic random integer weights in `[1, n]`.

Example:

```python
from dataset_loaders import (
	load_facebook_graph,
	load_gnutella_graph,
	load_tsplib_complete_graphs,
)

tsplib_graphs = load_tsplib_complete_graphs("tsplib")
facebook = load_facebook_graph("facebook_combined.txt.gz", seed=42)
gnutella = load_gnutella_graph("p2p-Gnutella08.txt.gz", seed=42)
```

## Usage Instructions

### Run the Full Benchmark
From the repository root:

```bash
python greedy_spanner.py
```

### What the Script Produces
After completion, the script writes outputs in the current working directory:
- `results.csv` (raw experiment rows)
- `fig1_sparseness.pdf`, `fig1_sparseness.png`
- `fig2_runtime.pdf`, `fig2_runtime.png`
- `fig3_stretch.pdf`, `fig3_stretch.png`

### Key Experiment Parameters (in code)
Inside [greedy_spanner.py](greedy_spanner.py):
- `SIZES`: graph sizes $n$
- `DENSITIES`: Erdos-Renyi edge probability values
- `REPS`: number of repetitions per setting
- `K_VALUES`: spanner parameters where stretch bound is $\alpha = 2k-1$

You can modify these constants to run larger or smaller experiments.

## Features / Functionality

### Implemented
- Greedy weighted $(2k-1)$-spanner construction.
- Connected random weighted graph generation (Erdos-Renyi with connectivity enforcement).
- Metric computation:
	- Sparseness: $|E_H|/n$
	- Lightness: $w(H)/w(MST(G))$
	- Effective stretch (exact for small graphs, sampled for larger graphs)
- Runtime and memory profiling during spanner construction:
	- Runtime via `time.perf_counter`
	- Peak memory via `tracemalloc`
- CSV export of raw results.
- Aggregation of repeated runs by `(n, density, k)`.
- Automated plotting for:
	- Sparseness vs graph size
	- Runtime vs graph size
	- Effective stretch vs graph size

### Present in Repository (Generated Artifacts)
- Result CSV and plot files are already included from prior runs.
- Academic report/presentation material is included as PDF/PPTX/TEX files.

### Pending / Not Yet Implemented
- Unit and integration tests.
- Command-line interface (CLI) arguments for experiment configuration.
- Configuration file support (e.g., YAML/JSON for experiment settings).
- Multi-file package/module layout (currently a single script implementation).
- Automated dependency lockfile and reproducible environment manifest.
- Continuous integration workflow for linting/testing.
- Explicit project license text.

## Project Structure

Top-level repository overview:

- [greedy_spanner.py](greedy_spanner.py): main implementation and benchmark pipeline.
- [README.md](README.md): repository documentation.
- `results.csv`: latest raw benchmark output.
- `fig1_sparseness.pdf` / `fig1_sparseness.png`: sparseness plots.
- `fig2_runtime.pdf` / `fig2_runtime.png`: runtime scaling plots.
- `fig3_stretch.pdf` / `fig3_stretch.png`: effective stretch plots.
- `ADA_Project_Interim_Report.tex` and related PDF/PPTX files: project report assets.
- `.git/`: version control metadata.

## Contributing Guidelines
Contributions are welcome, especially for:
- test coverage,
- CLI/configuration support,
- reproducibility tooling,
- additional datasets/graph models,
- algorithmic extensions and comparative baselines.

Suggested workflow:
1. Fork the repository.
2. Create a feature branch.
3. Keep changes focused and documented.
4. Run the script to verify outputs are generated correctly.
5. Open a pull request with a clear summary of changes and rationale.

For algorithm-related changes, include brief experimental evidence or references where possible.

## License
No license has been defined yet.

Placeholder: add a license file (for example, MIT, BSD-3-Clause, or Apache-2.0) before public redistribution.

## Acknowledgements / References

- Chimani, Markus and Stutzenstein, Philipp. Experimental context for practical spanner approximations (ESA 2022).
- NetworkX: graph algorithms and data structures.
- NumPy: numerical aggregation.
- Matplotlib: figure generation.

Repository-included reading/report assets:
- `Spanner_Approximations_In_Practice.pdf`
- `ADA_Project.pdf`
- `ADA_Project_Interim_Report.pdf`

