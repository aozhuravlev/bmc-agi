# BMC-AGI: Biomemetic Complex Prototype

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18733165.svg)](https://doi.org/10.5281/zenodo.18733165)

Prototype simulation code for the BMC (Biomemetic Complex) framework.

> **A Tension Is All You Need: The Biomemetic Complex as a Unified Computational Theory of Consciousness**
>
> Aleksey Zhuravlev (2026)
>
> Paper: [doi.org/10.5281/zenodo.18733165](https://doi.org/10.5281/zenodo.18733165) | Website: [bmc-theory.org](https://bmc-theory.org) | Demo: [bmc-theory.org/demo](https://bmc-theory.org/demo/)

## Overview

The Biomemetic Complex (BMC) framework models consciousness as emergent from **dynamic tension between two replicator systems**: a fixed utility layer G (analogous to genetic drives) and a dynamic memetic layer M (acquired knowledge). The framework produces a computable Consciousness Level (CL) metric, subsumes five competing theories (IIT, GNW, HOT, AST, PP) as special cases, and retrodicts 9/9 results from the COGITATE adversarial study (Nature 2025).

## Repository Contents

```
prototype/
  bmc/                            # BMC simulator engine
    simulator.py                  # Core BMC dynamics (~1300 lines)
    config.py                     # All parameters (~115 params)
    scaling_analysis.py           # Block 6: Scaling analysis (N=100..10000)
    graph.py, analysis.py, ...    # Supporting modules
  bmc_nodes_500.py                # 510-meme graph definition (21 clusters, 1205 edges)
  memplex_visualization.ipynb     # Interactive notebook: scenarios + Monte Carlo
  results/
    scaling_analysis.json          # Raw scaling data (38+126 runs)
    scaling_summary.png            # 2×2 summary panel
    scaling_*.png                  # Individual plots
figures/
  figure1_architecture.pdf         # Architecture diagram
```

## Prototype

The simulation implements a single BMC agent with:
- **510 meme nodes** across 21 semantic clusters + **8 utility drives** (Panksepp affective systems)
- **1205 signed edges**: positive (reinforcement) and negative (inhibition)
- **26 validation scenarios**: normal cognition, stress, hub attack, fatigue, immune response, hub displacement, sleep + BLEND, sleeper effect, κ consolidation (sensory/STM/LTM), and more
- **Monte Carlo validation**: 50+ seeds per scenario, ablation analysis (13/13 mechanisms necessary)

Key mechanisms:
- Four qualitative regimes (M-dominance, G-dominance, Balance, Conflict)
- CL metric: CL(t) = σ_norm · A_SMC · f(Balance) — computable in O(N log N)
- κ consolidation levels (sensory → STM → LTM) with differential decay
- SIT (Structural Incompleteness Tension) — curiosity/drive formalization
- Memetic immune system (I-layer: accept/reject based on compatibility)
- SMC (Self-Model Cluster) with recursion depth tracking

## Scaling Analysis (Block 6)

Demonstrates CL scale-invariance across two orders of magnitude:

| N | σ_SW | CL (steady-state) | Time/step |
|---|------|-------------------|-----------|
| 100 | 2.47±0.59 | 0.039±0.025 | 0.02s |
| 500 | 4.77±0.55 | 0.043±0.011 | 0.19s |
| 1000 | 5.99±0.82 | 0.033±0.007 | 0.58s |
| 5000 | 8.80±0.92 | 0.042±0.005 | 3.30s |
| 10000 | 10.69±1.57 | 0.055±0.002 | 6.33s |

CL remains in the ~0.03–0.05 corridor. Computation scales as O(N log N), compared to IIT's Φ which is O(2^N) and infeasible beyond N ≈ 20.

Phase transition: CL = 0 when SMC fraction = 0 (no self-model → no consciousness), with sharp onset at any SMC > 0.

Run: `cd prototype && python -m bmc.scaling_analysis`

## Requirements

```
python >= 3.9
networkx
numpy
matplotlib
tqdm
```

## Usage

```bash
# Interactive notebook
jupyter notebook prototype/memplex_visualization.ipynb

# Scaling analysis (takes ~30 min on 8 cores)
cd prototype && python -m bmc.scaling_analysis
```

## Citation

```bibtex
@misc{zhuravlev2026tension,
  author = {Zhuravlev, Aleksey},
  title = {A Tension Is All You Need: The Biomemetic Complex as a Unified Computational Theory of Consciousness},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18733165},
  url = {https://doi.org/10.5281/zenodo.18733165}
}
```

## License

MIT

## Contact

Aleksey Zhuravlev — a.o.zhuravlev@gmail.com
