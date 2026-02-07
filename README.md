# BMC-AGI: Biomemetic Complex for Artificial General Intelligence

Prototype simulation code for the paper:

> **A Tension Is All You Need: How Competing Drives Create Agency in AI**
>
> Aleksey Zhuravlev
>
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

## Overview

The Biomemetic Complex (BMC) framework proposes that goal-directed behavior emerges from **dynamic tension between two interacting systems**: a fixed utility layer (analogous to genetic drives) and a dynamic memetic layer (acquired cultural information). This repository contains the prototype simulation that validates the framework.

## Repository Contents

```
prototype/
  memplex_visualization.ipynb   # Full simulation: 46-node BMC graph, 9 scenarios
figures/
  figure1_architecture.pdf      # Architecture diagram (Figure 1 in paper)
```

## Prototype

The simulation implements a single BMC agent with:
- **46 nodes**: 7 utility drives (Panksepp affective systems) + 39 memes across 5 semantic clusters
- **Signed edges**: positive (reinforcement) and negative (inhibition) connections
- **9 scenarios**: normal cognition, acute stress, hub attack, fatigue, immune response (accept/reject), hub displacement, edge decay, sleep consolidation + BLEND, sleeper effect

Key mechanisms demonstrated:
- Four qualitative regimes (M-dominance, G-dominance, Balance, Conflict)
- Signed-edge structural balance analysis
- Memetic immune system (accept/reject based on compatibility)
- Panksepp affective binding (SEEKING, RAGE, FEAR, PANIC, CARE, PLAY, LUST)
- Utility inertia for observable defensive dynamics
- Fatigue-induced cognitive regression
- Hub fragility with asymmetric edge decay
- Sleep consolidation with memory BLEND
- Sleeper effect (delayed meme acceptance)
- Monte Carlo validation (50 seeds × 8 scenarios)

## Requirements

```
python >= 3.9
networkx
numpy
matplotlib
```

## Usage

Open `prototype/memplex_visualization.ipynb` in Jupyter and run all cells.

## Citation

```bibtex
@article{zhuravlev2025tension,
  title={A Tension Is All You Need: How Competing Drives Create Agency in AI},
  author={Zhuravlev, Aleksey},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT

## Contact

Aleksey Zhuravlev — a.o.zhuravlev@gmail.com
