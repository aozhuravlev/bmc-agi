"""BMC — Biomemetic Complex simulation engine."""
from .simulator import BMCSimulator
from .config import UTILITY_NODES

# graph.py imports are lazy — it requires bmc_nodes_500 module
# which may not be available in all environments (e.g. Docker demo).
# Callers that need build_bmc_graph should import from bmc.graph directly.
try:
    from .graph import (build_bmc_graph, ALL_MEMES, MEME_TO_CLUSTER, OPEN_MEMES_LIST,
                        SMC_MEMES, SMC_LEVEL_1, SMC_LEVEL_2)
except ImportError:
    pass
