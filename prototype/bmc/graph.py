"""BMC graph construction: utility + memetic layers."""
import numpy as np
import networkx as nx

from bmc_nodes_500 import (
    MEME_CLUSTERS, HUB_MEMES, NEGATIVE_EDGES, CROSS_LINKS,
    UTILITY_CONNECTIONS, INCOMPAT_SPEC, OPEN_MEMES,
    SMC_MEMES, SMC_LEVEL_1, SMC_LEVEL_2,
)
from .config import UTILITY_NODES

# ── Flat meme list and reverse mapping ──
ALL_MEMES = []
MEME_TO_CLUSTER = {}
for cluster, memes in MEME_CLUSTERS.items():
    for m in memes:
        ALL_MEMES.append(m)
        MEME_TO_CLUSTER[m] = cluster

# ── Open memes (added once at module level, not per build call) ──
OPEN_MEMES_LIST = [name for name, _, _, _ in OPEN_MEMES]
for name, cluster, _, _ in OPEN_MEMES:
    ALL_MEMES.append(name)
    MEME_TO_CLUSTER[name] = cluster


def build_bmc_graph(seed=42):
    """Build the full BMC graph: utility + memetic layers with semantic connections."""
    rng = np.random.RandomState(seed)
    G = nx.Graph()

    # ── Add meme nodes (with fidelity and age) ──
    for m in ALL_MEMES:
        G.add_node(m, layer='memetic', cluster=MEME_TO_CLUSTER[m],
                   activation=rng.uniform(0.05, 0.25),
                   fidelity=rng.uniform(0.3, 0.8),
                   age=rng.uniform(0, 100))

    # ── Intra-cluster edges (BA-like preferential attachment) ──
    for cluster, memes in MEME_CLUSTERS.items():
        n = len(memes)
        if n < 3:
            for i in range(n):
                for j in range(i+1, n):
                    G.add_edge(memes[i], memes[j],
                               weight=rng.uniform(0.4, 0.9), etype='intra')
            continue
        # Start with a triangle
        for i in range(3):
            for j in range(i+1, 3):
                G.add_edge(memes[i], memes[j],
                           weight=rng.uniform(0.5, 0.9), etype='intra')
        # Attach remaining nodes preferentially
        for k in range(3, n):
            existing = memes[:k]
            degrees = np.array([G.degree(e) + 1 for e in existing], dtype=float)
            probs = degrees / degrees.sum()
            n_attach = min(2, len(existing))
            targets = rng.choice(existing, size=n_attach, replace=False, p=probs)
            for t in targets:
                G.add_edge(memes[k], t,
                           weight=rng.uniform(0.3, 0.8), etype='intra')

    # ── Cross-cluster edges (sparse, positive) — from bmc_nodes_500 ──
    for m1, m2, w in CROSS_LINKS:
        if m1 in G.nodes() and m2 in G.nodes():
            G.add_edge(m1, m2, weight=w, etype='cross')

    # ── Negative edges — from bmc_nodes_500 ──
    for m1, m2, w in NEGATIVE_EDGES:
        if m1 in G.nodes() and m2 in G.nodes():
            if G.has_edge(m1, m2):
                G.edges[m1, m2]['weight'] = w
            else:
                G.add_edge(m1, m2, weight=w, etype='cross_neg')

    # ── Add utility nodes ──
    for u, props in UTILITY_NODES.items():
        G.add_node(u, layer='utility', base_activation=props['base'],
                   activation=props['base'])

    # ── Semantic utility → meme connections — from bmc_nodes_500 ──
    for u, connections in UTILITY_CONNECTIONS.items():
        for meme, weight, etype in connections:
            if meme in G.nodes():
                G.add_edge(u, meme, weight=weight, etype=etype)

    # ── Open memes (SIT: structural gaps / unsolved questions) ──
    for name, cluster, connections, closure in OPEN_MEMES:
        G.add_node(name, layer='memetic', cluster=cluster,
                   activation=rng.uniform(0.05, 0.20),
                   fidelity=rng.uniform(0.2, 0.5),
                   age=rng.uniform(0, 50),
                   is_open=True, closure=closure)
        for target, weight in connections:
            if target in G.nodes():
                G.add_edge(name, target, weight=weight, etype='cross')

    # ── Build incompatibility matrix — from bmc_nodes_500 ──
    incompatibility = {}
    for u in UTILITY_NODES:
        incompatibility[u] = {}
        for m in ALL_MEMES:
            incompatibility[u][m] = 0.0

    for u, memes_dict in INCOMPAT_SPEC.items():
        for m, val in memes_dict.items():
            if m in incompatibility[u]:
                incompatibility[u][m] = val

    return G, incompatibility
