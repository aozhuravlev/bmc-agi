"""Block 6: Scaling Analysis — CL scale-invariance and phase transitions.

Demonstrates that CL metric and phase transitions are scale-invariant
from N=100 to N=50000 nodes. Produces JSON results + PNG plots.
"""
import json
import os
import time
import sys
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import UTILITY_NODES
from .simulator import BMCSimulator


# ─────────────────────────────────────────────────────────────────────
# 1. Synthetic Graph Generator
# ─────────────────────────────────────────────────────────────────────

def generate_synthetic_bmc(N, seed=42, smc_fraction=0.06):
    """Generate a synthetic BMC graph with N meme nodes.

    Returns (G, incompatibility, smc_set, smc_levels) compatible
    with BMCSimulator.__init__.
    """
    rng = np.random.RandomState(seed)

    # BA graph → natural small-world, mean degree ≈ 6
    ba = nx.barabasi_albert_graph(N, m=3, seed=seed)

    # Cluster assignment via Louvain on BA topology
    communities = nx.community.louvain_communities(ba, seed=seed)
    node_cluster = {}
    for idx, comm in enumerate(communities):
        cname = f"cluster_{idx}"
        for n in comm:
            node_cluster[n] = cname

    # Build BMC graph
    G = nx.Graph()

    # Add meme nodes
    meme_names = [f"M{i}" for i in range(N)]
    for i, name in enumerate(meme_names):
        G.add_node(name, layer='memetic',
                   cluster=node_cluster.get(i, 'cluster_0'),
                   activation=rng.uniform(0.05, 0.25),
                   fidelity=rng.uniform(0.3, 0.7),
                   age=rng.randint(0, 50))

    # Add meme-meme edges from BA topology
    for u, v in ba.edges():
        mu, mv = meme_names[u], meme_names[v]
        # ~5% of cross-cluster edges are negative
        cu = node_cluster.get(u, '')
        cv = node_cluster.get(v, '')
        if cu != cv and rng.random() < 0.05:
            w = -rng.uniform(0.1, 0.5)
        else:
            w = rng.uniform(0.2, 0.8)
        G.add_edge(mu, mv, weight=w, etype='intra' if cu == cv else 'cross')

    # Add 8 utility nodes (fixed G layer)
    utility_names = list(UTILITY_NODES.keys())
    for uname, udata in UTILITY_NODES.items():
        G.add_node(uname, layer='utility',
                   base_activation=udata['base'],
                   activation=udata['base'])

    # Connect utilities to ~15% of memes
    for uname in utility_names:
        n_connections = max(1, int(0.15 * N))
        targets = rng.choice(meme_names, size=n_connections, replace=False)
        for mname in targets:
            r = rng.random()
            if r < 0.60:
                etype = 'redirect'
            elif r < 0.85:
                etype = 'suppress'
            else:
                etype = 'interpret'
            w = rng.uniform(0.2, 0.7)
            G.add_edge(uname, mname, weight=w, etype=etype)

    # Incompatibility: per utility, ~10% of memes get random(0.1, 0.5)
    incompatibility = {}
    for uname in utility_names:
        incompatibility[uname] = {}
        n_inc = max(1, int(0.10 * N))
        inc_targets = rng.choice(meme_names, size=n_inc, replace=False)
        for mname in inc_targets:
            incompatibility[uname][mname] = rng.uniform(0.1, 0.5)

    # SMC: smc_fraction × N memes, high-degree preferred, 70% L1 / 30% L2
    n_smc = int(smc_fraction * N)
    if n_smc > 0:
        n_smc = max(2, n_smc)  # at least 2 if non-zero
    smc_set = set()
    smc_levels = {}
    if n_smc > 0:
        degrees = np.array([ba.degree(i) for i in range(N)], dtype=float)
        degrees /= degrees.sum()
        smc_indices = rng.choice(N, size=n_smc, replace=False, p=degrees)
        smc_set = set(meme_names[i] for i in smc_indices)
        for m in smc_set:
            smc_levels[m] = 2 if rng.random() < 0.30 else 1

    # Open memes: max(3, N//100) with is_open=True
    n_open = max(3, N // 100)
    open_indices = rng.choice(N, size=min(n_open, N), replace=False)
    for i in open_indices:
        mname = meme_names[i]
        G.nodes[mname]['is_open'] = True
        G.nodes[mname]['closure'] = rng.uniform(0.1, 0.4)

    return G, incompatibility, smc_set, smc_levels


# ─────────────────────────────────────────────────────────────────────
# 2. Sampled σ_SW (avoids O(N²) exact shortest paths)
# ─────────────────────────────────────────────────────────────────────

def compute_sigma_sw_sampled(G, meme_nodes, n_bfs=200, seed=42):
    """Compute small-worldness σ for the meme subgraph.

    Uses analytical ER baseline (C_rand=p, L_rand=ln(N)/ln(Np)) for all N.
    L_actual: exact for N≤2000, sampled BFS for N>2000.
    """
    meme_sub = G.subgraph(meme_nodes).copy()
    if meme_sub.number_of_nodes() < 4:
        return 1.0

    # Use largest connected component
    if not nx.is_connected(meme_sub):
        largest_cc = max(nx.connected_components(meme_sub), key=len)
        meme_sub = meme_sub.subgraph(largest_cc).copy()

    n = meme_sub.number_of_nodes()
    m = meme_sub.number_of_edges()
    if n < 4:
        return 1.0

    C_actual = nx.average_clustering(meme_sub)
    p = 2 * m / (n * (n - 1))

    # L_actual: exact when feasible, sampled BFS for large graphs
    if n <= 2000:
        L_actual = nx.average_shortest_path_length(meme_sub)
    else:
        rng = np.random.RandomState(seed)
        nodes = list(meme_sub.nodes())
        sources = rng.choice(nodes, size=min(n_bfs, n), replace=False)
        total_dist = 0
        total_pairs = 0
        for src in sources:
            lengths = nx.single_source_shortest_path_length(meme_sub, src)
            total_dist += sum(lengths.values())
            total_pairs += len(lengths) - 1  # exclude self
        L_actual = total_dist / max(total_pairs, 1)

    # Analytical ER baseline — consistent across all N
    np_val = n * p
    C_rand = max(p, 1e-6)
    L_rand = np.log(n) / np.log(max(np_val, 1.01))

    sigma = (C_actual / max(C_rand, 1e-6)) / (L_actual / max(L_rand, 1e-6))
    return sigma


# ─────────────────────────────────────────────────────────────────────
# 3. Worker Function
# ─────────────────────────────────────────────────────────────────────

def run_scale_point(args):
    """Run one (N, seed) simulation point. Returns dict of metrics."""
    N, seed, n_steps, smc_fraction = args

    t0 = time.time()

    # Generate synthetic graph
    G, incompatibility, smc_set, smc_levels = generate_synthetic_bmc(
        N, seed=seed, smc_fraction=smc_fraction)

    # Create simulator
    sim = BMCSimulator(G, incompatibility, smc_memes=smc_set, smc_levels=smc_levels)
    sim.compact_history = True  # skip activations/edge_weights copy (massive speedup)

    # Compute + inject sampled σ_SW
    t_sigma_start = time.time()
    meme_nodes = sim.meme_nodes
    sigma = compute_sigma_sw_sampled(sim.G, meme_nodes, seed=seed)
    t_sigma = time.time() - t_sigma_start
    sim._sigma_sw_cache = sigma
    sim._sigma_sw_dirty = False

    # Config overrides for speed + isolation
    import bmc.config as cfg
    orig_q = cfg.Q_CHECK_INTERVAL
    orig_si = cfg.SIGN_INVERSION_ENABLED
    cfg.Q_CHECK_INTERVAL = 999999
    cfg.SIGN_INVERSION_ENABLED = False

    # Build stimuli schedule (generic, proportional phases)
    rng = np.random.RandomState(seed + 1000)
    warmup_end = int(n_steps * 0.2)
    active_end = int(n_steps * 0.8)
    stimuli_schedule = {}
    for t in range(n_steps):
        stim = {}
        if t < active_end:
            stim['SEEKING'] = 0.2
        if warmup_end <= t < active_end:
            # Random 5% memes get +0.15 each step
            n_stim = max(1, int(0.05 * N))
            stim_targets = rng.choice(meme_nodes, size=n_stim, replace=False)
            for m in stim_targets:
                stim[m] = 0.15
        stimuli_schedule[t] = stim

    # Run step loop with memory compaction
    step_data = []
    for t in range(n_steps):
        stim = stimuli_schedule.get(t, {})
        sim.step(stim)
        entry = sim.history[-1]
        step_data.append({
            'cl': entry['cl'],
            'a_smc': entry['a_smc'],
            'balance': entry['balance'],
            'conflict': entry['conflict'],
            'sit_total': entry['sit_total'],
            'fatigue': entry['fatigue'],
        })
        sim.history[-1] = {}  # free memory (~35 MB per entry at N=50K)

    # Restore config
    cfg.Q_CHECK_INTERVAL = orig_q
    cfg.SIGN_INVERSION_ENABLED = orig_si

    t_total = time.time() - t0
    t_per_step = (t_total - t_sigma) / n_steps

    # Steady-state: last 20% of steps
    ss_start = int(n_steps * 0.8)
    ss = step_data[ss_start:]
    cl_vals = [s['cl'] for s in ss]
    a_smc_vals = [s['a_smc'] for s in ss]
    balance_vals = [s['balance'] for s in ss]

    # σ_norm (same formula as compute_cl)
    sigma_norm = 1.0 - np.exp(-sigma / 5.0)

    return {
        'N': N,
        'seed': seed,
        'smc_fraction': smc_fraction,
        'sigma_sw': float(sigma),
        'sigma_norm': float(sigma_norm),
        'cl_mean': float(np.mean(cl_vals)),
        'cl_std': float(np.std(cl_vals)),
        'a_smc_mean': float(np.mean(a_smc_vals)),
        'balance_mean': float(np.mean(balance_vals)),
        'time_total': float(t_total),
        'time_sigma': float(t_sigma),
        'time_per_step': float(t_per_step),
        'timeseries_cl': [s['cl'] for s in step_data],
    }


# ─────────────────────────────────────────────────────────────────────
# 4. Experiments
# ─────────────────────────────────────────────────────────────────────

SCALE_POINTS = [100, 500, 1000, 5000, 10000]
PHASE_FRACTIONS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]
N_WORKERS = 8

# Adaptive params: more seeds/steps at small N (cheap), fewer at large N (expensive)
_SEEDS_A = {100: 10, 500: 10, 1000: 10, 5000: 5, 10000: 3}
_SEEDS_B = {100: 5, 500: 5, 1000: 5, 5000: 3, 10000: 3}
_N_STEPS = {100: 100, 500: 100, 1000: 100, 5000: 100, 10000: 50}


def _save_checkpoint(results_a, results_b, out_dir):
    """Incremental checkpoint — save current results to disk."""
    cp = {
        'experiment_a': [{k: v for k, v in r.items() if k != 'timeseries_cl'}
                         for r in results_a],
        'experiment_b': [{k: v for k, v in r.items() if k != 'timeseries_cl'}
                         for r in results_b],
        'metadata': {
            'scale_points': SCALE_POINTS,
            'phase_fractions': PHASE_FRACTIONS,
            'n_steps_by_N': _N_STEPS,
            'n_workers': N_WORKERS,
            'checkpoint': True,
            'a_done': len(results_a),
            'b_done': len(results_b),
        }
    }
    cp_path = os.path.join(out_dir, 'scaling_checkpoint.json')
    with open(cp_path, 'w') as f:
        json.dump(cp, f, indent=2)


def run_experiment_a(pool, out_dir):
    """Experiment A — Scaling: convergence + self-similarity + complexity.
    Adaptive seeds per N, 100 steps, fixed smc_fraction=0.06.
    """
    tasks = []
    for N in SCALE_POINTS:
        n_seeds = _SEEDS_A.get(N, 5)
        n_steps = _N_STEPS.get(N, 100)
        for seed in range(n_seeds):
            tasks.append((N, seed, n_steps, 0.06))

    total = len(tasks)
    print(f"Experiment A: {total} runs across {len(SCALE_POINTS)} scales")
    results = []
    pbar = tqdm(total=total, desc="Exp A", unit="run",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for result in pool.imap_unordered(run_scale_point, tasks):
        results.append(result)
        pbar.set_postfix_str(f"N={result['N']} CL={result['cl_mean']:.4f} σ={result['sigma_sw']:.1f}")
        pbar.update(1)
        _save_checkpoint(results, [], out_dir)
    pbar.close()
    return results


def run_experiment_b(pool, results_a, out_dir):
    """Experiment B — Phase Transition: CL vs SMC fraction.
    Same N values, 6 smc fractions, adaptive seeds per (N, fraction).
    """
    tasks = []
    for N in SCALE_POINTS:
        n_seeds = _SEEDS_B.get(N, 3)
        n_steps = _N_STEPS.get(N, 100)
        for frac in PHASE_FRACTIONS:
            for seed in range(n_seeds):
                tasks.append((N, seed + 100, n_steps, frac))

    total = len(tasks)
    print(f"Experiment B: {total} runs across {len(SCALE_POINTS)} scales × {len(PHASE_FRACTIONS)} fractions")
    results = []
    pbar = tqdm(total=total, desc="Exp B", unit="run",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for result in pool.imap_unordered(run_scale_point, tasks):
        results.append(result)
        pbar.set_postfix_str(f"N={result['N']} f={result['smc_fraction']:.2f} CL={result['cl_mean']:.4f}")
        pbar.update(1)
        _save_checkpoint(results_a, results, out_dir)
    pbar.close()
    return results


# ─────────────────────────────────────────────────────────────────────
# 5. Plotting
# ─────────────────────────────────────────────────────────────────────

def _aggregate_by_N(results, key):
    """Group results by N, return {N: (mean, std)}."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[r['N']].append(r[key])
    out = {}
    for N in sorted(groups):
        vals = groups[N]
        out[N] = (np.mean(vals), np.std(vals))
    return out


def _aggregate_phase(results):
    """Group results by (N, smc_fraction), return nested dict."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r['N'], r['smc_fraction'])].append(r['cl_mean'])
    out = {}
    for (N, frac), vals in groups.items():
        out.setdefault(N, {})[frac] = (np.mean(vals), np.std(vals))
    return out


def plot_sigma_sw(results_a, out_dir):
    """Plot 1: σ_SW vs N (semilog-x)."""
    agg = _aggregate_by_N(results_a, 'sigma_sw')
    Ns = sorted(agg.keys())
    means = [agg[N][0] for N in Ns]
    stds = [agg[N][1] for N in Ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(Ns, means, yerr=stds, fmt='o-', capsize=4, linewidth=2, markersize=6)
    ax.set_xscale('log')
    ax.set_xlabel('N (meme nodes)', fontsize=12)
    ax.set_ylabel('σ_SW (small-worldness)', fontsize=12)
    ax.set_title('Small-Worldness Convergence Across Scale', fontsize=14)
    ax.axhline(y=np.mean(means[-2:]), color='gray', linestyle='--', alpha=0.5,
               label=f'Large-N mean: {np.mean(means[-2:]):.2f}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'scaling_sigma_sw.png'), dpi=150)
    plt.close(fig)
    print("  → scaling_sigma_sw.png")


def plot_cl(results_a, out_dir):
    """Plot 2: CL vs N (semilog-x)."""
    agg = _aggregate_by_N(results_a, 'cl_mean')
    Ns = sorted(agg.keys())
    means = [agg[N][0] for N in Ns]
    stds = [agg[N][1] for N in Ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(Ns, means, yerr=stds, fmt='s-', capsize=4, linewidth=2, markersize=6,
                color='#e74c3c')
    ax.set_xscale('log')
    ax.set_xlabel('N (meme nodes)', fontsize=12)
    ax.set_ylabel('CL (Consciousness Level)', fontsize=12)
    ax.set_title('CL Scale-Invariance', fontsize=14)
    # Show CI band
    grand_mean = np.mean(means)
    ax.axhline(y=grand_mean, color='gray', linestyle='--', alpha=0.5,
               label=f'Grand mean: {grand_mean:.4f}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'scaling_cl.png'), dpi=150)
    plt.close(fig)
    print("  → scaling_cl.png")


def plot_time(results_a, out_dir):
    """Plot 3: Wall-clock time vs N (log-log)."""
    agg_total = _aggregate_by_N(results_a, 'time_total')
    agg_step = _aggregate_by_N(results_a, 'time_per_step')
    Ns = sorted(agg_total.keys())
    t_total = [agg_total[N][0] for N in Ns]
    t_step = [agg_step[N][0] for N in Ns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Total time
    ax1.loglog(Ns, t_total, 'o-', linewidth=2, markersize=6, label='BMC total (100 steps)')
    # O(N log N) reference
    ref_base = t_total[0] / (Ns[0] * np.log(Ns[0]))
    ref = [ref_base * N * np.log(N) for N in Ns]
    ax1.loglog(Ns, ref, '--', color='gray', alpha=0.6, label='O(N log N) reference')
    # IIT reference (exponential, shown symbolically)
    ax1.annotate('IIT Φ: O(2^N)\n(infeasible at N > 20)', xy=(0.65, 0.85),
                 xycoords='axes fraction', fontsize=9, color='red',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax1.set_xlabel('N', fontsize=12)
    ax1.set_ylabel('Total time (s)', fontsize=12)
    ax1.set_title('Total Computation Time', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Per-step time
    ax2.loglog(Ns, t_step, 's-', linewidth=2, markersize=6, color='#2ecc71',
               label='Per-step time')
    ax2.set_xlabel('N', fontsize=12)
    ax2.set_ylabel('Time per step (s)', fontsize=12)
    ax2.set_title('Per-Step Computation Time', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'scaling_time.png'), dpi=150)
    plt.close(fig)
    print("  → scaling_time.png")


def plot_phase_transition(results_b, out_dir):
    """Plot 4: CL vs SMC fraction, one curve per N."""
    agg = _aggregate_phase(results_b)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SCALE_POINTS)))

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, N in enumerate(SCALE_POINTS):
        if N not in agg:
            continue
        fracs = sorted(agg[N].keys())
        means = [agg[N][f][0] for f in fracs]
        stds = [agg[N][f][1] for f in fracs]
        ax.errorbar(fracs, means, yerr=stds, fmt='o-', capsize=3,
                    linewidth=2, markersize=5, color=colors[i],
                    label=f'N={N}')

    ax.set_xlabel('SMC fraction', fontsize=12)
    ax.set_ylabel('CL (steady-state mean)', fontsize=12)
    ax.set_title('Phase Transition: CL vs Self-Model Size', fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'scaling_phase_transition.png'), dpi=150)
    plt.close(fig)
    print("  → scaling_phase_transition.png")


def plot_summary(results_a, results_b, out_dir):
    """Plot 5: 2×2 combined panel."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: σ_SW vs N
    ax = axes[0, 0]
    agg = _aggregate_by_N(results_a, 'sigma_sw')
    Ns = sorted(agg.keys())
    means = [agg[N][0] for N in Ns]
    stds = [agg[N][1] for N in Ns]
    ax.errorbar(Ns, means, yerr=stds, fmt='o-', capsize=4, linewidth=2, markersize=5)
    ax.set_xscale('log')
    ax.set_xlabel('N')
    ax.set_ylabel('σ_SW')
    ax.set_title('(a) Small-Worldness Convergence')
    ax.grid(True, alpha=0.3)

    # Panel 2: CL vs N
    ax = axes[0, 1]
    agg = _aggregate_by_N(results_a, 'cl_mean')
    Ns = sorted(agg.keys())
    means = [agg[N][0] for N in Ns]
    stds = [agg[N][1] for N in Ns]
    ax.errorbar(Ns, means, yerr=stds, fmt='s-', capsize=4, linewidth=2, markersize=5,
                color='#e74c3c')
    ax.set_xscale('log')
    ax.set_xlabel('N')
    ax.set_ylabel('CL')
    ax.set_title('(b) CL Scale-Invariance')
    ax.grid(True, alpha=0.3)

    # Panel 3: Time scaling
    ax = axes[1, 0]
    agg_total = _aggregate_by_N(results_a, 'time_total')
    Ns = sorted(agg_total.keys())
    t_total = [agg_total[N][0] for N in Ns]
    ax.loglog(Ns, t_total, 'o-', linewidth=2, markersize=5, label='BMC')
    ref_base = t_total[0] / (Ns[0] * np.log(Ns[0]))
    ref = [ref_base * N * np.log(N) for N in Ns]
    ax.loglog(Ns, ref, '--', color='gray', alpha=0.6, label='O(N log N)')
    ax.set_xlabel('N')
    ax.set_ylabel('Total time (s)')
    ax.set_title('(c) Computation Time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Phase transition
    ax = axes[1, 1]
    agg = _aggregate_phase(results_b)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SCALE_POINTS)))
    for i, N in enumerate(SCALE_POINTS):
        if N not in agg:
            continue
        fracs = sorted(agg[N].keys())
        means = [agg[N][f][0] for f in fracs]
        stds = [agg[N][f][1] for f in fracs]
        ax.errorbar(fracs, means, yerr=stds, fmt='o-', capsize=2,
                    linewidth=1.5, markersize=4, color=colors[i],
                    label=f'N={N}')
    ax.set_xlabel('SMC fraction')
    ax.set_ylabel('CL')
    ax.set_title('(d) Phase Transition: CL vs SMC Size')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle('BMC Scaling Analysis — Block 6', fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, 'scaling_summary.png'), dpi=150)
    plt.close(fig)
    print("  → scaling_summary.png")


# ─────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Block 6: BMC Scaling Analysis")
    print("=" * 60)

    t_start = time.time()

    with Pool(processes=N_WORKERS) as pool:
        print("\n── Experiment A: Scaling ──")
        results_a = run_experiment_a(pool, out_dir)
        t_a = time.time() - t_start
        print(f"  Done in {t_a:.1f}s")

        print("\n── Experiment B: Phase Transition ──")
        t_b_start = time.time()
        results_b = run_experiment_b(pool, results_a, out_dir)
        t_b = time.time() - t_b_start
        print(f"  Done in {t_b:.1f}s")

    t_total = time.time() - t_start
    print(f"\nTotal compute: {t_total:.1f}s")

    # Save JSON (strip timeseries for compactness)
    json_results = {
        'experiment_a': [{k: v for k, v in r.items() if k != 'timeseries_cl'}
                         for r in results_a],
        'experiment_b': [{k: v for k, v in r.items() if k != 'timeseries_cl'}
                         for r in results_b],
        'metadata': {
            'scale_points': SCALE_POINTS,
            'phase_fractions': PHASE_FRACTIONS,
            'n_steps_by_N': _N_STEPS,
            'n_workers': N_WORKERS,
            'total_time_s': t_total,
        }
    }
    json_path = os.path.join(out_dir, 'scaling_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nJSON: {json_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_sigma_sw(results_a, out_dir)
    plot_cl(results_a, out_dir)
    plot_time(results_a, out_dir)
    plot_phase_transition(results_b, out_dir)
    plot_summary(results_a, results_b, out_dir)

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary — Experiment A (smc_fraction=0.06)")
    print("-" * 60)
    print(f"{'N':>8} {'σ_SW':>10} {'σ_norm':>10} {'CL':>12} {'t/step':>10}")
    print("-" * 60)
    agg_sigma = _aggregate_by_N(results_a, 'sigma_sw')
    agg_cl = _aggregate_by_N(results_a, 'cl_mean')
    agg_sigma_norm = _aggregate_by_N(results_a, 'sigma_norm')
    agg_tstep = _aggregate_by_N(results_a, 'time_per_step')
    for N in SCALE_POINTS:
        sw_m, sw_s = agg_sigma[N]
        sn_m, _ = agg_sigma_norm[N]
        cl_m, cl_s = agg_cl[N]
        ts_m, _ = agg_tstep[N]
        print(f"{N:>8} {sw_m:>7.2f}±{sw_s:.2f} {sn_m:>10.4f} "
              f"{cl_m:>8.4f}±{cl_s:.4f} {ts_m:>8.3f}s")

    print("\n" + "=" * 60)
    print("Summary — Experiment B (Phase Transition)")
    print("-" * 60)
    agg_phase = _aggregate_phase(results_b)
    header = f"{'N':>8}"
    for frac in PHASE_FRACTIONS:
        header += f"  f={frac:<6}"
    print(header)
    print("-" * (8 + len(PHASE_FRACTIONS) * 8))
    for N in SCALE_POINTS:
        if N not in agg_phase:
            continue
        row = f"{N:>8}"
        for frac in PHASE_FRACTIONS:
            if frac in agg_phase[N]:
                m, s = agg_phase[N][frac]
                row += f"  {m:.4f}"
            else:
                row += f"  {'—':>6}"
        print(row)

    print("\n" + "=" * 60)
    print("Block 6 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
