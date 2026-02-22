"""Monte Carlo orchestrator: run_monte_carlo() and report."""
import time
import multiprocessing as mp
import numpy as np

from .mc_worker import run_seed, init_worker
from .graph import build_bmc_graph, MEME_TO_CLUSTER, SMC_MEMES, SMC_LEVEL_1, SMC_LEVEL_2
from .analysis import detect_structural_balance
from .simulator import BMCSimulator
from .config import CONSOLIDATION_PRUNE_THRESHOLD

# SMC levels dict (built once, reused)
_SMC_LEVELS = {m: 1 for m in SMC_LEVEL_1}
_SMC_LEVELS.update({m: 2 for m in SMC_LEVEL_2})


def ci95(arr):
    """95% confidence interval: (mean, lo, hi)."""
    arr = np.array(arr, dtype=float)
    mean = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(len(arr))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def _create_simulator(seed=42, **kw):
    G_mc, INCOMPAT_mc = build_bmc_graph(seed=seed)
    return BMCSimulator(G_mc, INCOMPAT_mc,
                        smc_memes=SMC_MEMES, smc_levels=_SMC_LEVELS)


def run_monte_carlo(n_seeds=50, base_seed=42, n_workers=None, progress_cb=None):
    """Run all 8 scenarios across n_seeds graph realizations.

    Args:
        n_seeds: number of independent seeds
        base_seed: starting seed
        n_workers: number of parallel workers (default: min(cpu_count, 8))
        progress_cb: callable(i, n_seeds) called after each seed completes

    Returns:
        list of dicts (one per seed)
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    mc_seeds = list(range(base_seed, base_seed + n_seeds))

    engine_globals = {
        'create_simulator': _create_simulator,
        'detect_structural_balance': detect_structural_balance,
        'MEME_TO_CLUSTER': MEME_TO_CLUSTER,
        'CONSOLIDATION_PRUNE_THRESHOLD': CONSOLIDATION_PRUNE_THRESHOLD,
    }

    seed_results = []
    ctx = mp.get_context('fork')
    with ctx.Pool(n_workers, initializer=init_worker, initargs=(engine_globals,)) as pool:
        for i, result in enumerate(pool.imap_unordered(run_seed, mc_seeds), 1):
            seed_results.append(result)
            if progress_cb is not None:
                progress_cb(i, n_seeds)

    return seed_results


def report_monte_carlo(results, n_mc=None):
    """Format Monte Carlo results as a printable table string."""
    if n_mc is None:
        n_mc = len(results)

    mc = {}
    for key in results[0]:
        mc[key] = [r[key] for r in results]

    def _fmt_ci(lo, hi, decimals=3):
        fmt = f'%.{decimals}f'
        return f'[{fmt % lo}, {fmt % hi}]'

    W_NAME, W_MEAN, W_CI, W_PASS = 36, 10, 20, 8
    lines = []
    lines.append(f"{'Metric':<{W_NAME}} {'Mean':>{W_MEAN}} {'95% CI':>{W_CI}} {'Pass':>{W_PASS}}")
    lines.append('=' * ((W_NAME + W_MEAN + W_CI + W_PASS) + 3))

    m, lo, hi = ci95(mc['S1_peak_balance'])
    lines.append(f"{'S1: Peak balance':<{W_NAME}} {m:>{W_MEAN}.3f} {_fmt_ci(lo,hi):>{W_CI}} {'--':>{W_PASS}}")

    m, lo, hi = ci95(mc['S2_min_balance'])
    lt1 = sum(1 for x in mc['S2_min_balance'] if x < 1.0)
    lines.append(f"{'S2: Min balance (stress)':<{W_NAME}} {m:>{W_MEAN}.3f} {_fmt_ci(lo,hi):>{W_CI}} {f'{lt1}/{n_mc}':>{W_PASS}}")

    m, lo, hi = ci95(mc['S3_hub_drop_pct'])
    pos = sum(1 for x in mc['S3_hub_drop_pct'] if x > 0)
    lines.append(f"{'S3: Hub drop %':<{W_NAME}} {m:>{W_MEAN}.1f} {_fmt_ci(lo,hi,1):>{W_CI}} {f'{pos}/{n_mc}':>{W_PASS}}")

    m, lo, hi = ci95(mc['S3_fear_increase_pct'])
    pos3f = sum(1 for x in mc['S3_fear_increase_pct'] if x > 0)
    lines.append(f"{'S3: FEAR increase %':<{W_NAME}} {m:>{W_MEAN}.1f} {_fmt_ci(lo,hi,1):>{W_CI}} {f'{pos3f}/{n_mc}':>{W_PASS}}")

    m, lo, hi = ci95(mc['S4_peak_fatigue'])
    pos4 = sum(1 for x in mc['S4_peak_fatigue'] if x > 0.1)
    lines.append(f"{'S4: Peak fatigue':<{W_NAME}} {m:>{W_MEAN}.3f} {_fmt_ci(lo,hi):>{W_CI}} {f'{pos4}/{n_mc}':>{W_PASS}}")

    m, lo, hi = ci95(mc['S5a_score'])
    acc = sum(mc['S5a_accepted'])
    lines.append(f"{'S5a: Compatible score':<{W_NAME}} {m:>{W_MEAN}.3f} {_fmt_ci(lo,hi):>{W_CI}} {f'{acc}/{n_mc}':>{W_PASS}}")

    m, lo, hi = ci95(mc['S5b_score'])
    rej = sum(mc['S5b_accepted'])
    lines.append(f"{'S5b: Incompatible score':<{W_NAME}} {m:>{W_MEAN}.3f} {_fmt_ci(lo,hi):>{W_CI}} {f'{rej}/{n_mc}':>{W_PASS}}")

    ec_pass = sum(mc['S6_ec_shift'])
    lines.append(f"{'S6: WD shift (alt > orig.)':<{W_NAME}} {'':>{W_MEAN}} {'':>{W_CI}} {f'{ec_pass}/{n_mc}':>{W_PASS}}")

    m, lo, hi = ci95(mc['S7_blend_count'])
    blend_pos = sum(1 for x in mc['S7_blend_count'] if x > 0)
    lines.append(f"{'S7: BLEND nodes created':<{W_NAME}} {m:>{W_MEAN}.2f} {_fmt_ci(lo,hi,2):>{W_CI}} {f'{blend_pos}/{n_mc}':>{W_PASS}}")

    sf = sum(mc['S8_sign_flip'])
    lines.append(f"{'S8: Sleeper sign flip':<{W_NAME}} {'':>{W_MEAN}} {'':>{W_CI}} {f'{sf}/{n_mc}':>{W_PASS}}")

    sit_present = sum(mc['S9_sit_present'])
    lines.append(f"{'S9: SIT present':<{W_NAME}} {'':>{W_MEAN}} {'':>{W_CI}} {f'{sit_present}/{n_mc}':>{W_PASS}}")

    fc_reduces = sum(mc['S9_fc_reduces_sit'])
    lines.append(f"{'S9: False closure → SIT↓':<{W_NAME}} {'':>{W_MEAN}} {'':>{W_CI}} {f'{fc_reduces}/{n_mc}':>{W_PASS}}")

    m_cl, lo_cl, hi_cl = ci95(mc['S10_cl_max'])
    cl_pos = sum(mc['S10_cl_positive'])
    lines.append(f"{'S10: CL metric > 0':<{W_NAME}} {m_cl:>{W_MEAN}.4f} {_fmt_ci(lo_cl,hi_cl,4):>{W_CI}} {f'{cl_pos}/{n_mc}':>{W_PASS}}")

    inv_has = sum(mc['S11_has_inversions'])
    m_inv, lo_inv, hi_inv = ci95(mc['S11_inversions'])
    lines.append(f"{'S11: Sign inversions occur':<{W_NAME}} {m_inv:>{W_MEAN}.1f} {_fmt_ci(lo_inv,hi_inv,1):>{W_CI}} {f'{inv_has}/{n_mc}':>{W_PASS}}")

    cl_inc = sum(mc['S12_closure_increased'])
    m_clo, lo_clo, hi_clo = ci95(mc['S12_closure'])
    lines.append(f"{'S12: Action → closure↑':<{W_NAME}} {m_clo:>{W_MEAN}.3f} {_fmt_ci(lo_clo,hi_clo):>{W_CI}} {f'{cl_inc}/{n_mc}':>{W_PASS}}")

    m_bal, _, _ = ci95(mc['balance_ratio'])
    m_amb, _, _ = ci95(mc['mean_ambivalence'])
    lines.append(f"\n{'Structural balance ratio (mean)':<{W_NAME}} {m_bal:>{W_MEAN}.3f}")
    lines.append(f"{'Mean ambivalence':<{W_NAME}} {m_amb:>{W_MEAN}.4f}")

    total_pass = (lt1 + pos + pos3f + pos4 + acc + rej + ec_pass + blend_pos + sf
                  + sit_present + fc_reduces + cl_pos + inv_has + cl_inc)
    total_checks = 14 * n_mc
    pct_pass = 100 * total_pass / total_checks
    lines.append(f"\nOverall directional pass rate: {total_pass}/{total_checks} ({pct_pass:.0f}%)")

    return '\n'.join(lines)
