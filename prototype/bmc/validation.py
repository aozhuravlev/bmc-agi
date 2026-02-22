"""Validation: sensitivity analysis + ablation study for BMC prototype.

Run from terminal (from prototype/ directory):
    python -m bmc.validation                 # both
    python -m bmc.validation sensitivity     # sensitivity only
    python -m bmc.validation ablation        # ablation only

Results are saved to prototype/results/*.json
"""
import sys
import json
import time
import multiprocessing as mp
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

import bmc.config as _cfg
import bmc.simulator as _sim_mod
from .mc_worker import run_seed, init_worker
from .graph import build_bmc_graph, MEME_TO_CLUSTER, SMC_MEMES, SMC_LEVEL_1, SMC_LEVEL_2
from .analysis import detect_structural_balance
from .simulator import BMCSimulator

_SMC_LEVELS = {m: 1 for m in SMC_LEVEL_1}
_SMC_LEVELS.update({m: 2 for m in SMC_LEVEL_2})

# ── Check names (19 directional checks) ──
CHECK_NAMES = [
    'S2_stress', 'S3_hub_drop', 'S3_fear', 'S4_fatigue',
    'S5a_accept', 'S5b_reject', 'S6_hub_displ', 'S7_blend',
    'S8_sleeper', 'S9_sit', 'S9_fc', 'S10_cl',
    'S11_inversion', 'S12_action',
    'S13_sit_persist', 'S14_q_reorg', 'S15_rum_archive',
    'S16_spaced_rep', 'S17_util_inertia',
    'S18_kappa_trans', 'S19_kappa_decay',
]

# ── Sensitivity: 16 key hyperparameters ──
SENSITIVITY_PARAMS = [
    'LAMBDA_DECAY', 'LAMBDA_EDGE_POS', 'LAMBDA_EDGE_NEG',
    'THETA_HIGH', 'THETA_LOW', 'SIGMOID_GAIN', 'SOFTMAX_TEMP',
    'WM_SIZE', 'FATIGUE_RATE', 'SPACED_REP_FACTOR',
    'GAMMA_SIT', 'T_SEEK', 'ALPHA_CENTRALITY',
    'DELTA_W_REACTIVATE', 'SMC_RECURSION_WEIGHT', 'CL_BALANCE_OPT',
]

SENSITIVITY_MULTIPLIERS = [0.7, 0.85, 1.0, 1.15, 1.3]

# ── Ablation definitions ──
# Keys starting with '_' are special directives, not config overrides.
ABLATION_DEFS = {
    'baseline': {},
    'no_signed_edges': {'_special': 'abs_edges'},
    'no_SIT': {'GAMMA_SIT': 0.0, 'OPEN_MEME_PULSE_AMP': 0.0, 'OPEN_MEME_DECAY_RESISTANCE': 1.0},
    'no_SMC_CL': {'_special': 'no_smc'},
    'no_spaced_rep': {'SPACED_REP_FACTOR': 0.0},
    'no_BLEND': {'BLEND_THRESHOLD': 999.0},
    'no_fatigue': {'FATIGUE_RATE': 0.0},
    'no_utility_inertia': {'UTILITY_INERTIA': 0.0},
    'no_sign_inversion': {'SIGN_INVERSION_ENABLED': False},
    'no_Q_dynamics': {'Q_CHECK_INTERVAL': 999999},
    'no_rumination_limiter': {'RUMINATION_E_MAX': 999999.0, 'RUMINATION_MAX_CYCLES': 999999},
    'no_kappa': {'KAPPA_ENABLED': False},
}

# ── Helpers ──

_MISSING = object()


def _apply_overrides(overrides):
    """Patch config values in both config and simulator modules. Returns originals."""
    originals = {}
    for key, value in overrides.items():
        if key.startswith('_'):
            continue
        orig_cfg = getattr(_cfg, key)
        orig_sim = getattr(_sim_mod, key, _MISSING)
        originals[key] = (orig_cfg, orig_sim)
        setattr(_cfg, key, value)
        if orig_sim is not _MISSING:
            setattr(_sim_mod, key, value)
    return originals


def _restore_overrides(originals):
    """Restore original config values."""
    for key, (cfg_val, sim_val) in originals.items():
        setattr(_cfg, key, cfg_val)
        if sim_val is not _MISSING:
            setattr(_sim_mod, key, sim_val)


def _create_sim_default(seed=42):
    G, INCOMPAT = build_bmc_graph(seed=seed)
    return BMCSimulator(G, INCOMPAT, smc_memes=SMC_MEMES, smc_levels=_SMC_LEVELS)


def _create_sim_abs_edges(seed=42):
    """All edges → abs (ablation: no signed edges)."""
    sim = _create_sim_default(seed)
    for key in sim.edge_weights:
        sim.edge_weights[key] = abs(sim.edge_weights[key])
    return sim


def _create_sim_no_smc(seed=42):
    """Empty SMC set (ablation: no SMC/CL)."""
    G, INCOMPAT = build_bmc_graph(seed=seed)
    return BMCSimulator(G, INCOMPAT, smc_memes=set(), smc_levels={})


def _run_mc(n_seeds, base_seed, n_workers, create_sim_fn=None):
    """Run MC with given simulator factory. Config must be pre-patched."""
    if create_sim_fn is None:
        create_sim_fn = _create_sim_default

    engine_globals = {
        'create_simulator': create_sim_fn,
        'detect_structural_balance': detect_structural_balance,
        'MEME_TO_CLUSTER': MEME_TO_CLUSTER,
        'CONSOLIDATION_PRUNE_THRESHOLD': getattr(_cfg, 'CONSOLIDATION_PRUNE_THRESHOLD'),
    }

    seeds = list(range(base_seed, base_seed + n_seeds))
    ctx = mp.get_context('fork')
    with ctx.Pool(n_workers, initializer=init_worker, initargs=(engine_globals,)) as pool:
        results = list(pool.imap_unordered(run_seed, seeds))

    return results


def _compute_checks(results):
    """Per-check pass counts + overall pass rate."""
    n = len(results)
    mc = {}
    for key in results[0]:
        mc[key] = [r[key] for r in results]

    checks = {
        'S2_stress':    sum(1 for x in mc['S2_min_balance'] if x < 1.0),
        'S3_hub_drop':  sum(1 for x in mc['S3_hub_drop_pct'] if x > 0),
        'S3_fear':      sum(1 for x in mc['S3_fear_increase_pct'] if x > 0),
        'S4_fatigue':   sum(1 for x in mc['S4_peak_fatigue'] if x > 0.1),
        'S5a_accept':   sum(mc['S5a_accepted']),
        'S5b_reject':   sum(mc['S5b_accepted']),
        'S6_hub_displ': sum(mc['S6_ec_shift']),
        'S7_blend':     sum(1 for x in mc['S7_blend_count'] if x > 0),
        'S8_sleeper':   sum(mc['S8_sign_flip']),
        'S9_sit':       sum(mc['S9_sit_present']),
        'S9_fc':        sum(mc['S9_fc_reduces_sit']),
        'S10_cl':       sum(mc['S10_cl_positive']),
        'S11_inversion': sum(mc['S11_has_inversions']),
        'S12_action':   sum(mc['S12_closure_increased']),
        'S13_sit_persist': sum(mc['S13_sit_persistence']),
        'S14_q_reorg':     sum(mc['S14_q_reorganization']),
        'S15_rum_archive': sum(mc['S15_rumination_archival']),
        'S16_spaced_rep':  sum(mc['S16_spaced_rep']),
        'S17_util_inertia': sum(mc['S17_utility_inertia']),
        'S18_kappa_trans': sum(mc['S18_kappa_transition']),
        'S19_kappa_decay': sum(mc['S19_kappa_decay']),
    }

    total_pass = sum(checks.values())
    total_checks = len(CHECK_NAMES) * n
    pct = 100 * total_pass / total_checks
    return checks, total_pass, total_checks, pct


# ══════════════════════════════════════════════════════════════════════
#  Sensitivity analysis
# ══════════════════════════════════════════════════════════════════════

def run_sensitivity(n_seeds=50, base_seed=42, n_workers=None,
                    output_path='results/sensitivity.json'):
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    n_configs = len(SENSITIVITY_PARAMS) * (len(SENSITIVITY_MULTIPLIERS) - 1) + 1

    # ── baseline ──
    pbar = tqdm(total=n_configs, desc='Sensitivity', unit='cfg')
    pbar.set_postfix_str('baseline')
    bl_results = _run_mc(n_seeds, base_seed, n_workers)
    bl_checks, bl_pass, bl_total, bl_pct = _compute_checks(bl_results)
    pbar.set_postfix_str(f'baseline {bl_pct:.1f}%')
    pbar.update(1)

    output = {
        'params': SENSITIVITY_PARAMS,
        'multipliers': SENSITIVITY_MULTIPLIERS,
        'n_seeds': n_seeds,
        'check_names': CHECK_NAMES,
        'baseline_pass_rate': bl_pct,
        'baseline_per_check': bl_checks,
        'results': {},
    }

    for param in SENSITIVITY_PARAMS:
        nominal = getattr(_cfg, param)
        pd = {'nominal': nominal if not isinstance(nominal, np.generic) else nominal.item(),
              'rates': {}, 'per_check': {}}

        for mult in SENSITIVITY_MULTIPLIERS:
            if mult == 1.0:
                pd['rates']['1.0'] = bl_pct
                pd['per_check']['1.0'] = bl_checks
                continue

            value = nominal * mult
            if isinstance(nominal, int):
                value = max(1, round(value))

            pbar.set_postfix_str(f'{param} ×{mult}')
            originals = _apply_overrides({param: value})
            results = _run_mc(n_seeds, base_seed, n_workers)
            checks, passes, total, pct = _compute_checks(results)
            _restore_overrides(originals)

            pd['rates'][str(mult)] = pct
            pd['per_check'][str(mult)] = checks
            pbar.set_postfix_str(f'{param} ×{mult} → {pct:.1f}%')
            pbar.update(1)

        output['results'][param] = pd

    pbar.close()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {output_path}")
    return output


# ══════════════════════════════════════════════════════════════════════
#  Ablation study
# ══════════════════════════════════════════════════════════════════════

def run_ablation(n_seeds=50, base_seed=42, n_workers=None,
                 output_path='results/ablation.json'):
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    output = {
        'n_seeds': n_seeds,
        'check_names': CHECK_NAMES,
        'ablations': {},
    }

    pbar = tqdm(ABLATION_DEFS.items(), total=len(ABLATION_DEFS),
                desc='Ablation', unit='cfg')

    for name, overrides in pbar:
        pbar.set_postfix_str(name)

        config_ov = {k: v for k, v in overrides.items() if not k.startswith('_')}
        special = overrides.get('_special')

        create_fn = None
        if special == 'abs_edges':
            create_fn = _create_sim_abs_edges
        elif special == 'no_smc':
            create_fn = _create_sim_no_smc

        originals = _apply_overrides(config_ov) if config_ov else {}
        results = _run_mc(n_seeds, base_seed, n_workers, create_sim_fn=create_fn)
        if originals:
            _restore_overrides(originals)

        checks, passes, total, pct = _compute_checks(results)
        output['ablations'][name] = {'pass_rate': pct, 'per_check': checks}
        pbar.set_postfix_str(f'{name} → {pct:.1f}%')

    pbar.close()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {output_path}")
    return output


# ══════════════════════════════════════════════════════════════════════
#  CLI entry point: python -m bmc.validation [sensitivity|ablation|all]
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    t_start = time.time()

    if mode in ('sensitivity', 'all'):
        run_sensitivity()
    if mode in ('ablation', 'all'):
        run_ablation()

    print(f"\nTotal time: {time.time()-t_start:.0f}s")
