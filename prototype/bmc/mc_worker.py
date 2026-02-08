"""Monte Carlo worker for BMC simulations (multiprocessing).
Uses module-level _ENGINE dict set by init_worker() before fork."""

import numpy as np

_ENGINE = {}


def init_worker(engine_dict):
    """Called once per worker process to set engine globals."""
    global _ENGINE
    _ENGINE = engine_dict


def run_seed(seed):
    """Run all 8 scenarios (S1-S8) for one seed. Uses _ENGINE globals set via init_worker."""
    create_simulator = _ENGINE['create_simulator']
    detect_structural_balance = _ENGINE['detect_structural_balance']
    MEME_TO_CLUSTER = _ENGINE['MEME_TO_CLUSTER']

    result = {}

    # ── S1: Normal state (SEEKING stimulus) ──
    sim = create_simulator(seed=seed)
    schedule = {t: {'SEEKING': 0.3} for t in range(6)}
    sim.run(60, schedule)
    bals = [h['balance'] for h in sim.history]
    result['S1_peak_balance'] = max(bals)
    result['S1_peak_t'] = int(np.argmax(bals))
    result['mean_ambivalence'] = np.mean([h.get('mean_ambivalence', 0) for h in sim.history])

    # ── S2: Stress (all utility +0.4) ──
    sim = create_simulator(seed=seed)
    sim.run(20)
    stress_stim = {u: 0.4 for u in sim.utility_nodes}
    sim.run(30, {t: dict(stress_stim) for t in range(30)})
    sim.run(30)
    bals = [h['balance'] for h in sim.history]
    result['S2_min_balance'] = min(bals[20:50])
    result['S2_recovery_balance'] = np.mean(bals[-10:])

    # ── S3: Challenge to core beliefs (hub attack: Science_trust) ──
    sim = create_simulator(seed=seed)
    ch_sched = {}
    for t in range(20, 40):
        ch_sched[t] = {'Science_trust': -0.3, 'Rationalism': -0.2,
                       'FEAR': 0.25, 'DISGUST': 0.3}
    sim.run(70, ch_sched)
    hub_before = sim.history[19]['activations'].get('Science_trust', 0)
    hub_during = sim.history[30]['activations'].get('Science_trust', 0)
    drop_pct = (1 - hub_during / max(hub_before, 1e-6)) * 100
    result['S3_hub_drop_pct'] = drop_pct
    fear_before = sim.history[19]['activations'].get('FEAR', 0)
    fear_during = sim.history[30]['activations'].get('FEAR', 0)
    result['S3_fear_increase_pct'] = (fear_during / max(fear_before, 1e-6) - 1) * 100
    conf_base = np.mean([h['conflict'] for h in sim.history[15:20]])
    conf_attack = np.mean([h['conflict'] for h in sim.history[20:40]])
    result['S3_conflict_change'] = conf_attack - conf_base

    # ── S4: Fatigue (skills cluster memes) ──
    sim = create_simulator(seed=seed)
    active_memes = ['Tool_use', 'Cooking_skill', 'Navigation', 'Planning_ahead',
                    'Risk_assessment', 'Impulse_delay', 'Resource_management']
    intense = {}
    for t in range(80):
        stim = {m: 0.2 for m in active_memes if m in sim.activations}
        stim['RAGE'] = 0.2
        intense[t] = stim
    sim.run(100, intense)
    result['S4_balance_t20'] = sim.history[19]['balance']
    result['S4_balance_t80'] = sim.history[79]['balance']
    result['S4_peak_fatigue'] = max(h['fatigue'] for h in sim.history)

    # ── S5a: Compatible meme (Critical_thinking) ──
    sim5a = create_simulator(seed=seed)
    sim5a.run(15)
    acc_a, sc_a = sim5a.introduce_meme(
        'Critical_thinking', 'knowledge',
        [('Science_trust', 0.6, 'intra'), ('Worldview', 0.5, 'cross'),
         ('Moral_framework', 0.4, 'cross'), ('Philosophy_mind', 0.4, 'cross')],
        {'SEEKING': 0.8, 'FEAR': -0.1, 'RAGE': 0.3,
         'LUST': 0.0, 'CARE': 0.1, 'PANIC_GRIEF': -0.1,
         'PLAY': 0.1, 'DISGUST': -0.2},
        initial_activation=0.3, threshold=0.0)
    result['S5a_score'] = sc_a
    result['S5a_accepted'] = int(acc_a)

    # ── S5b: Incompatible meme (Flat_earth) ──
    sim5b = create_simulator(seed=seed)
    sim5b.run(15)
    acc_b, sc_b = sim5b.introduce_meme(
        'Flat_earth', 'beliefs',
        [('Science_trust', 0.3, 'cross'), ('Worldview', 0.3, 'cross'),
         ('Evolution_theory', 0.2, 'cross')],
        {'SEEKING': -0.3, 'FEAR': 0.1, 'RAGE': -0.2,
         'LUST': 0.0, 'CARE': -0.1, 'PANIC_GRIEF': 0.1,
         'PLAY': -0.1, 'DISGUST': -0.8},
        initial_activation=0.3, threshold=0.0)
    result['S5b_score'] = sc_b
    result['S5b_accepted'] = int(not acc_b)  # pass = rejection

    # ── S6: Hub displacement (top hub by degree) ──
    sim = create_simulator(seed=seed)
    sim.run(10)
    deg6 = {n: sim.G.degree(n) for n in sim.meme_nodes}
    top_hub = max(deg6, key=deg6.get)
    hub_neighbors = [n for n in sim.G.neighbors(top_hub) if n in sim.meme_nodes]

    meme_set6 = set(sim.meme_nodes)
    def _wd(sim_, node):
        return sum(abs(sim_.edge_weights.get((node, nb), 0))
                   for nb in sim_.G.neighbors(node) if nb in meme_set6)
    wd_hub_before = _wd(sim, top_hub)

    sim.run(15, {t: {top_hub: -0.4} for t in range(15)})
    alt_name = f'{top_hub}_Alternative'
    alt_conns = [(nb, 0.6, 'cross') for nb in hub_neighbors[:5]]
    sim.introduce_meme(alt_name, sim.G.nodes[top_hub].get('cluster', 'beliefs'),
                       alt_conns,
                       {'SEEKING': 0.5, 'FEAR': -0.1, 'RAGE': 0.4,
                        'LUST': 0.0, 'CARE': 0.2, 'PANIC_GRIEF': -0.1,
                        'PLAY': 0.2, 'DISGUST': -0.1},
                       initial_activation=0.5, threshold=-1.0)

    total_k = sum(deg6.values())
    beta = 0.15
    for t in range(40):
        sim.step({alt_name: 0.25, top_hub: -0.25})
        for nb in hub_neighbors:
            old_key = (top_hub, nb)
            new_key = (alt_name, nb)
            if old_key in sim.edge_weights and new_key in sim.edge_weights:
                k_old = deg6.get(top_hub, 1)
                k_new = sim.G.degree(alt_name)
                delta = beta * max(k_old - k_new, 0) / total_k
                old_w = sim.edge_weights[old_key]
                transfer = min(delta, abs(old_w) * 0.3)
                sim.edge_weights[old_key] = np.clip(old_w - transfer, -1.0, 1.0)
                sim.edge_weights[(nb, top_hub)] = sim.edge_weights[old_key]
                sim.edge_weights[new_key] = min(1.0, sim.edge_weights[new_key] + transfer)
                sim.edge_weights[(nb, alt_name)] = sim.edge_weights[new_key]
    sim.run(15)

    wd_hub_after = _wd(sim, top_hub)
    wd_alt_after = _wd(sim, alt_name)
    result['S6_ec_shift'] = int(wd_alt_after > wd_hub_after)

    # ── S7: Sleep consolidation + BLEND ──
    CONSOLIDATION_PRUNE_THRESHOLD = _ENGINE.get('CONSOLIDATION_PRUNE_THRESHOLD', 0.05)
    sim7 = create_simulator(seed=seed)
    sim7.run(30, {t: {'SEEKING': 0.3, 'Knowledge_value': 0.2} for t in range(30)})
    n_edges_pre = sum(1 for k, w in sim7.edge_weights.items()
                      if abs(w) > CONSOLIDATION_PRUNE_THRESHOLD)
    blends = sim7.sleep_consolidation()
    n_edges_post = sum(1 for k, w in sim7.edge_weights.items()
                       if abs(w) > CONSOLIDATION_PRUNE_THRESHOLD)
    result['S7_blend_count'] = len(blends)
    result['S7_edges_pruned'] = (n_edges_pre - n_edges_post) // 2

    # ── S8: Sleeper effect ──
    sim8 = create_simulator(seed=seed)
    sim8.run(10)
    acc8, _ = sim8.introduce_meme(
        'Alt_med_mc', 'beliefs',
        [('Science_trust', 0.4, 'cross'), ('Moral_framework', 0.3, 'cross')],
        {'SEEKING': -0.2, 'FEAR': 0.2, 'RAGE': -0.3, 'LUST': 0.0,
         'CARE': 0.3, 'PANIC_GRIEF': 0.2, 'PLAY': 0.0, 'DISGUST': -0.6},
        initial_activation=0.3, threshold=0.0)
    w_initial = sim8.get_weight('Alt_med_mc', 'Science_trust')
    sim8.run(40)  # decay period
    sim8.run(20, {t: {'CARE': 0.3, 'SEEKING': 0.2, 'Alt_med_mc': 0.15} for t in range(20)})
    w_final = sim8.get_weight('Alt_med_mc', 'Science_trust')
    result['S8_sign_flip'] = int(w_final > w_initial)

    # ── Structural balance ──
    sb = detect_structural_balance(sim.G, sim.edge_weights)
    result['balance_ratio'] = sb['ratio']

    return result
