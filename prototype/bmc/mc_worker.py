"""Monte Carlo worker for BMC simulations (multiprocessing).
Uses module-level _ENGINE dict set by init_worker() before fork."""

import numpy as np
from bmc.config import N_CRIT, SPACED_REP_MIN_GAP as _SRMG

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

    # Compare WD only for shared neighbors (those involved in transfer)
    shared_nbs = set(hub_neighbors[:5])
    wd_hub_shared = sum(abs(sim.edge_weights.get((top_hub, nb), 0)) for nb in shared_nbs)
    wd_alt_shared = sum(abs(sim.edge_weights.get((alt_name, nb), 0)) for nb in shared_nbs)
    result['S6_ec_shift'] = int(wd_alt_shared > wd_hub_shared)

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

    # ── S9: SIT (persistent curiosity + false closure) ──
    sim9 = create_simulator(seed=seed)

    # Direct MoL SIT contribution (independent of cluster name)
    def _mol_sit(sim_):
        mol = 'Meaning_of_life'
        nbs = list(sim_.G.neighbors(mol))
        if not nbs:
            return 0.0
        relevance = np.mean([abs(sim_.get_weight(mol, nb)) for nb in nbs])
        cl = sim_.closure.get(mol, 0.0)
        cluster = sim_.G.nodes[mol].get('cluster', '')
        cluster_memes = [m for m in sim_.meme_nodes
                         if sim_.G.nodes[m].get('cluster') == cluster]
        cluster_cent = np.mean([sim_.centrality.get(m, 0) for m in cluster_memes])
        return relevance * (1 - cl) * cluster_cent

    # Phase 1: Run with open memes active, measure SEEKING boost
    sim9.run(30)
    seek_with_sit = np.mean([h['activations'].get('SEEKING', 0)
                             for h in sim9.history[-10:]])
    sit_initial = np.mean([h.get('sit_total', 0) for h in sim9.history[-10:]])
    sit_mol_pre = _mol_sit(sim9)

    # Phase 2: False closure on MoL's current cluster
    fc_cluster = sim9.G.nodes['Meaning_of_life'].get('cluster', 'beliefs')
    sim9.false_closure('Meaning_of_life', 'God_exists', fc_cluster)
    sim9.run(30)
    sit_mol_post = _mol_sit(sim9)
    seek_after_fc = np.mean([h['activations'].get('SEEKING', 0)
                             for h in sim9.history[-10:]])

    result['S9_sit_present'] = int(sit_initial > 0.0001)
    result['S9_fc_reduces_sit'] = int(sit_mol_post < sit_mol_pre)
    result['S9_seek_drops_after_fc'] = int(seek_after_fc < seek_with_sit)

    # ── S10: SMC + CL metric ──
    sim10 = create_simulator(seed=seed)
    sim10.run(20, {t: {'SEEKING': 0.5, 'Core_identity': 0.5} for t in range(20)})
    cl_values = [h.get('cl', 0) for h in sim10.history]
    a_smc_values = [h.get('a_smc', 0) for h in sim10.history]
    result['S10_cl_max'] = max(cl_values) if cl_values else 0
    result['S10_a_smc_mean'] = np.mean(a_smc_values) if a_smc_values else 0
    result['S10_cl_positive'] = int(max(cl_values) > 0.01)

    # ── S11: Sign inversion (bifurcation) ──
    sim11 = create_simulator(seed=seed)
    sim11.run(30)
    total_inversions = sum(len(h.get('inversions', [])) for h in sim11.history)
    result['S11_inversions'] = total_inversions
    result['S11_has_inversions'] = int(total_inversions > 0)

    # ── S12: Action loop (focused stimulation → closure increase) ──
    sim12 = create_simulator(seed=seed)
    for gap in list(sim12.open_memes):
        if gap != 'Career_purpose_question':
            sim12.archived_gaps.add(gap)
    sim12.run(80, {t: {'SEEKING': 0.8, 'Core_identity': 0.7, 'Career_growth': 0.8}
                   for t in range(80)})
    final_closure = sim12.closure.get('Career_purpose_question', 0)
    n_actions = sum(len([e for e in h.get('action_events', []) if e[0] == 'action'])
                    for h in sim12.history)
    n_aha = sum(len([e for e in h.get('action_events', []) if e[0] == 'aha'])
                for h in sim12.history)
    result['S12_closure'] = final_closure
    result['S12_n_actions'] = n_actions
    result['S12_closure_increased'] = int(final_closure > 0.3)  # initial=0.3
    result['S12_has_aha'] = int(n_aha > 0)

    # ── S13: SIT persistence (open meme edge retention) ──
    sim13 = create_simulator(seed=seed)
    # Focus on one open meme
    for gap in list(sim13.open_memes):
        if gap != 'Meaning_of_life':
            sim13.archived_gaps.add(gap)
    # Record initial edge weights
    mol_edges_init = {}
    for nb in sim13.G.neighbors('Meaning_of_life'):
        if nb in sim13.utility_set:
            continue
        w = abs(sim13.get_weight('Meaning_of_life', nb))
        if w > 0.1:
            mol_edges_init[nb] = w
    # Phase 1: mild SEEKING stimulation
    sim13.run(15, {t: {'SEEKING': 0.3} for t in range(15)})
    # Phase 2: pure decay
    sim13.run(45)
    # Measure edge retention
    if mol_edges_init:
        mol_retention = np.mean([abs(sim13.get_weight('Meaning_of_life', nb)) / mol_edges_init[nb]
                                 for nb in mol_edges_init])
    else:
        mol_retention = 0.0
    result['S13_edge_retention'] = mol_retention
    result['S13_sit_persistence'] = int(mol_retention > 0.8)

    # ── S15: Rumination stress-test (limiter archives unproductive gaps) ──
    sim15 = create_simulator(seed=seed)
    # Mild stimulation — closure progress < RUMINATION_THETA_MIN
    sim15.run(100, {t: {'SEEKING': 0.2} for t in range(100)})
    n_archived = len(sim15.archived_gaps)
    result['S15_n_archived'] = n_archived
    result['S15_rumination_archival'] = int(n_archived > 0)

    # ── S14: Q-dynamics cluster reorganization ──
    sim14 = create_simulator(seed=seed)
    n_clusters_start = len(set(sim14.G.nodes[m].get('cluster') for m in sim14.meme_nodes))
    # Dissonance pressure: conflicting utilities activate memes across clusters
    stim14 = {'FEAR': 0.4, 'DISGUST': 0.4, 'RAGE': 0.3, 'SEEKING': 0.3}
    sim14.run(60, {t: stim14 for t in range(60)})
    n_clusters_end = len(set(sim14.G.nodes[m].get('cluster') for m in sim14.meme_nodes))
    cluster_delta = n_clusters_start - n_clusters_end
    result['S14_cluster_delta'] = cluster_delta
    result['S14_q_reorganization'] = int(cluster_delta >= 2)

    # ── S16: Spaced repetition edge retention ──
    sim16 = create_simulator(seed=seed)
    # Pick 5 knowledge-cluster memes
    know_memes = [m for m in sim16.meme_nodes
                  if sim16.G.nodes[m].get('cluster') == 'knowledge'][:5]
    know_set = set(know_memes)
    stim16 = {m: 0.5 for m in know_memes}
    stim16['SEEKING'] = 0.4
    # 3 spaced bursts: 10-step stimulation, 15-step rest
    for _ in range(3):
        sim16.run(10, {t: stim16 for t in range(10)})
        sim16.run(15)
    # Record pre-decay weight
    pre_weights = []
    for m1 in know_memes:
        for m2 in sim16.G.neighbors(m1):
            if m2 in know_set and m2 > m1:
                pre_weights.append(abs(sim16.get_weight(m1, m2)))
    w_pre = np.mean(pre_weights) if pre_weights else 0
    # Decay period
    sim16.run(30)
    # Post-decay weight
    post_weights = []
    for m1 in know_memes:
        for m2 in sim16.G.neighbors(m1):
            if m2 in know_set and m2 > m1:
                post_weights.append(abs(sim16.get_weight(m1, m2)))
    w_post = np.mean(post_weights) if post_weights else 0
    retention16 = w_post / max(w_pre, 1e-6)
    result['S16_retention'] = retention16
    result['S16_spaced_rep'] = int(retention16 > 0.95)

    # ── S17: Utility inertia smoothing ──
    sim17 = create_simulator(seed=seed)
    sim17.run(10)  # warm up
    # Alternating FEAR/CARE every 5 steps
    for t in range(60):
        if (t // 5) % 2 == 0:
            sim17.step({'FEAR': 0.5, 'CARE': -0.1})
        else:
            sim17.step({'CARE': 0.5, 'FEAR': -0.1})
    fear_acts = [h['activations'].get('FEAR', 0) for h in sim17.history[10:]]
    volatility = np.std(np.diff(fear_acts))
    result['S17_volatility'] = volatility
    result['S17_utility_inertia'] = int(volatility < 0.22)

    # ── S18: κ-transition (spaced repetition → LTM) ──
    sim18 = create_simulator(seed=seed)
    sim18.compact_history = True
    # Introduce a fresh meme (starts at κ=0)
    sim18.run(10)
    acc18, _ = sim18.introduce_meme(
        'Kappa_test_meme', 'knowledge',
        [('Science_trust', 0.6, 'intra'), ('Rationalism', 0.5, 'cross'),
         ('Knowledge_value', 0.5, 'cross')],
        {'SEEKING': 0.5, 'FEAR': -0.1, 'RAGE': 0.0,
         'LUST': 0.0, 'CARE': 0.1, 'PANIC_GRIEF': 0.0,
         'PLAY': 0.1, 'DISGUST': -0.1},
        initial_activation=0.4, threshold=-1.0)
    # Spaced repetition: N_CRIT+1 bursts with gaps ≥ SPACED_REP_MIN_GAP
    for burst in range(N_CRIT + 1):
        sim18.run(_SRMG + 2, {t: {'Kappa_test_meme': 0.6, 'Science_trust': 0.5,
                                    'SEEKING': 0.4} for t in range(_SRMG + 2)})
        sim18.run(5)  # rest
    kappa_final = sim18.nodes.get('Kappa_test_meme')
    result['S18_kappa'] = kappa_final.kappa if kappa_final else -1
    result['S18_n_react'] = kappa_final.n_react if kappa_final else 0
    result['S18_kappa_transition'] = int(kappa_final.kappa == 2) if kappa_final else 0

    # ── S19: κ-dependent decay (sensory vs STM vs LTM) ──
    sim19 = create_simulator(seed=seed)
    sim19.compact_history = True
    # Pick 3 memes, force different κ levels
    k_memes = ['Science_trust', 'Rationalism', 'Knowledge_value']
    # Ensure all exist
    if all(m in sim19.nodes for m in k_memes):
        # Force κ levels (set i_passed=False for κ=0 to prevent auto-promotion)
        sim19.nodes[k_memes[0]].kappa = 0  # sensory
        sim19.nodes[k_memes[0]].i_passed = False
        sim19.nodes[k_memes[1]].kappa = 1  # STM
        sim19.nodes[k_memes[2]].kappa = 2  # LTM
        # Set equal initial activation
        for m in k_memes:
            sim19.nodes[m].activation = 0.8
        # Record initial
        a_init = {m: sim19.nodes[m].activation for m in k_memes}
        # Run 50 steps without stimuli (pure decay)
        sim19.run(50)
        a_final = {m: sim19.nodes[m].activation for m in k_memes}
        result['S19_a_k0'] = a_final[k_memes[0]]
        result['S19_a_k1'] = a_final[k_memes[1]]
        result['S19_a_k2'] = a_final[k_memes[2]]
        # Pass: κ=0 decays much faster than κ=2, κ=1 in between
        result['S19_kappa_decay'] = int(
            a_final[k_memes[0]] < 0.5 * a_final[k_memes[2]]
            and a_final[k_memes[1]] < a_final[k_memes[2]]
        )
    else:
        result['S19_a_k0'] = 0
        result['S19_a_k1'] = 0
        result['S19_a_k2'] = 0
        result['S19_kappa_decay'] = 0

    # ── Structural balance ──
    sb = detect_structural_balance(sim.G, sim.edge_weights)
    result['balance_ratio'] = sb['ratio']

    return result
