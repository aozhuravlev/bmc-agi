"""BMCSimulator — Biomemetic Complex dynamics engine."""
import numpy as np
import networkx as nx

from .config import (
    THETA_HIGH, THETA_LOW, LAMBDA_DECAY, SIGMOID_GAIN,
    LAMBDA_EDGE_POS, LAMBDA_EDGE_NEG, ALPHA_CENTRALITY,
    DELTA_W_REACTIVATE, CONSOLIDATION_TRACE_LEVEL, SAVINGS_MULTIPLIER,
    FIDELITY_GAMMA, FIDELITY_LAMBDA, FIDELITY_BETA,
    ALPHA_RECRUIT, T_SEEK,
    CONSOLIDATION_PRUNE_THRESHOLD, SHY_DOWNSCALE_FACTOR, BLEND_THRESHOLD,
    UTILITY_SPREAD_REDIRECT, UTILITY_SPREAD_SUPPRESS, UTILITY_SPREAD_INTERPRET,
    UTILITY_INERTIA,
    WM_SIZE, SOFTMAX_TEMP,
    FATIGUE_RATE, RECOVERY_RATE, FATIGUE_SUPPRESSION,
)


class BMCSimulator:
    """Biomemetic Complex dynamics simulator with signed edges, Panksepp systems,
    SEEKING metasystem, fidelity, ambivalence, and DISGUST I-layer."""

    def __init__(self, G, incompatibility):
        self.G = G.copy()
        self.incompatibility = incompatibility
        self.meme_nodes = [n for n in G.nodes() if G.nodes[n].get('layer') == 'memetic']
        self.utility_nodes = [n for n in G.nodes() if G.nodes[n].get('layer') == 'utility']
        self.utility_set = set(self.utility_nodes)

        # State
        self.activations = {n: G.nodes[n].get('activation', 0.1) for n in G.nodes()}
        self.fatigue = 0.0

        # Fidelity and age (meme-only attributes)
        self.fidelity = {n: G.nodes[n].get('fidelity', 0.5) for n in self.meme_nodes}
        self.node_age = {n: G.nodes[n].get('age', 0) for n in self.meme_nodes}

        # Edge weights — signed, w ∈ [-1, +1]
        self.edge_weights = {}
        for u, v in G.edges():
            w = G.edges[u, v].get('weight', 0.5)
            self.edge_weights[(u, v)] = w
            self.edge_weights[(v, u)] = w

        # Precompute centrality for differential decay
        self.centrality = nx.degree_centrality(G)

        # History
        self.history = []
        self.prev_wm = set()
        self._prev_wm_activations = None

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-SIGMOID_GAIN * (x - 0.5)))

    def get_weight(self, u, v):
        return self.edge_weights.get((u, v), 0.0)

    def meme_state(self, node):
        a = self.activations[node]
        if a > THETA_HIGH:
            return 'active'
        elif a < THETA_LOW:
            return 'sleeping'
        else:
            return 'decorative'

    def step(self, stimuli=None):
        """One simulation step."""
        if stimuli is None:
            stimuli = {}

        new_act = {}

        # ── (a) Utility activation: SEEKING as metasystem ──
        non_seeking_acts = {}
        for u in self.utility_nodes:
            if u == 'SEEKING':
                continue
            base = self.G.nodes[u].get('base_activation', 0.4)
            stim = stimuli.get(u, 0.0)
            feedback = 0.0
            for nb in self.G.neighbors(u):
                if nb not in self.utility_set:
                    w = self.get_weight(u, nb)
                    etype = self.G.edges[u, nb].get('etype', 'redirect')
                    if etype == 'suppress':
                        feedback -= 0.1 * abs(w) * self.activations[nb]
                    else:
                        feedback += 0.05 * w * self.activations[nb]
            target = np.clip(base + stim + feedback, 0.0, 1.0)
            non_seeking_acts[u] = UTILITY_INERTIA * self.activations[u] + (1 - UTILITY_INERTIA) * target

        # SEEKING = T_SEEK * base + Σ α_s * a_s (recruitment by other systems)
        if 'SEEKING' in self.activations:
            seek_base = self.G.nodes['SEEKING'].get('base_activation', 0.45)
            seek_stim = stimuli.get('SEEKING', 0.0)
            recruit_sum = sum(
                ALPHA_RECRUIT.get(s, 0.0) * non_seeking_acts.get(s, self.activations.get(s, 0))
                for s in ALPHA_RECRUIT
            )
            seek_target = np.clip(T_SEEK * seek_base + recruit_sum + seek_stim, 0.0, 1.0)
            seek_feedback = 0.0
            for nb in self.G.neighbors('SEEKING'):
                if nb not in self.utility_set:
                    w = self.get_weight('SEEKING', nb)
                    seek_feedback += 0.05 * w * self.activations[nb]
            seek_target = np.clip(seek_target + seek_feedback, 0.0, 1.0)
            non_seeking_acts['SEEKING'] = UTILITY_INERTIA * self.activations['SEEKING'] + \
                                          (1 - UTILITY_INERTIA) * seek_target

        for u in self.utility_nodes:
            new_act[u] = non_seeking_acts.get(u, self.activations[u])

        # DISGUST special: activates more when incompatible input detected
        if 'DISGUST' in new_act:
            disgust_boost = 0.0
            for m in self.meme_nodes:
                neg_edges = sum(1 for nb in self.G.neighbors(m)
                               if nb in set(self.meme_nodes)
                               and self.get_weight(m, nb) < 0
                               and self.activations[m] > THETA_HIGH
                               and self.activations[nb] > THETA_HIGH)
                if neg_edges > 0:
                    disgust_boost += 0.02 * neg_edges
            new_act['DISGUST'] = np.clip(new_act['DISGUST'] + disgust_boost, 0.0, 1.0)

        # ── (b) Spreading activation for memes (signed weights) ──
        fatigue_factor = max(0.0, 1.0 - FATIGUE_SUPPRESSION * self.fatigue)
        for m in self.meme_nodes:
            incoming_meme = 0.0
            incoming_utility = 0.0
            n_meme_neighbors = 0
            for nb in self.G.neighbors(m):
                w = self.get_weight(m, nb)
                if nb in self.utility_set:
                    etype = self.G.edges[m, nb].get('etype', 'redirect')
                    if etype == 'redirect':
                        incoming_utility += UTILITY_SPREAD_REDIRECT * w * self.activations[nb]
                    elif etype == 'suppress':
                        incoming_utility -= UTILITY_SPREAD_SUPPRESS * abs(w) * self.activations[nb]
                    else:  # interpret
                        incoming_utility += UTILITY_SPREAD_INTERPRET * w * self.activations[nb]
                else:
                    incoming_meme += w * self.activations[nb]
                    n_meme_neighbors += 1
            if n_meme_neighbors > 0:
                incoming_meme /= n_meme_neighbors
            stim = stimuli.get(m, 0.0)
            raw = (1 - LAMBDA_DECAY) * self.activations[m] + incoming_meme + incoming_utility + stim
            new_act[m] = self.sigmoid(raw) * fatigue_factor
        pre_wm_act = dict(new_act)

        # ── (c) Lateral inhibition → Working Memory selection ──
        meme_acts = np.array([new_act[m] for m in self.meme_nodes])
        if meme_acts.max() > 0:
            exp_a = np.exp(meme_acts / SOFTMAX_TEMP)
            softmax_a = exp_a / exp_a.sum()
            wm_indices = np.argsort(softmax_a)[-WM_SIZE:]
            wm_set = set(self.meme_nodes[i] for i in wm_indices)
            for i, m in enumerate(self.meme_nodes):
                if i not in wm_indices:
                    new_act[m] *= 0.85
        else:
            wm_set = set()

        # ── (d) Asymmetric edge decay ──
        for (u, v), w in list(self.edge_weights.items()):
            cu = self.centrality.get(u, 0.01)
            cv = self.centrality.get(v, 0.01)
            lam_base = LAMBDA_EDGE_POS if w >= 0 else LAMBDA_EDGE_NEG
            lam = lam_base / (1 + ALPHA_CENTRALITY * cu * cv)
            new_w = w * np.exp(-lam)
            a_u = pre_wm_act.get(u, self.activations.get(u, 0))
            a_v = pre_wm_act.get(v, self.activations.get(v, 0))
            if a_u >= THETA_HIGH and a_v >= THETA_HIGH:
                is_dormant = abs(new_w) <= CONSOLIDATION_TRACE_LEVEL
                boost = DELTA_W_REACTIVATE * (SAVINGS_MULTIPLIER if is_dormant else 1.0)
                if w >= 0:
                    new_w = min(1.0, new_w + boost)
                else:
                    new_w = max(-1.0, new_w - boost * 0.5)
            self.edge_weights[(u, v)] = np.clip(new_w, -1.0, 1.0)

        # ── (e) Fatigue ──
        n_active = sum(1 for m in self.meme_nodes if new_act[m] > THETA_HIGH)
        activity_level = min(n_active / (3 * WM_SIZE), 1.0)
        self.fatigue = np.clip(
            self.fatigue + FATIGUE_RATE * activity_level - RECOVERY_RATE * (1.0 - activity_level),
            0.0, 1.0
        )

        # ── (f) Fidelity update ──
        k_max = max((self.G.degree(m) for m in self.meme_nodes), default=1)
        k_max = max(k_max, 1)
        for m in self.meme_nodes:
            k_m = self.G.degree(m)
            self.node_age[m] = self.node_age.get(m, 0) + 1
            age = self.node_age[m]
            self.fidelity[m] = ((k_m / k_max) ** FIDELITY_GAMMA *
                                np.exp(-FIDELITY_LAMBDA * 1) *
                                (1 - np.exp(-FIDELITY_BETA * age)))

        # ── Apply new activations ──
        self.activations = new_act

        # ── (g) Ambivalence metric ──
        ambivalence = {}
        for m in self.meme_nodes:
            neighbor_weights = [self.get_weight(m, nb) for nb in self.G.neighbors(m)
                                if nb not in self.utility_set]
            if len(neighbor_weights) > 1:
                ambivalence[m] = float(np.std(neighbor_weights))
            else:
                ambivalence[m] = 0.0
        mean_ambivalence = np.mean(list(ambivalence.values())) if ambivalence else 0.0

        # ── Compute metrics ──
        wm_meme_acts = [self.activations[m] for m in wm_set] if wm_set else [0.0]
        A_meme = np.mean(wm_meme_acts)
        A_utility = np.mean([self.activations[u] for u in self.utility_nodes])
        balance = A_meme / max(A_utility, 1e-6)

        conflict = 0.0
        for u in self.utility_nodes:
            for m in self.meme_nodes:
                inc = self.incompatibility.get(u, {}).get(m, 0.0)
                if inc > 0:
                    conflict += abs(self.activations[u] - self.activations[m]) * inc

        dissonance = 0.0
        for u in self.utility_nodes:
            for m in self.meme_nodes:
                inc = self.incompatibility.get(u, {}).get(m, 0.0)
                if inc > 0:
                    dissonance += min(self.activations[u], self.activations[m]) * inc

        tension = 0.0
        for u in self.utility_nodes:
            a_u = self.activations[u]
            for nb in self.G.neighbors(u):
                if nb not in self.utility_set:
                    a_m = self.activations[nb]
                    w = self.get_weight(u, nb)
                    etype = self.G.edges[u, nb].get('etype', 'redirect')
                    if etype == 'suppress':
                        tension += abs(w) * a_u * a_m
                    else:
                        tension += abs(w) * max(0, a_u - a_m)

        # Set-based stability
        if self.prev_wm:
            sym_diff = len(wm_set.symmetric_difference(self.prev_wm))
            stability = 1.0 - sym_diff / max(len(wm_set), 1)
        else:
            stability = 1.0

        # Weighted WM stability (cosine similarity)
        curr_wm_activations = {m: self.activations.get(m, 0.0) for m in wm_set}
        if self._prev_wm_activations is not None:
            all_keys = set(curr_wm_activations) | set(self._prev_wm_activations)
            vec_curr = np.array([curr_wm_activations.get(k, 0.0) for k in all_keys])
            vec_prev = np.array([self._prev_wm_activations.get(k, 0.0) for k in all_keys])
            norm_c = np.linalg.norm(vec_curr)
            norm_p = np.linalg.norm(vec_prev)
            if norm_c > 1e-12 and norm_p > 1e-12:
                wm_stability_weighted = float(np.dot(vec_curr, vec_prev) / (norm_c * norm_p))
            else:
                wm_stability_weighted = 0.0
        else:
            wm_stability_weighted = 1.0

        self.prev_wm = wm_set
        self._prev_wm_activations = curr_wm_activations

        states = {'active': 0, 'decorative': 0, 'sleeping': 0}
        for m in self.meme_nodes:
            states[self.meme_state(m)] += 1

        self.history.append({
            'activations': dict(self.activations),
            'edge_weights': dict(self.edge_weights),
            'balance': balance,
            'conflict': conflict,
            'dissonance': dissonance,
            'tension': tension,
            'stability': stability,
            'wm_stability_weighted': wm_stability_weighted,
            'fatigue': self.fatigue,
            'A_meme': A_meme,
            'A_utility': A_utility,
            'wm': set(wm_set),
            'states': dict(states),
            'mean_ambivalence': mean_ambivalence,
        })

    def run(self, n_steps, stimuli_schedule=None):
        """Run simulation for n_steps."""
        if stimuli_schedule is None:
            stimuli_schedule = {}
        for t in range(n_steps):
            stim = stimuli_schedule.get(t, {})
            self.step(stim)

    def introduce_meme(self, name, cluster, connections, compatibility,
                       initial_activation=0.3, threshold=0.0):
        """
        Introduce a foreign meme. Rejected memes get negative edges (not ignored).
        DISGUST activates on rejection.
        """
        score = 0.0
        for node, compat_val in compatibility.items():
            if node in self.activations:
                score += self.activations[node] * compat_val

        accepted = score >= threshold

        if accepted:
            self.G.add_node(name, layer='memetic', cluster=cluster,
                            activation=initial_activation,
                            fidelity=0.3, age=0)
            self.meme_nodes.append(name)
            self.activations[name] = initial_activation
            self.fidelity[name] = 0.3
            self.node_age[name] = 0

            for target, weight, etype in connections:
                if target in self.G.nodes():
                    self.G.add_edge(name, target, weight=weight, etype=etype)
                    self.edge_weights[(name, target)] = weight
                    self.edge_weights[(target, name)] = weight

            for u in self.utility_nodes:
                if u not in self.incompatibility:
                    self.incompatibility[u] = {}
                compat_val = compatibility.get(u, 0.0)
                self.incompatibility[u][name] = max(0.0, -compat_val)

            self.centrality = nx.degree_centrality(self.G)
        else:
            # Rejection: add meme with NEGATIVE edges (sleeper potential)
            self.G.add_node(name, layer='memetic', cluster=cluster,
                            activation=0.05, fidelity=0.1, age=0)
            self.meme_nodes.append(name)
            self.activations[name] = 0.05
            self.fidelity[name] = 0.1
            self.node_age[name] = 0

            for target, weight, etype in connections:
                if target in self.G.nodes():
                    neg_w = -abs(weight) * 0.5
                    self.G.add_edge(name, target, weight=neg_w, etype='cross_neg')
                    self.edge_weights[(name, target)] = neg_w
                    self.edge_weights[(target, name)] = neg_w

            for u in self.utility_nodes:
                if u not in self.incompatibility:
                    self.incompatibility[u] = {}
                compat_val = compatibility.get(u, 0.0)
                self.incompatibility[u][name] = max(0.0, -compat_val)

            # DISGUST activates on rejection
            if 'DISGUST' in self.activations:
                self.activations['DISGUST'] = min(1.0,
                    self.activations['DISGUST'] + 0.3)
            # Also boost FEAR as defensive reaction
            if 'FEAR' in self.activations:
                self.activations['FEAR'] = min(1.0,
                    self.activations['FEAR'] + 0.15)

            self.centrality = nx.degree_centrality(self.G)

        return accepted, score

    def sleep_consolidation(self):
        """Sleep cycle: DECOMPOSE → CONNECT → BLEND → PRUNE → STRENGTHEN."""
        rng = np.random.RandomState(len(self.history))

        # 1. DECOMPOSE — reduce activations (entering sleep)
        for m in self.meme_nodes:
            self.activations[m] *= 0.3
        for u in self.utility_nodes:
            self.activations[u] *= 0.5

        # 2. CONNECT — strengthen edges between co-active pairs
        for m1 in self.meme_nodes:
            for m2 in self.meme_nodes:
                if m1 >= m2:
                    continue
                key = (m1, m2)
                if key in self.edge_weights:
                    a1 = self.activations.get(m1, 0)
                    a2 = self.activations.get(m2, 0)
                    if a1 > 0.05 and a2 > 0.05:
                        w = self.edge_weights[key]
                        boost = 0.05 * min(a1, a2)
                        self.edge_weights[key] = np.clip(w + boost, -1.0, 1.0)
                        self.edge_weights[(m2, m1)] = self.edge_weights[key]

        # 3. BLEND — create new node from two memes of different clusters
        blend_candidates = []
        for m1 in self.meme_nodes:
            for m2 in self.meme_nodes:
                if m1 >= m2:
                    continue
                c1 = self.G.nodes[m1].get('cluster', '')
                c2 = self.G.nodes[m2].get('cluster', '')
                if c1 != c2:
                    key = (m1, m2)
                    w = self.edge_weights.get(key, 0)
                    if w > BLEND_THRESHOLD:
                        blend_candidates.append((m1, m2, w))

        blend_created = []
        if blend_candidates:
            blend_candidates.sort(key=lambda x: x[2], reverse=True)
            m1, m2, w = blend_candidates[0]
            blend_name = f'BLEND_{m1[:8]}_{m2[:8]}'
            if blend_name not in self.G.nodes():
                c1 = self.G.nodes[m1].get('cluster', 'beliefs')
                self.G.add_node(blend_name, layer='memetic', cluster=c1,
                                activation=0.2, fidelity=0.3, age=0, is_blend=True)
                self.meme_nodes.append(blend_name)
                self.activations[blend_name] = 0.2
                self.fidelity[blend_name] = 0.3
                self.node_age[blend_name] = 0

                # Inherit subset of edges from parents
                for parent in [m1, m2]:
                    for nb in list(self.G.neighbors(parent)):
                        if nb != blend_name and nb not in self.utility_set:
                            pw = self.get_weight(parent, nb)
                            if abs(pw) > 0.2 and rng.random() < 0.5:
                                inh_w = pw * 0.6
                                self.G.add_edge(blend_name, nb, weight=inh_w, etype='cross')
                                self.edge_weights[(blend_name, nb)] = inh_w
                                self.edge_weights[(nb, blend_name)] = inh_w

                # Connect to parents
                self.G.add_edge(blend_name, m1, weight=0.6, etype='cross')
                self.edge_weights[(blend_name, m1)] = 0.6
                self.edge_weights[(m1, blend_name)] = 0.6
                self.G.add_edge(blend_name, m2, weight=0.6, etype='cross')
                self.edge_weights[(blend_name, m2)] = 0.6
                self.edge_weights[(m2, blend_name)] = 0.6

                # Update incompatibility for new node
                for u in self.utility_nodes:
                    if u not in self.incompatibility:
                        self.incompatibility[u] = {}
                    avg_inc = (self.incompatibility.get(u, {}).get(m1, 0) +
                               self.incompatibility.get(u, {}).get(m2, 0)) / 2
                    self.incompatibility[u][blend_name] = avg_inc

                self.centrality = nx.degree_centrality(self.G)
                blend_created.append(blend_name)

        # 4a. SHY — proportional synaptic downscaling (Tononi & Cirelli 2003, 2014)
        for (u, v), w in list(self.edge_weights.items()):
            if u in self.utility_set or v in self.utility_set:
                continue
            sign = 1.0 if w >= 0 else -1.0
            self.edge_weights[(u, v)] = sign * abs(w) * (1.0 - SHY_DOWNSCALE_FACTOR)

        # 4b. PRUNE — reduce sub-threshold edges to trace level (not deletion)
        for (u, v), w in list(self.edge_weights.items()):
            if u in self.utility_set or v in self.utility_set:
                continue
            if abs(w) < CONSOLIDATION_PRUNE_THRESHOLD:
                sign = 1.0 if w >= 0 else -1.0
                self.edge_weights[(u, v)] = sign * CONSOLIDATION_TRACE_LEVEL

        # 5. STRENGTHEN — boost edges between high-fidelity nodes
        for m1 in self.meme_nodes:
            for m2 in self.meme_nodes:
                if m1 >= m2:
                    continue
                key = (m1, m2)
                if key in self.edge_weights:
                    f1 = self.fidelity.get(m1, 0.5)
                    f2 = self.fidelity.get(m2, 0.5)
                    if f1 > 0.5 and f2 > 0.5:
                        w = self.edge_weights[key]
                        boost = 0.03 * (f1 + f2)
                        self.edge_weights[key] = np.clip(w + boost, -1.0, 1.0)
                        self.edge_weights[(m2, m1)] = self.edge_weights[key]

        return blend_created

    def reset(self):
        """Reset to initial state."""
        for n in self.G.nodes():
            if self.G.nodes[n].get('layer') == 'utility':
                self.activations[n] = self.G.nodes[n].get('base_activation', 0.4)
            else:
                self.activations[n] = np.random.uniform(0.05, 0.25)
        self.fatigue = 0.0
        self.fidelity = {n: self.G.nodes[n].get('fidelity', 0.5) for n in self.meme_nodes}
        self.node_age = {n: self.G.nodes[n].get('age', 0) for n in self.meme_nodes}
        for u, v in self.G.edges():
            w = self.G.edges[u, v].get('weight', 0.5)
            self.edge_weights[(u, v)] = w
            self.edge_weights[(v, u)] = w
        self.history = []
        self.prev_wm = set()
        self._prev_wm_activations = None
