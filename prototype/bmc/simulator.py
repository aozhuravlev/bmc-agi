"""BMCSimulator — Biomemetic Complex dynamics engine."""
import numpy as np
import networkx as nx
from dataclasses import dataclass

from .config import (
    THETA_HIGH, THETA_LOW, LAMBDA_DECAY, SIGMOID_GAIN,
    LAMBDA_EDGE_POS, LAMBDA_EDGE_NEG, ALPHA_CENTRALITY,
    DELTA_W_REACTIVATE, CONSOLIDATION_TRACE_LEVEL, SAVINGS_MULTIPLIER,
    SPACED_REP_FACTOR, SPACED_REP_MIN_GAP,
    FIDELITY_GAMMA, FIDELITY_LAMBDA, FIDELITY_BETA,
    ALPHA_RECRUIT, T_SEEK,
    CONSOLIDATION_PRUNE_THRESHOLD, SHY_DOWNSCALE_FACTOR, BLEND_THRESHOLD,
    UTILITY_SPREAD_REDIRECT, UTILITY_SPREAD_SUPPRESS, UTILITY_SPREAD_INTERPRET,
    UTILITY_INERTIA,
    WM_SIZE, SOFTMAX_TEMP, WM_DELTA_SALIENCE_BOOST, WM_HABITUATION_RATE, WM_DELTA_THRESHOLD,
    FATIGUE_RATE, RECOVERY_RATE, FATIGUE_SUPPRESSION,
    GAMMA_SIT, SIT_LP_WINDOW,
    OPEN_MEME_DECAY_RESISTANCE, OPEN_MEME_PULSE_AMP, OPEN_MEME_PULSE_PERIOD,
    FALSE_CLOSURE_VALIDITY,
    SMC_RECURSION_WEIGHT,
    CL_BALANCE_OPT, CL_BALANCE_SIGMA, CL_SIGMA_SW_SAMPLES, CL_SIGMA_SW_SCALE,
    SIGN_INVERSION_ENABLED,
    Q_CHECK_INTERVAL, Q_CRIT, Q_MERGE, Q_ALPHA, Q_BETA, Q_GAMMA, Q_EDGE_WEAKEN,
    RUMINATION_THETA_MIN, RUMINATION_MAX_CYCLES, RUMINATION_DECAY_FACTOR,
    RUMINATION_E_MAX, RUMINATION_E_CRITICAL, RUMINATION_COST,
    ACTION_SIT_THRESHOLD, ACTION_CL_THRESHOLD, ACTION_CLOSURE_BOOST,
    ACTION_AHA_THRESHOLD, ACTION_AHA_REWARD, ACTION_ENERGY_COST,
    KAPPA_ENABLED, N_CRIT, F_LTM, THETA_G_CONS, KAPPA_DECAY_MULT,
)


@dataclass
class NodeState:
    """Per-node state for BMC simulator.
    Future phases will add: psi, tau_supp, recon_window, habit, n_exec."""
    activation: float = 0.1
    fidelity: float = 0.5
    age: int = 0
    is_meme: bool = True
    # κ consolidation (Phase 1.1)
    kappa: int = 1              # 0=sensory, 1=STM, 2=LTM
    n_react: int = 0            # co-activation count (with refractory period)
    i_passed: bool = True       # passed I-filter (existing memes = True)
    emotional_tag: bool = False  # G-driven fast consolidation
    g_alignment: float = 0.0    # alignment with active G-programs


class _NodeAttrProxy:
    """Dict-like proxy over a NodeState attribute for backward-compatible access.
    Supports get/set/pop/in/iter/dict() — used by memplex-llm code."""
    __slots__ = ('_nodes', '_attr', '_meme_only')

    def __init__(self, nodes, attr, meme_only=False):
        self._nodes = nodes
        self._attr = attr
        self._meme_only = meme_only

    def _ok(self, key):
        if key not in self._nodes:
            return False
        if self._meme_only and not self._nodes[key].is_meme:
            return False
        return True

    def get(self, key, default=None):
        if not self._ok(key):
            return default
        return getattr(self._nodes[key], self._attr)

    def __getitem__(self, key):
        return getattr(self._nodes[key], self._attr)

    def __setitem__(self, key, value):
        if key not in self._nodes:
            self._nodes[key] = NodeState()
        setattr(self._nodes[key], self._attr, value)

    def __contains__(self, key):
        return self._ok(key)

    def __iter__(self):
        if self._meme_only:
            return (k for k, ns in self._nodes.items() if ns.is_meme)
        return iter(self._nodes)

    def pop(self, key, *args):
        if not self._ok(key):
            if args:
                return args[0]
            raise KeyError(key)
        val = getattr(self._nodes[key], self._attr)
        del self._nodes[key]
        return val

    def keys(self):
        return list(self)

    def values(self):
        return [getattr(self._nodes[k], self._attr) for k in self]

    def items(self):
        return [(k, getattr(self._nodes[k], self._attr)) for k in self]

    def __len__(self):
        return sum(1 for _ in self)

    def update(self, d):
        for k, v in d.items():
            self[k] = v


class BMCSimulator:
    """Biomemetic Complex dynamics simulator with signed edges, Panksepp systems,
    SEEKING metasystem, fidelity, ambivalence, and DISGUST I-layer."""

    def __init__(self, G, incompatibility, smc_memes=None, smc_levels=None):
        self.G = G.copy()
        self.incompatibility = incompatibility
        self.meme_nodes = [n for n in G.nodes() if G.nodes[n].get('layer') == 'memetic']
        self.utility_nodes = [n for n in G.nodes() if G.nodes[n].get('layer') == 'utility']
        self.utility_set = set(self.utility_nodes)

        # SMC (Self-Model Cluster)
        meme_set = set(self.meme_nodes)
        if smc_memes is not None:
            self.smc_memes = sorted(smc_memes & meme_set)
        else:
            self.smc_memes = []
        # smc_levels: {meme: 1 or 2}
        self.smc_levels = smc_levels or {}

        # Per-node state: consolidated into NodeState
        self.nodes = {}
        for n in G.nodes():
            is_meme = n not in self.utility_set
            self.nodes[n] = NodeState(
                activation=G.nodes[n].get('activation', 0.1),
                fidelity=G.nodes[n].get('fidelity', 0.5) if is_meme else 0.0,
                age=G.nodes[n].get('age', 0) if is_meme else 0,
                is_meme=is_meme,
            )
        self.fatigue = 0.0

        # Edge weights — signed, w ∈ [-1, +1]
        self.edge_weights = {}
        for u, v in G.edges():
            w = G.edges[u, v].get('weight', 0.5)
            self.edge_weights[(u, v)] = w
            self.edge_weights[(v, u)] = w

        # Spaced-repetition consolidation: per-edge co-activation count
        # Each co-activation (both nodes > θ_high) reduces effective decay rate
        # Refractory period prevents continuous WM co-presence from overcounting
        self.edge_reactivations = {}
        self._last_react_step = {}  # (u,v) → last step when reactivation was counted
        self._last_react_step_node = {}  # node → last step when node co-activation was counted

        # Precompute centrality for differential decay
        self.centrality = nx.degree_centrality(G)

        # Small-worldness σ_SW (expensive — recompute only on structural change)
        self._sigma_sw_dirty = True
        self._sigma_sw_cache = None

        # SIT (Structural Incompleteness Tension)
        self.open_memes = set(n for n in self.meme_nodes
                              if G.nodes[n].get('is_open', False))
        self.closure = {n: G.nodes[n].get('closure', 0.0)
                        for n in self.open_memes}
        self.sit_per_cluster = {}
        self.lp_history = []

        # Q dynamics — cluster registry tracks splits/merges
        self.q_events = []  # list of (step, event_type, details)

        # Rumination limiter — energy budget and per-gap counters
        self.energy = RUMINATION_E_MAX
        self.rumination_counters = {}  # {open_meme: counter}
        self.archived_gaps = set()    # gaps forced to archive
        self._prev_closure = dict(self.closure)  # for LP tracking

        # History
        self.history = []
        self.compact_history = False  # when True, skip activations/edge_weights in history
        self.prev_wm = set()
        self._prev_wm_activations = None
        self._prev_activations = {n: ns.activation for n, ns in self.nodes.items()}
        self._wm_tenure = {}  # node → consecutive steps in WM without significant Δa

    # ── Backward-compatible dict-like properties for memplex-llm ──
    @property
    def activations(self):
        return _NodeAttrProxy(self.nodes, 'activation')

    @activations.setter
    def activations(self, d):
        for k, v in d.items():
            if k in self.nodes:
                self.nodes[k].activation = v
            else:
                self.nodes[k] = NodeState(activation=v)

    @property
    def fidelity(self):
        return _NodeAttrProxy(self.nodes, 'fidelity', meme_only=True)

    @fidelity.setter
    def fidelity(self, d):
        for k, v in d.items():
            if k in self.nodes:
                self.nodes[k].fidelity = v

    @property
    def node_age(self):
        return _NodeAttrProxy(self.nodes, 'age', meme_only=True)

    @node_age.setter
    def node_age(self, d):
        for k, v in d.items():
            if k in self.nodes:
                self.nodes[k].age = v

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-SIGMOID_GAIN * (x - 0.5)))

    def get_weight(self, u, v):
        return self.edge_weights.get((u, v), 0.0)

    def meme_state(self, node):
        if node in self.open_memes:
            return 'open'
        a = self.nodes[node].activation
        if a > THETA_HIGH:
            return 'active'
        elif a < THETA_LOW:
            return 'sleeping'
        else:
            return 'decorative'

    def compute_sit(self):
        """Compute SIT per cluster from open memes."""
        sit = {}
        clusters = set(self.G.nodes[m].get('cluster')
                        for m in self.meme_nodes)
        for cluster in clusters:
            cluster_memes = [m for m in self.meme_nodes
                             if self.G.nodes[m].get('cluster') == cluster]
            cluster_open = [m for m in cluster_memes if m in self.open_memes]
            if not cluster_open:
                sit[cluster] = 0.0
                continue
            cluster_cent = np.mean([self.centrality.get(m, 0)
                                    for m in cluster_memes])
            total = 0.0
            for om in cluster_open:
                neighbors = list(self.G.neighbors(om))
                relevance = (np.mean([abs(self.get_weight(om, nb))
                                      for nb in neighbors])
                             if neighbors else 0)
                cl = self.closure.get(om, 0.0)
                total += relevance * (1 - cl)
            sit[cluster] = total * cluster_cent
        return sit

    def compute_lp(self):
        """Compute Learning Progress: d/dt closure per cluster."""
        if len(self.lp_history) < 2:
            return {c: 0.1 for c in self.sit_per_cluster}
        window = min(SIT_LP_WINDOW, len(self.lp_history))
        prev = self.lp_history[-window]
        curr = {}
        for c in self.sit_per_cluster:
            open_in_c = [m for m in self.open_memes
                         if self.G.nodes[m].get('cluster') == c]
            if open_in_c:
                curr[c] = np.mean([self.closure.get(m, 0) for m in open_in_c])
            else:
                curr[c] = 0.0
        lp = {}
        for c in curr:
            lp[c] = max(0, curr.get(c, 0) - prev.get(c, 0))
        return lp

    @property
    def sigma_sw(self):
        if self._sigma_sw_dirty or self._sigma_sw_cache is None:
            self._sigma_sw_cache = self._compute_sigma_sw()
            self._sigma_sw_dirty = False
        return self._sigma_sw_cache

    def _compute_sigma_sw(self):
        """Compute small-worldness σ = (C/C_rand) / (L/L_rand) for meme subgraph."""
        meme_sub = self.G.subgraph(self.meme_nodes).copy()
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
        L_actual = nx.average_shortest_path_length(meme_sub)
        p = 2 * m / (n * (n - 1))
        C_samples, L_samples = [], []
        for s in range(CL_SIGMA_SW_SAMPLES):
            er = nx.erdos_renyi_graph(n, p, seed=s)
            if nx.is_connected(er):
                C_samples.append(nx.average_clustering(er))
                L_samples.append(nx.average_shortest_path_length(er))
        C_rand = np.mean(C_samples) if C_samples else max(p, 1e-6)
        L_rand = np.mean(L_samples) if L_samples else max(L_actual, 1e-6)
        sigma = (C_actual / max(C_rand, 1e-6)) / (L_actual / max(L_rand, 1e-6))
        return sigma

    def compute_cl(self, a_smc, balance):
        """CL(t) = σ_norm · A_SMC · f(Balance).
        σ_norm = 1 - exp(-σ_SW / scale) ∈ [0, 1]
        f(Balance) = exp(-(Balance - B_opt)² / (2σ_B²))"""
        sigma_norm = 1.0 - np.exp(-self.sigma_sw / CL_SIGMA_SW_SCALE)
        f_balance = np.exp(-((balance - CL_BALANCE_OPT) ** 2) / (2 * CL_BALANCE_SIGMA ** 2))
        return sigma_norm * a_smc * f_balance

    def compute_a_smc(self):
        """Mean activation of SMC memes: A_SMC = (1/|SMC|) Σ a_m."""
        if not self.smc_memes:
            return 0.0
        return np.mean([self.nodes[m].activation for m in self.smc_memes])

    def compute_smc_recursion_depth(self):
        """Continuous recursion depth: weighted sum of level-1 and level-2 SMC activity.
        Level 0 = no SMC activity, Level 1 = direct self-model, Level 2 = meta-self-model."""
        if not self.smc_memes:
            return 0.0
        l1_act = [self.nodes[m].activation for m in self.smc_memes
                   if self.smc_levels.get(m, 1) == 1]
        l2_act = [self.nodes[m].activation for m in self.smc_memes
                   if self.smc_levels.get(m, 1) == 2]
        l1_mean = np.mean(l1_act) if l1_act else 0.0
        l2_mean = np.mean(l2_act) if l2_act else 0.0
        # Depth: 0 if nothing active, up to 2 if level-2 memes dominate
        depth = l1_mean + SMC_RECURSION_WEIGHT * l2_mean
        return min(depth, 2.0)

    def _cluster_dissonance(self, cluster_name):
        """Dissonance between a cluster's memes and the utility (G) layer.
        D = Σ_m (Σ_u inc(u,m) · |a_u - a_m|) for m in cluster with a_m > θ_low."""
        d = 0.0
        for m in self.meme_nodes:
            if self.G.nodes[m].get('cluster') != cluster_name:
                continue
            if self.nodes[m].activation < THETA_LOW:
                continue
            for u in self.utility_nodes:
                inc = self.incompatibility.get(u, {}).get(m, 0.0)
                if inc > 0:
                    d += inc * abs(self.nodes[u].activation - self.nodes[m].activation)
        return d

    def _cross_edge_strength(self, cluster_a, cluster_b):
        """Sum of positive cross-edges between two clusters."""
        total = 0.0
        for m1 in self.meme_nodes:
            if self.G.nodes[m1].get('cluster') != cluster_a:
                continue
            for nb in self.G.neighbors(m1):
                if nb in self.utility_set:
                    continue
                if self.G.nodes[nb].get('cluster') == cluster_b:
                    w = self.get_weight(m1, nb)
                    if w > 0:
                        total += w
        return total

    def check_q_dynamics(self):
        """Check for memplex splitting/merging. Called every Q_CHECK_INTERVAL steps."""
        t_step = len(self.history)
        meme_sub = self.G.subgraph(self.meme_nodes)

        # Get current cluster assignments
        clusters = {}
        for m in self.meme_nodes:
            c = self.G.nodes[m].get('cluster', 'unknown')
            clusters.setdefault(c, []).append(m)

        events = []

        # ── Splitting check ──
        for cname, members in list(clusters.items()):
            if len(members) < 6:  # too small to split
                continue
            sub = meme_sub.subgraph(members)
            if sub.number_of_edges() == 0:
                continue

            # Compute intra-cluster modularity using weighted edges
            try:
                comms = nx.community.louvain_communities(sub, seed=42)
            except Exception:
                continue
            if len(comms) < 2:
                continue
            q = nx.community.modularity(sub, comms)

            # Dissonance pressure
            d = self._cluster_dissonance(cname)
            cross_total = 0.0
            for other_c in clusters:
                if other_c != cname:
                    cross_total += self._cross_edge_strength(cname, other_c)

            dq = Q_ALPHA * d - Q_BETA * cross_total

            if q > Q_CRIT and dq > 0:
                # Split: assign the two largest subcommunities to new cluster names
                sorted_comms = sorted(comms, key=len, reverse=True)
                new_c1 = cname
                new_c2 = f"{cname}_split_{t_step}"
                for m in sorted_comms[1]:
                    self.G.nodes[m]['cluster'] = new_c2
                    # Weaken cross-edges between the split halves
                    for nb in self.G.neighbors(m):
                        if nb in self.utility_set:
                            continue
                        if self.G.nodes[nb].get('cluster') == new_c1:
                            w = self.get_weight(m, nb)
                            weakened = w * (1 - Q_EDGE_WEAKEN)
                            self.edge_weights[(m, nb)] = weakened
                            self.edge_weights[(nb, m)] = weakened
                events.append(('split', cname, new_c2, q, dq))

        # ── Merging check ──
        cluster_names = list(set(self.G.nodes[m].get('cluster') for m in self.meme_nodes))
        for i, ca in enumerate(cluster_names):
            for cb in cluster_names[i+1:]:
                cross = self._cross_edge_strength(ca, cb)
                if cross < 0.5:  # need meaningful connections
                    continue
                d_a = self._cluster_dissonance(ca)
                d_b = self._cluster_dissonance(cb)
                # Estimate union dissonance (approximation: min of individual)
                d_union = min(d_a, d_b) * 0.7
                dq_merge = -Q_GAMMA * (d_a + d_b - d_union)
                if dq_merge < -Q_MERGE * 0.1:
                    # Merge: rename all cb memes to ca
                    for m in self.meme_nodes:
                        if self.G.nodes[m].get('cluster') == cb:
                            self.G.nodes[m]['cluster'] = ca
                    # Strengthen cross-edges between merged members
                    events.append(('merge', ca, cb, cross, dq_merge))
                    break  # one merge per check

        self.q_events.extend([(t_step, *e) for e in events])
        return events

    def check_rumination(self):
        """Check for unproductive rumination on open memes.
        Returns list of events: ('archive', gap), ('emergency_blend',), or ('warning', gap, remaining)."""
        events = []
        if not self.open_memes:
            return events

        active_gaps = self.open_memes - self.archived_gaps

        for gap in list(active_gaps):
            cluster = self.G.nodes[gap].get('cluster', '')
            gap_sit = self.sit_per_cluster.get(cluster, 0.0)

            # LP per gap: closure change since last check
            curr_cl = self.closure.get(gap, 0.0)
            prev_cl = self._prev_closure.get(gap, curr_cl)
            gap_lp = curr_cl - prev_cl
            self._prev_closure[gap] = curr_cl

            if gap_lp > RUMINATION_THETA_MIN:
                self.rumination_counters[gap] = 0
                continue

            # Only count rumination if gap is actively attended (SIT > 0)
            if gap_sit <= 0:
                continue

            self.rumination_counters[gap] = self.rumination_counters.get(gap, 0) + 1
            self.energy -= RUMINATION_COST * gap_sit  # cost proportional to SIT

            if self.rumination_counters[gap] > RUMINATION_MAX_CYCLES:
                # FORCE SWITCH: archive gap, reduce SIT
                self.archived_gaps.add(gap)
                self.closure[gap] = min(1.0, self.closure.get(gap, 0) + 0.5)
                # Reduce SIT for this cluster
                if cluster in self.sit_per_cluster:
                    self.sit_per_cluster[cluster] *= RUMINATION_DECAY_FACTOR
                events.append(('archive', gap, self.rumination_counters[gap]))

            if self.energy < RUMINATION_E_CRITICAL * RUMINATION_E_MAX:
                # EMERGENCY: force all gaps to archive, trigger consolidation
                for g in list(active_gaps):
                    self.archived_gaps.add(g)
                    self.closure[g] = min(1.0, self.closure.get(g, 0) + 0.3)
                events.append(('emergency_blend',))
                self.sleep_consolidation()
                self.energy = RUMINATION_E_MAX * 0.5  # partial recovery
                break

        # Passive energy recovery
        self.energy = min(RUMINATION_E_MAX, self.energy + 0.005)
        return events

    def action_loop(self, cl):
        """Executive action loop: SIT → plan → execute → closure update.
        Returns list of events: ('action', gap, closure_delta), ('aha', gap)."""
        events = []
        if cl < ACTION_CL_THRESHOLD:
            return events  # CL too low to plan
        if self.energy < ACTION_ENERGY_COST:
            return events  # no energy

        # Find highest-SIT active gap
        active_gaps = self.open_memes - self.archived_gaps
        if not active_gaps:
            return events

        best_gap = None
        best_sit = 0.0
        for gap in active_gaps:
            cluster = self.G.nodes[gap].get('cluster', '')
            gap_sit = self.sit_per_cluster.get(cluster, 0.0)
            if gap_sit > best_sit:
                best_sit = gap_sit
                best_gap = gap

        if best_gap is None or best_sit < ACTION_SIT_THRESHOLD:
            return events

        # Execute action: boost closure and strengthen edges to neighbors
        self.energy -= ACTION_ENERGY_COST
        old_closure = self.closure.get(best_gap, 0.0)

        # Closure improvement proportional to SEEKING activation and CL
        seeking_act = self.nodes['SEEKING'].activation
        closure_delta = ACTION_CLOSURE_BOOST * seeking_act * cl
        new_closure = min(1.0, old_closure + closure_delta)
        self.closure[best_gap] = new_closure

        # Strengthen edges around the gap (action creates new associations)
        for nb in self.G.neighbors(best_gap):
            if nb in self.utility_set:
                continue
            w = self.get_weight(best_gap, nb)
            if w > 0:
                boost = 0.02 * seeking_act
                self.edge_weights[(best_gap, nb)] = min(1.0, w + boost)
                self.edge_weights[(nb, best_gap)] = min(1.0, w + boost)

        events.append(('action', best_gap, closure_delta))

        # AHA moment check
        if new_closure >= ACTION_AHA_THRESHOLD:
            # Reward: boost SEEKING, zero SIT for this gap
            if 'SEEKING' in self.nodes:
                self.nodes['SEEKING'].activation = min(1.0,
                    self.nodes['SEEKING'].activation + ACTION_AHA_REWARD)
            cluster = self.G.nodes[best_gap].get('cluster', '')
            if cluster in self.sit_per_cluster:
                self.sit_per_cluster[cluster] = 0.0
            events.append(('aha', best_gap, new_closure))

        return events

    def step(self, stimuli=None):
        """One simulation step."""
        if stimuli is None:
            stimuli = {}

        # Snapshot activations for delta-salience (phasic dopamine analog)
        self._prev_activations = {n: ns.activation for n, ns in self.nodes.items()}

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
                        feedback -= 0.1 * abs(w) * self.nodes[nb].activation
                    else:
                        feedback += 0.05 * w * self.nodes[nb].activation
            target = np.clip(base + stim + feedback, 0.0, 1.0)
            non_seeking_acts[u] = UTILITY_INERTIA * self.nodes[u].activation + (1 - UTILITY_INERTIA) * target

        # SEEKING = T_SEEK * (base + SIT_signal) + Σ α_s * a_s (recruitment by other systems)
        sit_values = self.compute_sit()
        lp_values = self.compute_lp()
        if 'SEEKING' in self.nodes:
            seek_base = self.G.nodes['SEEKING'].get('base_activation', 0.45)
            seek_stim = stimuli.get('SEEKING', 0.0)
            recruit_sum = sum(
                ALPHA_RECRUIT.get(s, 0.0) * non_seeking_acts.get(s, self.nodes[s].activation)
                for s in ALPHA_RECRUIT
            )
            # SIT: structural gaps boost SEEKING even without external stimuli
            sit_signal = GAMMA_SIT * sum(
                sit_values.get(c, 0) * max(lp_values.get(c, 0), 0.01)
                for c in sit_values)
            seek_target = np.clip(
                T_SEEK * (seek_base + sit_signal) + recruit_sum + seek_stim,
                0.0, 1.0)
            seek_feedback = 0.0
            for nb in self.G.neighbors('SEEKING'):
                if nb not in self.utility_set:
                    w = self.get_weight('SEEKING', nb)
                    seek_feedback += 0.05 * w * self.nodes[nb].activation
            seek_target = np.clip(seek_target + seek_feedback, 0.0, 1.0)
            non_seeking_acts['SEEKING'] = UTILITY_INERTIA * self.nodes['SEEKING'].activation + \
                                          (1 - UTILITY_INERTIA) * seek_target

        for u in self.utility_nodes:
            new_act[u] = non_seeking_acts.get(u, self.nodes[u].activation)

        # DISGUST special: activates more when incompatible input detected
        if 'DISGUST' in new_act:
            disgust_boost = 0.0
            for m in self.meme_nodes:
                neg_edges = sum(1 for nb in self.G.neighbors(m)
                               if nb not in self.utility_set
                               and self.get_weight(m, nb) < 0
                               and self.nodes[m].activation > THETA_HIGH
                               and self.nodes[nb].activation > THETA_HIGH)
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
                        incoming_utility += UTILITY_SPREAD_REDIRECT * w * self.nodes[nb].activation
                    elif etype == 'suppress':
                        incoming_utility -= UTILITY_SPREAD_SUPPRESS * abs(w) * self.nodes[nb].activation
                    else:  # interpret
                        incoming_utility += UTILITY_SPREAD_INTERPRET * w * self.nodes[nb].activation
                else:
                    incoming_meme += w * self.nodes[nb].activation
                    n_meme_neighbors += 1
            if n_meme_neighbors > 0:
                incoming_meme /= n_meme_neighbors
            stim = stimuli.get(m, 0.0)
            # κ-dependent self-decay: sensory memes decay faster, LTM slower
            lam_self = LAMBDA_DECAY
            if KAPPA_ENABLED:
                lam_self *= KAPPA_DECAY_MULT.get(self.nodes[m].kappa, 1.0)
                lam_self = min(lam_self, 0.95)  # cap to prevent negative retention
            raw = (1 - lam_self) * self.nodes[m].activation + incoming_meme + incoming_utility + stim
            new_act[m] = self.sigmoid(raw) * fatigue_factor

        # ── (b2) Open meme pulsation (SIT) ──
        t_step = len(self.history)
        for om in self.open_memes:
            if om in new_act:
                pulse = OPEN_MEME_PULSE_AMP * np.sin(
                    2 * np.pi * t_step / OPEN_MEME_PULSE_PERIOD)
                cl = self.closure.get(om, 0)
                new_act[om] = np.clip(new_act[om] + pulse * (1 - cl), 0.0, 1.0)

        pre_wm_act = dict(new_act)

        # ── (c) Lateral inhibition → Working Memory selection ──
        # Reserve WM slots for DYN_ nodes (conversation-derived content)
        # so they don't get crowded out by high-baseline static clusters.
        DYN_RESERVED_SLOTS = 3
        meme_acts = np.array([new_act[m] for m in self.meme_nodes])
        if meme_acts.max() > 0:
            # Separate DYN_ and static nodes
            dyn_indices = [i for i, m in enumerate(self.meme_nodes) if m.startswith("DYN_")]
            static_indices = [i for i, m in enumerate(self.meme_nodes) if not m.startswith("DYN_")]

            wm_indices_set = set()

            # Fill reserved DYN_ slots (by raw activation, top-K)
            if dyn_indices:
                dyn_acts = [(i, meme_acts[i]) for i in dyn_indices]
                dyn_acts.sort(key=lambda x: x[1], reverse=True)
                n_dyn = min(DYN_RESERVED_SLOTS, len(dyn_acts))
                for i, _ in dyn_acts[:n_dyn]:
                    wm_indices_set.add(i)

            # Update WM tenure for habituation (before WM selection uses it).
            # Nodes in prev_wm without significant Δa accumulate tenure → lower salience.
            new_tenure = {}
            for node in self.prev_wm:
                delta = abs(new_act.get(node, 0.0) - self._prev_activations.get(node, 0.0))
                if delta < WM_DELTA_THRESHOLD:
                    new_tenure[node] = self._wm_tenure.get(node, 0) + 1
                # else: dishabituation — tenure resets to 0 (absent from dict)
            self._wm_tenure = new_tenure

            # Fill remaining slots from ALL nodes by delta-modulated softmax.
            # Phasic boost (dopamine analog) + habituation (neural adaptation).
            remaining = WM_SIZE - len(wm_indices_set)
            if remaining > 0:
                delta_acts = np.array([
                    abs(new_act[m] - self._prev_activations.get(m, new_act[m]))
                    for m in self.meme_nodes
                ])
                tenure = np.array([
                    self._wm_tenure.get(m, 0) for m in self.meme_nodes
                ], dtype=float)
                salience = (meme_acts
                            * (1 + WM_DELTA_SALIENCE_BOOST * delta_acts)
                            / (1 + WM_HABITUATION_RATE * tenure))
                # Log-sum-exp trick for numerical stability
                log_s = salience / SOFTMAX_TEMP
                log_s -= log_s.max()
                exp_a = np.exp(log_s)
                softmax_a = exp_a / exp_a.sum()
                ranked = np.argsort(softmax_a)[::-1]
                for idx in ranked:
                    if idx not in wm_indices_set:
                        wm_indices_set.add(idx)
                        if len(wm_indices_set) >= WM_SIZE:
                            break

            wm_set = set(self.meme_nodes[i] for i in wm_indices_set)
            for i, m in enumerate(self.meme_nodes):
                if i not in wm_indices_set:
                    new_act[m] *= 0.85
        else:
            wm_set = set()

        # ── (d) Asymmetric edge decay ──
        for (u, v), w in list(self.edge_weights.items()):
            cu = self.centrality.get(u, 0.01)
            cv = self.centrality.get(v, 0.01)
            lam_base = LAMBDA_EDGE_POS if w >= 0 else LAMBDA_EDGE_NEG
            lam = lam_base / (1 + ALPHA_CENTRALITY * cu * cv)
            # Open meme edges decay slower (SIT: persistent structural gaps)
            if u in self.open_memes or v in self.open_memes:
                lam *= OPEN_MEME_DECAY_RESISTANCE
            # Spaced repetition: past co-activations reduce decay rate
            # (early LTP → late LTP → structural LTP consolidation)
            n_react = self.edge_reactivations.get((u, v), 0)
            if n_react > 0:
                lam /= (1 + SPACED_REP_FACTOR * n_react)
            # κ-dependent decay: sensory edges decay 10× faster, LTM 10× slower
            if KAPPA_ENABLED:
                ku = self.nodes[u].kappa if self.nodes[u].is_meme else 1
                kv = self.nodes[v].kappa if self.nodes[v].is_meme else 1
                k_min = min(ku, kv)  # weakest link determines decay
                lam *= KAPPA_DECAY_MULT.get(k_min, 1.0)
            new_w = w * np.exp(-lam)
            a_u = pre_wm_act.get(u, self.nodes[u].activation)
            a_v = pre_wm_act.get(v, self.nodes[v].activation)
            if a_u >= THETA_HIGH and a_v >= THETA_HIGH:
                is_dormant = abs(new_w) <= CONSOLIDATION_TRACE_LEVEL
                boost = DELTA_W_REACTIVATE * (SAVINGS_MULTIPLIER if is_dormant else 1.0)
                if w >= 0:
                    new_w = min(1.0, new_w + boost)
                else:
                    new_w = max(-1.0, new_w - boost * 0.5)
                # Increment co-activation counter only after refractory gap
                # Exclude edges touching open memes (they have their own
                # persistence via decay resistance; consolidation is for learning)
                if w >= 0 and u not in self.open_memes and v not in self.open_memes:
                    current_step = len(self.history)
                    last_step = self._last_react_step.get((u, v), -SPACED_REP_MIN_GAP - 1)
                    if current_step - last_step >= SPACED_REP_MIN_GAP:
                        self.edge_reactivations[(u, v)] = n_react + 1
                        self._last_react_step[(u, v)] = current_step
                    # Node-level n_react for κ transitions
                    if KAPPA_ENABLED:
                        for node in (u, v):
                            if self.nodes[node].is_meme:
                                last_n = self._last_react_step_node.get(node, -SPACED_REP_MIN_GAP - 1)
                                if current_step - last_n >= SPACED_REP_MIN_GAP:
                                    self.nodes[node].n_react += 1
                                    self._last_react_step_node[node] = current_step
            self.edge_weights[(u, v)] = np.clip(new_w, -1.0, 1.0)

        # ── (d2) Sign inversion (bifurcation) ──
        inversions = []
        if SIGN_INVERSION_ENABLED:
            seen = set()
            for (u, v), w in list(self.edge_weights.items()):
                if w >= 0 or (v, u) in seen:
                    continue
                if u in self.utility_set or v in self.utility_set:
                    continue
                seen.add((u, v))
                # Indirect pressure: Σ_k w_uk · w_kv · a_k over common neighbors
                common = set(self.G.neighbors(u)) & set(self.G.neighbors(v))
                common -= self.utility_set
                pressure = 0.0
                for k in common:
                    w_uk = self.get_weight(u, k)
                    w_kv = self.get_weight(k, v)
                    a_k = pre_wm_act.get(k, self.nodes[k].activation)
                    pressure += w_uk * w_kv * a_k
                if pressure > abs(w):
                    # Bifurcation: flip sign, preserve magnitude
                    new_w = abs(w)
                    self.edge_weights[(u, v)] = new_w
                    self.edge_weights[(v, u)] = new_w
                    inversions.append((u, v, w, new_w))

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
            self.nodes[m].age += 1
            age = self.nodes[m].age
            self.nodes[m].fidelity = ((k_m / k_max) ** FIDELITY_GAMMA *
                                      np.exp(-FIDELITY_LAMBDA * 1) *
                                      (1 - np.exp(-FIDELITY_BETA * age)))

        # ── Apply new activations ──
        for n, val in new_act.items():
            self.nodes[n].activation = val

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
        wm_meme_acts = [self.nodes[m].activation for m in wm_set] if wm_set else [0.0]
        A_meme = np.mean(wm_meme_acts)
        A_utility = np.mean([self.nodes[u].activation for u in self.utility_nodes])
        balance = A_meme / max(A_utility, 1e-6)

        conflict = 0.0
        for u in self.utility_nodes:
            for m in self.meme_nodes:
                inc = self.incompatibility.get(u, {}).get(m, 0.0)
                if inc > 0:
                    conflict += abs(self.nodes[u].activation - self.nodes[m].activation) * inc

        dissonance = 0.0
        for u in self.utility_nodes:
            for m in self.meme_nodes:
                inc = self.incompatibility.get(u, {}).get(m, 0.0)
                if inc > 0:
                    dissonance += min(self.nodes[u].activation, self.nodes[m].activation) * inc

        tension = 0.0
        for u in self.utility_nodes:
            a_u = self.nodes[u].activation
            for nb in self.G.neighbors(u):
                if nb not in self.utility_set:
                    a_m = self.nodes[nb].activation
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
        curr_wm_activations = {m: self.nodes[m].activation for m in wm_set}
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

        states = {'active': 0, 'decorative': 0, 'sleeping': 0, 'open': 0}
        for m in self.meme_nodes:
            states[self.meme_state(m)] += 1

        # SIT history tracking
        self.sit_per_cluster = sit_values
        lp_closure = {}
        for c in sit_values:
            open_in_c = [m for m in self.open_memes
                         if self.G.nodes[m].get('cluster') == c]
            if open_in_c:
                lp_closure[c] = np.mean([self.closure.get(m, 0)
                                         for m in open_in_c])
            else:
                lp_closure[c] = 0.0
        self.lp_history.append(lp_closure)

        sit_total = sum(sit_values.get(c, 0) * max(lp_values.get(c, 0), 0.01)
                        for c in sit_values)

        # ── (h) Q dynamics (periodic) ──
        t_step = len(self.history)
        q_events_this_step = []
        if t_step > 0 and t_step % Q_CHECK_INTERVAL == 0:
            q_events_this_step = self.check_q_dynamics()

        # ── SMC & CL metrics ──
        a_smc = self.compute_a_smc()
        smc_depth = self.compute_smc_recursion_depth()
        cl = self.compute_cl(a_smc, balance)

        # ── (i) Action loop (before rumination — actions improve LP) ──
        action_events = self.action_loop(cl)

        # ── (j) Rumination limiter ──
        rumination_events = self.check_rumination()

        # ── (k) κ consolidation update ──
        if KAPPA_ENABLED:
            # Update g_alignment and emotional_tag based on co-active G-programs
            active_g = [(u, self.nodes[u].activation) for u in self.utility_nodes
                        if self.nodes[u].activation >= THETA_HIGH]
            for m in self.meme_nodes:
                ns = self.nodes[m]
                if ns.activation >= THETA_HIGH and active_g:
                    # g_alignment = max alignment with any active G-program
                    g_align = 0.0
                    for u, a_u in active_g:
                        w = self.get_weight(m, u)
                        if w > 0:
                            g_align = max(g_align, w * a_u)
                    ns.g_alignment = max(ns.g_alignment, g_align)
                    if g_align > THETA_G_CONS:
                        ns.emotional_tag = True
                # Recalculate κ
                if not ns.i_passed:
                    ns.kappa = 0
                elif (ns.n_react >= N_CRIT
                      or ns.fidelity >= F_LTM
                      or (ns.g_alignment > THETA_G_CONS and ns.emotional_tag)):
                    ns.kappa = 2
                else:
                    ns.kappa = max(ns.kappa, 1)  # never demote

        entry = {
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
            'sit_total': sit_total,
            'sit_per_cluster': dict(sit_values),
            'a_smc': a_smc,
            'smc_depth': smc_depth,
            'cl': cl,
            'inversions': inversions,
            'q_events': q_events_this_step,
            'rumination_events': rumination_events,
            'energy': self.energy,
            'action_events': action_events,
        }
        if not self.compact_history:
            entry['activations'] = {n: ns.activation for n, ns in self.nodes.items()}
            entry['edge_weights'] = dict(self.edge_weights)
        self.history.append(entry)

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
        # Guard: if node already exists, just boost activation (no duplicate in meme_nodes)
        if self.G.has_node(name):
            self.nodes[name].activation = min(1.0, self.nodes[name].activation + initial_activation)
            return True, self.nodes[name].activation

        score = 0.0
        for node, compat_val in compatibility.items():
            if node in self.nodes:
                score += self.nodes[node].activation * compat_val

        accepted = score >= threshold

        if accepted:
            self.G.add_node(name, layer='memetic', cluster=cluster,
                            activation=initial_activation,
                            fidelity=0.3, age=0)
            self.meme_nodes.append(name)
            self.nodes[name] = NodeState(
                activation=initial_activation, fidelity=0.3, age=0, is_meme=True,
                kappa=0, n_react=0, i_passed=True,
            )

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
            self._sigma_sw_dirty = True
        else:
            # Rejection: add meme with NEGATIVE edges (sleeper potential)
            self.G.add_node(name, layer='memetic', cluster=cluster,
                            activation=0.05, fidelity=0.1, age=0)
            self.meme_nodes.append(name)
            self.nodes[name] = NodeState(
                activation=0.05, fidelity=0.1, age=0, is_meme=True,
                kappa=0, n_react=0, i_passed=False,
            )

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
            if 'DISGUST' in self.nodes:
                self.nodes['DISGUST'].activation = min(1.0,
                    self.nodes['DISGUST'].activation + 0.3)
            # Also boost FEAR as defensive reaction
            if 'FEAR' in self.nodes:
                self.nodes['FEAR'].activation = min(1.0,
                    self.nodes['FEAR'].activation + 0.15)

            self.centrality = nx.degree_centrality(self.G)
            self._sigma_sw_dirty = True

        return accepted, score

    def sleep_consolidation(self):
        """Sleep cycle: DECOMPOSE → CONNECT → BLEND → PRUNE → STRENGTHEN."""
        rng = np.random.RandomState(len(self.history))

        # 1. DECOMPOSE — reduce activations (entering sleep)
        for m in self.meme_nodes:
            self.nodes[m].activation *= 0.3
        for u in self.utility_nodes:
            self.nodes[u].activation *= 0.5

        # 2. CONNECT — strengthen edges between co-active pairs
        for m1 in self.meme_nodes:
            for m2 in self.meme_nodes:
                if m1 >= m2:
                    continue
                key = (m1, m2)
                if key in self.edge_weights:
                    a1 = self.nodes[m1].activation
                    a2 = self.nodes[m2].activation
                    if a1 > 0.05 and a2 > 0.05:
                        w = self.edge_weights[key]
                        boost = 0.05 * min(a1, a2)
                        self.edge_weights[key] = np.clip(w + boost, -1.0, 1.0)
                        self.edge_weights[(m2, m1)] = self.edge_weights[key]

        # 3. BLEND — create new node from two memes of different clusters
        #    SIT: prioritize candidates from high-SIT clusters
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
                        sit_bonus = max(self.sit_per_cluster.get(c1, 0),
                                        self.sit_per_cluster.get(c2, 0))
                        blend_candidates.append((m1, m2, w + sit_bonus))

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
                self.nodes[blend_name] = NodeState(
                    activation=0.2, fidelity=0.3, age=0, is_meme=True,
                    kappa=0, n_react=0, i_passed=True,
                )

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
                self._sigma_sw_dirty = True
                blend_created.append(blend_name)

        # 4a. SHY — proportional synaptic downscaling (Tononi & Cirelli 2003, 2014)
        # κ=2 (LTM) edges are protected from downscaling
        for (u, v), w in list(self.edge_weights.items()):
            if u in self.utility_set or v in self.utility_set:
                continue
            # κ-dependent protection: LTM edges resist SHY
            if KAPPA_ENABLED:
                ku = self.nodes[u].kappa if self.nodes[u].is_meme else 1
                kv = self.nodes[v].kappa if self.nodes[v].is_meme else 1
                if min(ku, kv) >= 2:
                    continue  # LTM edges skip SHY
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
                    f1 = self.nodes[m1].fidelity
                    f2 = self.nodes[m2].fidelity
                    if f1 > 0.5 and f2 > 0.5:
                        w = self.edge_weights[key]
                        boost = 0.03 * (f1 + f2)
                        self.edge_weights[key] = np.clip(w + boost, -1.0, 1.0)
                        self.edge_weights[(m2, m1)] = self.edge_weights[key]

        return blend_created

    def false_closure(self, open_meme_name, answer_name, cluster):
        """Introduce a false-closure node: invalid answer to an open question.

        Sets closure of open meme to 0.8, but the answer has low fidelity
        and near-zero validity. SIT for that cluster drops.
        """
        self.G.add_node(answer_name, layer='memetic', cluster=cluster,
                        activation=0.2, fidelity=0.1, age=0,
                        validity=FALSE_CLOSURE_VALIDITY)
        self.meme_nodes.append(answer_name)
        self.nodes[answer_name] = NodeState(
            activation=0.2, fidelity=0.1, age=0, is_meme=True,
            kappa=0, n_react=0, i_passed=True,
        )

        # Connect answer to the open meme
        self.G.add_edge(answer_name, open_meme_name, weight=0.6, etype='cross')
        self.edge_weights[(answer_name, open_meme_name)] = 0.6
        self.edge_weights[(open_meme_name, answer_name)] = 0.6

        # Connect to a few neighbors of the open meme
        for nb in list(self.G.neighbors(open_meme_name)):
            if nb != answer_name and nb not in self.utility_set:
                w = self.get_weight(open_meme_name, nb)
                if abs(w) > 0.3:
                    inh_w = w * 0.4
                    self.G.add_edge(answer_name, nb, weight=inh_w, etype='cross')
                    self.edge_weights[(answer_name, nb)] = inh_w
                    self.edge_weights[(nb, answer_name)] = inh_w

        # Update incompatibility
        for u in self.utility_nodes:
            if u not in self.incompatibility:
                self.incompatibility[u] = {}
            self.incompatibility[u][answer_name] = 0.0

        # Set closure of the open meme to 0.8 (mostly closed)
        if open_meme_name in self.closure:
            self.closure[open_meme_name] = 0.8

        self.centrality = nx.degree_centrality(self.G)
        self._sigma_sw_dirty = True

    def reset(self):
        """Reset to initial state."""
        for n in self.G.nodes():
            if self.G.nodes[n].get('layer') == 'utility':
                self.nodes[n].activation = self.G.nodes[n].get('base_activation', 0.4)
            else:
                self.nodes[n].activation = np.random.uniform(0.05, 0.25)
        self.fatigue = 0.0
        for n in self.meme_nodes:
            self.nodes[n].fidelity = self.G.nodes[n].get('fidelity', 0.5)
            self.nodes[n].age = self.G.nodes[n].get('age', 0)
        for u, v in self.G.edges():
            w = self.G.edges[u, v].get('weight', 0.5)
            self.edge_weights[(u, v)] = w
            self.edge_weights[(v, u)] = w
        self.closure = {n: self.G.nodes[n].get('closure', 0.0)
                        for n in self.open_memes}
        self.sit_per_cluster = {}
        self.lp_history = []
        self.q_events = []
        self.edge_reactivations = {}
        self._last_react_step = {}
        self._last_react_step_node = {}
        # Reset κ fields
        for n in self.meme_nodes:
            self.nodes[n].kappa = 1
            self.nodes[n].n_react = 0
            self.nodes[n].i_passed = True
            self.nodes[n].emotional_tag = False
            self.nodes[n].g_alignment = 0.0
        self.energy = RUMINATION_E_MAX
        self.rumination_counters = {}
        self.archived_gaps = set()
        self._prev_closure = dict(self.closure)
        self._sigma_sw_dirty = True
        self.history = []
        self.prev_wm = set()
        self._prev_wm_activations = None
        self._prev_activations = {n: ns.activation for n, ns in self.nodes.items()}
        self._wm_tenure = {}
