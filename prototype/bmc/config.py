"""Simulation parameters and constants."""
from collections import OrderedDict

# ── Activation thresholds ──
THETA_HIGH = 0.5    # above → Active
THETA_LOW  = 0.1    # below → Sleeping; between → Decorative

# ── Spreading activation ──
LAMBDA_DECAY = 0.15          # self-decay coefficient
SIGMOID_GAIN = 5.0           # steepness of sigmoid

# ── Edge decay — asymmetric (negativity bias) ──
LAMBDA_EDGE_POS = 0.05       # decay for positive edges
LAMBDA_EDGE_NEG = 0.025      # decay for negative edges (slower — negativity bias)
ALPHA_CENTRALITY = 0.5       # centrality protection factor
DELTA_W_REACTIVATE = 0.15    # weight boost on co-activation

# ── Fidelity (orthogonal to weight) ──
FIDELITY_GAMMA = 0.5         # degree exponent
FIDELITY_LAMBDA = 0.03       # temporal decay
FIDELITY_BETA = 0.1          # age maturation rate

# ── SEEKING metasystem recruitment ──
ALPHA_RECRUIT = {
    'FEAR': 0.15, 'RAGE': 0.10, 'LUST': 0.12,
    'CARE': 0.10, 'PANIC_GRIEF': 0.15, 'PLAY': 0.08,
}
T_SEEK = 0.6                 # SEEKING base scaling

# ── Sleep consolidation ──
CONSOLIDATION_PRUNE_THRESHOLD = 0.05
CONSOLIDATION_TRACE_LEVEL = 0.01   # dormant edge weight floor (silent synapse analog)
SHY_DOWNSCALE_FACTOR = 0.12        # proportional synaptic downscaling during sleep (SHY)
SAVINGS_MULTIPLIER = 2.5            # reactivation boost for dormant edges (savings effect)
BLEND_THRESHOLD = 0.4

# ── Utility→Meme interface coefficients (per etype) ──
UTILITY_SPREAD_REDIRECT  = 0.15   # redirect: utility gently guides memes
UTILITY_SPREAD_SUPPRESS  = 0.30   # suppress: stronger inhibitory signal
UTILITY_SPREAD_INTERPRET = 0.10   # interpret: weak cognitive filter

# ── Utility dynamics ──
UTILITY_INERTIA = 0.3            # utility activation memory (0=stateless, 1=frozen)

# ── Lateral inhibition ──
WM_SIZE = 7                  # working memory capacity (7±2)
SOFTMAX_TEMP = 0.15          # softmax temperature (low = sharper)

# ── Fatigue ──
FATIGUE_RATE = 0.02          # fatigue accumulation per active step
RECOVERY_RATE = 0.01         # passive recovery per step
FATIGUE_SUPPRESSION = 0.5    # how fatigue suppresses meme activation

# ── Utility nodes: 7 Panksepp systems + DISGUST (I-layer) ──
UTILITY_NODES = OrderedDict([
    ('SEEKING',     {'base': 0.45, 'valence': '+', 'nt': 'Dopamine, glutamate'}),
    ('FEAR',        {'base': 0.35, 'valence': '-', 'nt': 'Glutamate, CRF, CCK'}),
    ('RAGE',        {'base': 0.30, 'valence': '-', 'nt': 'Substance P, acetylcholine'}),
    ('LUST',        {'base': 0.30, 'valence': '+', 'nt': 'Gonadal steroids, vasopressin/oxytocin'}),
    ('CARE',        {'base': 0.40, 'valence': '+', 'nt': 'Oxytocin, prolactin'}),
    ('PANIC_GRIEF', {'base': 0.35, 'valence': '-', 'nt': 'CRF, glutamate'}),
    ('PLAY',        {'base': 0.35, 'valence': '+', 'nt': 'Opioids, endocannabinoids'}),
    ('DISGUST',     {'base': 0.25, 'valence': '-', 'nt': 'Insula activation',
                     'role': 'I-layer: negative weight assignment'}),
])
