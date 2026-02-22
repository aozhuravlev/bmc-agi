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
SPACED_REP_FACTOR = 2.0             # decay-rate reduction per co-activation
                                    # λ_eff = λ / (1 + factor · n_reactivations)
                                    # models early→late LTP consolidation
SPACED_REP_MIN_GAP = 10             # minimum steps between counted co-activations
                                    # prevents continuous WM co-presence from counting
                                    # as multiple "repetitions"; models refractory period
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
WM_DELTA_SALIENCE_BOOST = 3.0  # delta-activation salience boost (phasic dopamine gain)
WM_HABITUATION_RATE = 0.4      # tenure-based suppression rate (neural adaptation)
WM_DELTA_THRESHOLD = 0.03      # min |Δa| to count as "significant change" (resets tenure)

# ── Fatigue ──
FATIGUE_RATE = 0.02          # fatigue accumulation per active step
RECOVERY_RATE = 0.01         # passive recovery per step
FATIGUE_SUPPRESSION = 0.5    # how fatigue suppresses meme activation

# ── Utility nodes: 7 Panksepp systems + DISGUST (I-layer) ──
# ── SIT (Structural Incompleteness Tension) ──
GAMMA_SIT = 0.25                  # SIT weight in SEEKING formula
SIT_LP_WINDOW = 5                 # timesteps for LP computation (d/dt closure)
OPEN_MEME_DECAY_RESISTANCE = 0.5  # open meme edges decay this factor slower
OPEN_MEME_PULSE_AMP = 0.08       # pulsation amplitude for open memes
OPEN_MEME_PULSE_PERIOD = 6       # pulsation period (timesteps)
FALSE_CLOSURE_VALIDITY = 0.05    # validity of false-closure nodes (near zero)

# ── SMC (Self-Model Cluster) ──
SMC_RECURSION_WEIGHT = 1.5       # level-2 memes contribute 1.5× to recursion depth

# ── CL (Consciousness Level) metric ──
CL_BALANCE_OPT = 1.5             # optimal M/G balance
CL_BALANCE_SIGMA = 0.7           # width of balance bell curve (σ_B)
CL_SIGMA_SW_SAMPLES = 20         # random graph samples for σ_SW baseline
CL_SIGMA_SW_SCALE = 5.0          # σ_SW normalization: σ_norm = 1 - exp(-σ/scale)

# ── Sign inversion (bifurcation) ──
SIGN_INVERSION_ENABLED = True    # enable/disable sign inversion check

# ── Q dynamics (splitting/merging) ──
Q_CHECK_INTERVAL = 10            # check splitting/merging every N steps
Q_CRIT = 0.7                    # modularity threshold for splitting
Q_MERGE = 0.3                   # modularity threshold for merging
Q_ALPHA = 0.1                   # dissonance → splitting rate
Q_BETA = 0.05                   # cross-edge strength → splitting resistance
Q_GAMMA = 0.1                   # merging rate when union reduces dissonance
Q_EDGE_WEAKEN = 0.15            # how much to weaken cross-edges on dissonance

# ── Rumination limiter ──
RUMINATION_THETA_MIN = 0.01      # minimum LP for "productive" reflection
RUMINATION_MAX_CYCLES = 50       # max cycles without progress before force switch
RUMINATION_DECAY_FACTOR = 0.3    # SIT multiplier on archival
RUMINATION_E_MAX = 1.0           # maximum energy budget
RUMINATION_E_CRITICAL = 0.1      # threshold for emergency BLEND
RUMINATION_COST = 0.02           # energy cost per rumination cycle

# ── Action loop ──
ACTION_SIT_THRESHOLD = 0.001     # minimum SIT to trigger action
ACTION_CL_THRESHOLD = 0.03      # minimum CL to enable planning
ACTION_CLOSURE_BOOST = 0.08     # closure increase per successful action
ACTION_AHA_THRESHOLD = 0.95     # closure level that triggers AHA moment
ACTION_AHA_REWARD = 0.3         # activation boost to SEEKING on AHA
ACTION_ENERGY_COST = 0.005      # energy cost per action (≈ recovery rate)

# ── κ consolidation levels (NM Part VIII, AGI_F Part III) ──
KAPPA_ENABLED = True             # feature switch
N_CRIT = 5                       # co-activation threshold for κ: 1→2
F_LTM = 0.8                     # fidelity shortcut for κ: 1→2 (hub fast-track)
THETA_G_CONS = 0.6              # G-alignment threshold for emotional consolidation
KAPPA_DECAY_MULT = {0: 10.0, 1: 1.0, 2: 0.1}  # κ-dependent edge decay multiplier

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
