"""Visualization functions for BMC simulations."""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def setup_style():
    """Configure matplotlib dark_background style."""
    plt.style.use('dark_background')


def visualize_bmc(sim, title='BMC Structure', step_idx=-1):
    """Two-level graph: utility on top, memes below.
    Signed edges: green(+) solid, red(-) dashed. Thickness = |w|."""
    fig, ax = plt.subplots(figsize=(12, 9))

    G = sim.G
    if sim.history:
        acts = sim.history[step_idx]['activations']
    else:
        acts = sim.activations

    # Layout: spring for memes, fixed row for utility
    meme_sub = G.subgraph(sim.meme_nodes)
    pos_meme = nx.spring_layout(meme_sub, k=1.8, iterations=80, seed=42)
    for n in pos_meme:
        pos_meme[n][1] -= 0.3
    pos = dict(pos_meme)
    n_u = len(sim.utility_nodes)
    for i, u in enumerate(sim.utility_nodes):
        pos[u] = np.array([(i - n_u/2) * 0.30, 1.5])

    degrees = dict(G.degree())

    # ── Draw meme-meme edges (signed colors) ──
    meme_edges = [(u, v) for u, v in G.edges() if u in sim.meme_nodes and v in sim.meme_nodes]
    pos_edges = [(u, v) for u, v in meme_edges if sim.edge_weights.get((u, v), 0) >= 0]
    neg_edges = [(u, v) for u, v in meme_edges if sim.edge_weights.get((u, v), 0) < 0]

    if pos_edges:
        pos_widths = [max(0.3, abs(sim.edge_weights.get((u, v), 0.3)) * 2) for u, v in pos_edges]
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, alpha=0.2,
                               edge_color='#2ca02c', width=pos_widths, ax=ax)
    if neg_edges:
        neg_widths = [max(0.3, abs(sim.edge_weights.get((u, v), 0.3)) * 3) for u, v in neg_edges]
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges, alpha=0.4,
                               edge_color='#d62728', width=neg_widths,
                               style='dashed', ax=ax)

    # Utility-meme edges
    um_edges = [(u, v) for u, v in G.edges()
                if (u in sim.utility_nodes) != (v in sim.utility_nodes)]
    um_colors = []
    for u, v in um_edges:
        etype = G.edges[u, v].get('etype', 'redirect')
        if etype == 'suppress': um_colors.append('#d62728')
        elif etype == 'interpret': um_colors.append('#9467bd')
        else: um_colors.append('#7f7f7f')
    nx.draw_networkx_edges(G, pos, edgelist=um_edges, alpha=0.3,
                           edge_color=um_colors, style='dashed', ax=ax)

    # ── Draw meme nodes ──
    meme_colors = [acts.get(m, 0) for m in sim.meme_nodes]
    meme_sizes = [200 + degrees.get(m, 1) * 60 for m in sim.meme_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=sim.meme_nodes,
                           node_size=meme_sizes, node_color=meme_colors,
                           cmap=plt.cm.YlOrRd, vmin=0, vmax=1, ax=ax)

    # ── Draw utility nodes ──
    util_colors = [acts.get(u, 0) for u in sim.utility_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=sim.utility_nodes,
                           node_size=900, node_color=util_colors,
                           cmap=plt.cm.Blues, vmin=0, vmax=1,
                           node_shape='s', ax=ax)

    # Labels for utility
    util_labels = {u: u.replace('_', '\n') for u in sim.utility_nodes}
    nx.draw_networkx_labels(G, pos, labels=util_labels,
                            font_size=6, font_color='white', ax=ax)

    # Labels for top hubs (degree > 5)
    hub_labels = {m: m.replace('_', '\n') for m in sim.meme_nodes if degrees.get(m, 0) >= 5}
    nx.draw_networkx_labels(G, pos, labels=hub_labels,
                            font_size=6, font_color='red', ax=ax)

    # ── Legend ──
    ax.text(0.02, 0.98, 'Utility (Panksepp systems)', transform=ax.transAxes,
            fontsize=10, color='#6baed6', va='top')
    ax.text(0.02, 0.95, 'Meme (size=degree, color=activation)', transform=ax.transAxes,
            fontsize=10, color='#fc8d59', va='top')
    ax.text(0.02, 0.92, 'Green solid = positive edge, Red dashed = negative edge',
            transform=ax.transAxes, fontsize=9, color='#aaa', va='top')

    if sim.history:
        h = sim.history[step_idx]
        info = (f"Balance={h['balance']:.2f}  Conflict={h['conflict']:.2f}  "
                f"Fatigue={h['fatigue']:.2f}  "
                f"Ambivalence={h.get('mean_ambivalence', 0):.3f}  "
                f"Active={h['states']['active']}  "
                f"Decorative={h['states']['decorative']}  "
                f"Sleeping={h['states']['sleeping']}")
        ax.set_xlabel(info, fontsize=10)

    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    return fig


def plot_scenario(sim, title):
    """Plot all metrics for a completed simulation (including ambivalence)."""
    h = sim.history
    steps = range(len(h))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    # (a) Layer activations
    ax = axes[0, 0]
    ax.plot(steps, [x['A_meme'] for x in h], label='A_meme (mean)', color='#fc8d59', lw=2)
    ax.plot(steps, [x['A_utility'] for x in h], label='A_utility (mean)', color='#6baed6', lw=2)
    ax.set_ylabel('Mean Activation'); ax.set_title('Layer Activations')
    ax.legend(); ax.grid(alpha=0.3)

    # (b) Balance
    ax = axes[0, 1]
    bal = [x['balance'] for x in h]
    ax.plot(steps, bal, color='#98df8a', lw=2)
    ax.axhline(1.0, color='white', ls='--', alpha=0.4, label='Equilibrium')
    ax.fill_between(steps, bal, 1.0, alpha=0.2,
                    color='#ff7f7f', where=[b < 1 for b in bal])
    ax.fill_between(steps, bal, 1.0, alpha=0.2,
                    color='#7fbfff', where=[b >= 1 for b in bal])
    ax.set_ylabel('Balance (A_meme / A_utility)'); ax.set_title('Balance(t)')
    ax.legend(); ax.grid(alpha=0.3)

    # (c) Conflict, Dissonance & Tension
    ax = axes[0, 2]
    conf = [x['conflict'] for x in h]
    diss = [x['dissonance'] for x in h]
    tens = [x['tension'] for x in h]
    ax.plot(steps, conf, color='#d62728', lw=2, label='Conflict')
    ax.plot(steps, diss, color='#9467bd', lw=2, label='Dissonance', ls='--')
    ax.fill_between(steps, conf, alpha=0.2, color='#ff7f7f')
    ax.fill_between(steps, diss, alpha=0.2, color='#c5b0d5')
    ax.plot(steps, tens, color='#17becf', lw=2, label='Tension', ls='-.')
    ax.fill_between(steps, tens, alpha=0.1, color='#17becf')
    ax.set_ylabel('Conflict / Dissonance / Tension'); ax.set_title('Conflict, Dissonance & Tension')
    ax.legend(); ax.grid(alpha=0.3)

    # (d) Stability (set-based + weighted)
    ax = axes[1, 0]
    stab = [x['stability'] for x in h]
    stab_w = [x.get('wm_stability_weighted', 1.0) for x in h]
    ax.plot(steps, stab, color='#2ca02c', lw=2, label='Set-based')
    ax.plot(steps, stab_w, color='cyan', lw=2, ls='--', label='Weighted (cosine)')
    ax.set_ylabel('Stability'); ax.set_title('Self-Stability(t)')
    ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(alpha=0.3)

    # (e) Fatigue + Ambivalence
    ax = axes[1, 1]
    fat = [x['fatigue'] for x in h]
    amb = [x.get('mean_ambivalence', 0) for x in h]
    ax.plot(steps, fat, color='#ff7f0e', lw=2, label='Fatigue')
    ax.plot(steps, amb, color='#e377c2', lw=2, ls='--', label='Ambivalence')
    ax.set_ylabel('Value'); ax.set_title('Fatigue & Ambivalence')
    ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(alpha=0.3)

    # (f) Meme state counts
    ax = axes[1, 2]
    ax.stackplot(steps,
                 [x['states']['active'] for x in h],
                 [x['states']['decorative'] for x in h],
                 [x['states']['sleeping'] for x in h],
                 labels=['Active', 'Decorative', 'Sleeping'],
                 colors=['#d62728', '#ff7f0e', '#1f77b4'], alpha=0.7)
    ax.set_ylabel('Meme count'); ax.set_title('Meme States')
    ax.legend(loc='upper right'); ax.grid(alpha=0.3)

    for a in axes.flat:
        a.set_xlabel('Time Step')

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_structural_balance(result):
    """Bar chart of balanced vs unbalanced triangles."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if result['total'] > 0:
        ax.bar(['Balanced', 'Unbalanced'], [result['balanced'], result['unbalanced']],
               color=['#2ca02c', '#d62728'], alpha=0.8, edgecolor='white')
        ax.set_ylabel('Triangle count')
        ax.set_title(f"Structural Balance: {result['classification']} (ratio={result['ratio']:.2f})")
        ax.grid(alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No triangles with signed edges found',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Structural Balance')
    plt.tight_layout()
    return fig


def plot_degree_distribution(metrics):
    """Three-panel plot: degree distribution, communities, small-worldness."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) Degree distribution — log-log
    ax = axes[0]
    unique_degs, counts = np.unique(metrics['meme_degrees_sorted'], return_counts=True)
    ax.scatter(unique_degs, counts, color='#fc8d59', s=60, zorder=5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Degree k'); ax.set_ylabel('Count')
    ax.set_title('Degree Distribution (meme layer)')
    ax.grid(alpha=0.3)
    ax.text(0.95, 0.95, f'H = k_max/<k> = {metrics["H"]:.1f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(facecolor='black', alpha=0.7))

    # (b) Modularity (Louvain)
    ax = axes[1]
    ax.bar(range(len(metrics['comm_sizes'])), metrics['comm_sizes'], color='#6baed6')
    ax.set_xlabel('Community'); ax.set_ylabel('Size')
    ax.set_title(f'Louvain communities (Q = {metrics["Q"]:.3f})')
    ax.grid(alpha=0.3)

    # (c) Small-worldness
    ax = axes[2]
    labels = ['C_actual', 'C_random', 'L_actual', 'L_random']
    values = [metrics['C_actual'], metrics['C_random'],
              metrics['L_actual'], metrics['L_random']]
    colors_bar = ['#fc8d59', '#aaa', '#6baed6', '#aaa']
    ax.bar(labels, values, color=colors_bar)
    ax.set_title(f'Small-worldness (sigma = {metrics["sigma"]:.2f})')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_percolation(result):
    """Percolation curves: random failure vs targeted attack."""
    fig, ax = plt.subplots(figsize=(7, 4))
    fractions = result['fractions']
    ax.plot(fractions, result['gcc_random'], 'o-', color='#6baed6',
            label='Random failure', lw=2, ms=4)
    ax.plot(fractions, result['gcc_targeted'], 's-', color='#d62728',
            label='Targeted attack (hubs first)', lw=2, ms=4)
    f_c = result['f_c']
    if not np.isnan(f_c):
        ax.axvline(f_c, color='yellow', ls='--', alpha=0.6,
                   label=f'Theoretical f_c = {f_c:.2f}')
    ax.set_xlabel('Fraction of nodes removed')
    ax.set_ylabel('GCC / N (relative giant component)')
    ax.set_title('Percolation on Positive Subgraph: Robustness vs Fragility')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_immune_response(sim5a, sim5b, score_a, score_b, accepted_a, accepted_b):
    """S5: 6-panel comparison of compatible vs incompatible meme introduction."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    for col, (sim5, label, meme_name, score_val, was_accepted) in enumerate([
        (sim5a, '5a: Compatible (Critical_thinking)', 'Critical_thinking', score_a, accepted_a),
        (sim5b, '5b: Incompatible (Flat_earth)', 'Flat_earth', score_b, accepted_b),
    ]):
        h = sim5.history
        steps = range(len(h))

        # Conflict & Dissonance
        ax = axes[0, col]
        conf = [x['conflict'] for x in h]
        diss = [x['dissonance'] for x in h]
        ax.plot(steps, conf, color='#d62728', lw=2, label='Conflict')
        ax.plot(steps, diss, color='#9467bd', lw=2, label='Dissonance', ls='--')
        ax.fill_between(steps, conf, alpha=0.2, color='#ff7f7f')
        ax.fill_between(steps, diss, alpha=0.2, color='#c5b0d5')
        tens = [x['tension'] for x in h]
        ax.plot(steps, tens, color='#17becf', lw=2, label='Tension', ls='-.')
        ax.fill_between(steps, tens, alpha=0.1, color='#17becf')
        ax.axvline(15, color='yellow', ls='--', alpha=0.5, label='Meme introduced')
        ax.set_ylabel('Conflict / Diss. / Tension')
        ax.set_title(f'{label}: Conflict, Dissonance & Tension')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Meme activation + DISGUST + FEAR
        ax = axes[1, col]
        disgust_act = [x['activations'].get('DISGUST', 0) for x in h]
        fear_act = [x['activations'].get('FEAR', 0) for x in h]
        ax.plot(steps, disgust_act, color='#9467bd', lw=2, label='DISGUST', ls='--')
        ax.plot(steps, fear_act, color='#ff7f0e', lw=1, label='FEAR', ls=':')
        if meme_name in sim5.activations:
            meme_act = [x['activations'].get(meme_name, 0) for x in h]
            ax.plot(steps, meme_act, color='#2ca02c', lw=2, label=meme_name)
        else:
            ax.axhline(0, color='#d62728', ls=':', alpha=0.5,
                       label=f'{meme_name} (rejected)')
            ax.annotate(f'REJECTED\nS(X) = {score_val:.2f}',
                        xy=(15, 0), xytext=(25, -0.3),
                        fontsize=11, color='#ff4444', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#ff4444', lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a0000',
                                  edgecolor='#ff4444', alpha=0.8))
        ax.axvline(15, color='yellow', ls='--', alpha=0.5)
        ymin = -0.5 if not was_accepted else -0.05
        ax.set_ylim(ymin, 1.1)
        ax.set_ylabel('Activation / Score')
        ax.set_title(f'{label}: Key Activations')
        ax.set_xlabel('Time Step'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # S(X) score comparison
    ax = axes[0, 2]
    bars = ax.bar(['Critical_thinking\n(compatible)', 'Flat_earth\n(incompatible)'],
                  [score_a, score_b],
                  color=['#2ca02c', '#d62728'], alpha=0.8, edgecolor='white', width=0.5)
    ax.axhline(0, color='white', ls='-', alpha=0.6, lw=1)
    ax.set_ylabel('Compatibility Score S(X)')
    ax.set_title('Immune Response: S(X) Scores')
    for bar, val, acc in zip(bars, [score_a, score_b], [accepted_a, accepted_b]):
        lbl = 'ACCEPTED' if acc else 'REJECTED'
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, -0.08, f'{lbl}\n{val:.2f}',
                    ha='center', va='top', fontweight='bold', fontsize=11, color='#2ca02c')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 0.08, f'{lbl}\n{val:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11, color='#ff4444')
    ax.grid(alpha=0.3, axis='y')

    # Balance comparison
    ax = axes[1, 2]
    bal_a = [x['balance'] for x in sim5a.history]
    bal_b = [x['balance'] for x in sim5b.history]
    ax.plot(range(len(bal_a)), bal_a, color='#2ca02c', lw=2, label='5a: Compatible')
    ax.plot(range(len(bal_b)), bal_b, color='#d62728', lw=2, label='5b: Incompatible')
    ax.axhline(1.0, color='white', ls='--', alpha=0.3)
    ax.axvline(15, color='yellow', ls='--', alpha=0.5, label='Meme introduced')
    ax.set_xlabel('Time Step'); ax.set_ylabel('Balance')
    ax.set_title('Balance Comparison')
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle('Scenario 5: Immune Response to Foreign Memes', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_hub_displacement(sim, top_hub, alt_name, top_hub_meme_neighbors,
                          wd_before, wd_total_before, wd_after, wd_total_after,
                          alt_intro=25):
    """S6: 3-panel hub displacement visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    h = sim.history
    steps = range(len(h))

    # (a) Activations over time
    ax = axes[0]
    hub_act = [x['activations'].get(top_hub, 0) for x in h]
    alt_act = [x['activations'].get(alt_name, 0) for x in h]
    ax.plot(steps, hub_act, color='#d62728', lw=2, label=f'{top_hub} (original)')
    ax.plot(steps, alt_act, color='#2ca02c', lw=2, label=f'{alt_name[:20]}... (new)')
    ax.axvline(10, color='yellow', ls='--', alpha=0.4, label='Crisis starts')
    ax.axvline(alt_intro, color='cyan', ls='--', alpha=0.4, label='Alternative introduced')
    ax.set_xlabel('Time Step'); ax.set_ylabel('Activation')
    ax.set_title('Hub Displacement: Activation Dynamics')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) Weighted degree share: before vs after
    ax = axes[1]
    key_nodes = [top_hub, alt_name] + top_hub_meme_neighbors[:4]
    key_nodes = [n for n in key_nodes if n in wd_after]
    ec_b = [100 * wd_before.get(n, 0) / wd_total_before for n in key_nodes]
    ec_a = [100 * wd_after.get(n, 0) / wd_total_after for n in key_nodes]
    x_pos = np.arange(len(key_nodes))
    width = 0.35
    ax.bar(x_pos - width/2, ec_b, width, label='Before', color='#6baed6', alpha=0.8)
    ax.bar(x_pos + width/2, ec_a, width, label='After', color='#fc8d59', alpha=0.8)
    ax.set_xticks(x_pos)
    short_labels = [n[:12] for n in key_nodes]
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('WD share (%)')
    ax.set_title('Weighted Degree Share (decay-normalized)')
    ax.legend(); ax.grid(alpha=0.3)

    # (c) Edge weight transfer
    ax = axes[2]
    old_weights_t = []
    new_weights_t = []
    for step_h in h:
        ew = step_h['edge_weights']
        old_sum = sum(abs(ew.get((top_hub, nb), 0)) for nb in top_hub_meme_neighbors)
        new_sum = sum(abs(ew.get((alt_name, nb), 0)) for nb in top_hub_meme_neighbors[:5])
        old_weights_t.append(old_sum)
        new_weights_t.append(new_sum)
    ax.plot(steps, old_weights_t, color='#d62728', lw=2, label=f'{top_hub} edge sum')
    ax.plot(steps, new_weights_t, color='#2ca02c', lw=2, label=f'Alternative edge sum')
    ax.axvline(alt_intro, color='cyan', ls='--', alpha=0.4, label='Alternative introduced')
    ax.set_xlabel('Time Step'); ax.set_ylabel('Sum of |edge weights|')
    ax.set_title('Edge Weight Transfer')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle('Scenario 6: Hub Displacement (Destabilization)', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig, key_nodes


def plot_edge_decay(central_weights, peripheral_weights, negative_weights,
                    peripheral_weights_react, central_edge, peripheral_edge,
                    neg_edge, reactivation_steps):
    """Edge decay demo: central vs peripheral vs negative, with spaced repetition."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(central_weights,
            label=f'Central: {central_edge[0][:12]}--{central_edge[1][:12]}',
            color='#fc8d59', lw=2)
    ax.plot(peripheral_weights,
            label=f'Peripheral: {peripheral_edge[0][:12]}--{peripheral_edge[1][:12]}',
            color='#6baed6', lw=2)
    ax.plot(negative_weights,
            label=f'Negative: {neg_edge[0][:12]}--{neg_edge[1][:12]}',
            color='#d62728', lw=2, ls='--')
    ax.axhline(0, color='white', ls=':', alpha=0.3)
    ax.set_xlabel('Time Step'); ax.set_ylabel('Edge Weight')
    ax.set_title('Edge Decay: Central vs Peripheral vs Negative')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(peripheral_weights, label='No reactivation', color='#6baed6', lw=2, ls='--')
    ax.plot(peripheral_weights_react, label='Spaced repetition', color='#2ca02c', lw=2)
    for rs in reactivation_steps:
        ax.axvline(rs, color='yellow', alpha=0.4, ls=':')
    ax.set_xlabel('Time Step'); ax.set_ylabel('Edge Weight')
    ax.set_title('Spaced Repetition vs Forgetting')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_sleeper_effect(weight_trajectories, act_trajectory, amb_trajectory,
                        disgust_trajectory, target_edges):
    """S8: 4-panel sleeper effect visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    steps = range(len(act_trajectory))

    # (a) Edge weight trajectories
    ax = axes[0, 0]
    colors = ['#d62728', '#ff7f0e', '#9467bd']
    for (e, traj), c in zip(weight_trajectories.items(), colors):
        ax.plot(steps, traj, lw=2, label=f'\u2192{e[1][:12]}', color=c)
    ax.axhline(0, color='white', ls=':', alpha=0.4)
    ax.axvline(40, color='cyan', ls='--', alpha=0.5, label='Re-exposure')
    ax.axvline(60, color='yellow', ls='--', alpha=0.3, label='Settle')
    ax.set_xlabel('Time Step'); ax.set_ylabel('Edge Weight')
    ax.set_title('Sleeper Effect: Edge Weight Trajectory')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) Meme activation
    ax = axes[0, 1]
    ax.plot(steps, act_trajectory, color='#2ca02c', lw=2, label='Alternative_medicine')
    ax.plot(steps, disgust_trajectory, color='#9467bd', lw=2, ls='--', label='DISGUST')
    ax.axvline(40, color='cyan', ls='--', alpha=0.5, label='Re-exposure')
    ax.axvline(60, color='yellow', ls='--', alpha=0.3)
    ax.set_xlabel('Time Step'); ax.set_ylabel('Activation')
    ax.set_title('Sleeper Effect: Activation Trajectory')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) Ambivalence
    ax = axes[1, 0]
    ax.plot(steps, amb_trajectory, color='#e377c2', lw=2)
    ax.axvline(40, color='cyan', ls='--', alpha=0.5, label='Re-exposure')
    ax.set_xlabel('Time Step'); ax.set_ylabel('Mean Ambivalence')
    ax.set_title('Ambivalence During Sleeper Effect')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) Summary bar: initial vs final edge weights
    ax = axes[1, 1]
    labels = [e[1][:12] for e in target_edges]
    initial_w = [weight_trajectories[e][0] for e in target_edges]
    final_w = [weight_trajectories[e][-1] for e in target_edges]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, initial_w, width, label='Initial (rejected)',
           color='#d62728', alpha=0.8)
    ax.bar(x + width/2, final_w, width, label='Final (after sleeper)',
           color='#2ca02c', alpha=0.8)
    ax.axhline(0, color='white', ls='-', alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Edge Weight')
    ax.set_title('Edge Weight: Before vs After Sleeper Effect')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    fig.suptitle('Scenario 8: Sleeper Effect (negative \u2192 neutral \u2192 positive)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_filmstrip(sim, keyframes, title):
    """Static filmstrip: metrics at key timepoints."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    metrics_keys = ['A_meme', 'A_utility', 'conflict', 'fatigue', 'balance']
    colors = {'A_meme': '#fc8d59', 'A_utility': '#6baed6', 'conflict': '#d62728',
              'fatigue': '#ff7f0e', 'balance': '#2ca02c'}

    n_steps = len(sim.history)
    for idx, (ax, t) in enumerate(zip(axes.flat, keyframes)):
        h = sim.history[t]
        ts = list(range(t + 1))
        for mk in metrics_keys:
            vals = [sim.history[i].get(mk, 0) for i in ts]
            ax.plot(ts, vals, label=mk, color=colors[mk], lw=1.5,
                    ls='--' if mk == 'conflict' else (':'  if mk == 'fatigue' else '-'))
        ax.set_title(f't={t}  bal={h["balance"]:.2f}  fat={h["fatigue"]:.3f}', fontsize=9)
        ax.set_xlim(0, n_steps)
        ax.set_ylim(0, max(0.5, max(x['balance'] for x in sim.history) * 0.5))
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper left')

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig
