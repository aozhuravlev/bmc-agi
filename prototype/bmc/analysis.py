"""Structural analysis: degree metrics, structural balance, percolation."""
import numpy as np
import networkx as nx


def compute_degree_metrics(G):
    """Degree distribution, hub dominance H, modularity Q, small-worldness sigma."""
    meme_nodes = [n for n in G.nodes() if G.nodes[n].get('layer') == 'memetic']
    meme_degrees_raw = np.array([G.degree(n) for n in meme_nodes])
    meme_degrees_sorted = sorted(meme_degrees_raw, reverse=True)

    # Hub dominance
    k_max = max(meme_degrees_raw)
    k_mean = np.mean(meme_degrees_raw)
    H = k_max / k_mean

    # Modularity (Louvain)
    meme_subgraph = G.subgraph(meme_nodes).copy()
    communities = nx.community.louvain_communities(meme_subgraph, seed=42)
    Q = nx.community.modularity(meme_subgraph, communities)
    comm_sizes = sorted([len(c) for c in communities], reverse=True)

    # Small-worldness
    if nx.is_connected(meme_subgraph):
        sub = meme_subgraph
    else:
        largest_cc = max(nx.connected_components(meme_subgraph), key=len)
        sub = meme_subgraph.subgraph(largest_cc)

    L_actual = nx.average_shortest_path_length(sub)
    C_actual = nx.average_clustering(sub)
    n_nodes = sub.number_of_nodes()
    n_edges = sub.number_of_edges()
    p_er = 2 * n_edges / (n_nodes * (n_nodes - 1))

    _L_samples, _C_samples = [], []
    for _s in range(100):
        _er = nx.erdos_renyi_graph(n_nodes, p_er, seed=_s)
        if nx.is_connected(_er):
            _L_samples.append(nx.average_shortest_path_length(_er))
            _C_samples.append(nx.average_clustering(_er))
    L_random = np.mean(_L_samples) if _L_samples else L_actual
    C_random = np.mean(_C_samples) if _C_samples else p_er
    sigma = (C_actual / max(C_random, 1e-6)) / (L_actual / max(L_random, 1e-6))

    # Top hubs
    top5_idx = np.argsort(meme_degrees_raw)[-5:][::-1]
    top_hubs = [meme_nodes[i] for i in top5_idx]

    # Assortativity
    r = nx.degree_assortativity_coefficient(meme_subgraph)

    # Triangles
    tri_per_node = nx.triangles(meme_subgraph)
    n_triangles = sum(tri_per_node.values()) // 3
    n_er = meme_subgraph.number_of_nodes()
    m_er = meme_subgraph.number_of_edges()
    p_er_graph = 2 * m_er / (n_er * (n_er - 1))
    er_expected_triangles = int((n_er * (n_er-1) * (n_er-2) / 6) * p_er_graph**3)
    er_graph = nx.erdos_renyi_graph(n_er, p_er_graph, seed=42)
    er_triangles = sum(nx.triangles(er_graph).values()) // 3

    return {
        'meme_nodes': meme_nodes,
        'meme_degrees_raw': meme_degrees_raw,
        'meme_degrees_sorted': meme_degrees_sorted,
        'H': H, 'k_max': k_max, 'k_mean': k_mean,
        'Q': Q, 'communities': communities, 'comm_sizes': comm_sizes,
        'sigma': sigma,
        'L_actual': L_actual, 'C_actual': C_actual,
        'L_random': L_random, 'C_random': C_random,
        'top_hubs': top_hubs,
        'r': r,
        'n_triangles': n_triangles,
        'er_triangles': er_triangles,
        'er_expected_triangles': er_expected_triangles,
        'n_nodes': n_nodes, 'p_er': p_er_graph,
    }


def detect_structural_balance(G, edge_weights=None):
    """Analyze structural balance in a signed network.
    Counts balanced/unbalanced triangles (product of edge signs)."""
    meme_nodes_set = set(n for n in G.nodes() if G.nodes[n].get('layer') == 'memetic')
    meme_sub = G.subgraph(meme_nodes_set)

    balanced = 0
    unbalanced = 0

    for n in meme_sub.nodes():
        neighbors = list(meme_sub.neighbors(n))
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                ni, nj = neighbors[i], neighbors[j]
                if meme_sub.has_edge(ni, nj):
                    if edge_weights:
                        w1 = edge_weights.get((n, ni), G.edges[n, ni].get('weight', 0.5))
                        w2 = edge_weights.get((n, nj), G.edges[n, nj].get('weight', 0.5))
                        w3 = edge_weights.get((ni, nj), G.edges[ni, nj].get('weight', 0.5))
                    else:
                        w1 = G.edges[n, ni].get('weight', 0.5)
                        w2 = G.edges[n, nj].get('weight', 0.5)
                        w3 = G.edges[ni, nj].get('weight', 0.5)

                    s1 = 1 if w1 >= 0 else -1
                    s2 = 1 if w2 >= 0 else -1
                    s3 = 1 if w3 >= 0 else -1
                    product = s1 * s2 * s3

                    if product > 0:
                        balanced += 1
                    else:
                        unbalanced += 1

    balanced //= 3
    unbalanced //= 3
    total = balanced + unbalanced

    if total > 0:
        ratio = balanced / total
    else:
        ratio = 1.0

    if ratio > 0.9:
        classification = 'Strict balance'
    elif ratio > 0.6:
        classification = 'Weak balance'
    else:
        classification = 'Unbalanced'

    return {
        'balanced': balanced,
        'unbalanced': unbalanced,
        'total': total,
        'ratio': ratio,
        'classification': classification,
    }


def gcc_size(graph):
    """Relative size of the giant connected component."""
    if graph.number_of_nodes() == 0:
        return 0
    return len(max(nx.connected_components(graph), key=len))


def run_percolation(G, fractions=None):
    """Random failure vs targeted attack on positive subgraph."""
    if fractions is None:
        fractions = np.linspace(0, 0.9, 30)

    meme_nodes = [n for n in G.nodes() if G.nodes[n].get('layer') == 'memetic']
    meme_sub_full = G.subgraph(meme_nodes).copy()

    # Build positive-only subgraph
    pos_edges = [(u, v) for u, v, d in meme_sub_full.edges(data=True)
                 if d.get('weight', 0) >= 0]
    meme_sub_perc = nx.Graph()
    meme_sub_perc.add_nodes_from(meme_sub_full.nodes(data=True))
    for u, v in pos_edges:
        meme_sub_perc.add_edge(u, v, **meme_sub_full.edges[u, v])

    n_neg_removed = meme_sub_full.number_of_edges() - meme_sub_perc.number_of_edges()
    degrees_perc = np.array([meme_sub_perc.degree(n) for n in meme_sub_perc.nodes()])

    # Molloy-Reed criterion
    k_mean = degrees_perc.mean()
    k2_mean = (degrees_perc**2).mean()
    kappa = k2_mean / k_mean if k_mean > 0 else 0
    f_c = 1 - 1 / (kappa - 1) if kappa > 2 else float('nan')

    node_list = list(meme_sub_perc.nodes())
    n_total = len(node_list)

    # (a) Random removal
    np.random.seed(123)
    random_order = np.random.permutation(node_list)
    gcc_random = []
    for f in fractions:
        n_remove = int(f * n_total)
        H = meme_sub_perc.copy()
        H.remove_nodes_from(random_order[:n_remove])
        gcc_random.append(gcc_size(H) / n_total if H.number_of_nodes() > 0 else 0)

    # (b) Targeted removal (highest degree first)
    degree_order = sorted(node_list, key=lambda n: meme_sub_perc.degree(n), reverse=True)
    gcc_targeted = []
    for f in fractions:
        n_remove = int(f * n_total)
        H = meme_sub_perc.copy()
        H.remove_nodes_from(degree_order[:n_remove])
        gcc_targeted.append(gcc_size(H) / n_total if H.number_of_nodes() > 0 else 0)

    return {
        'fractions': fractions,
        'gcc_random': gcc_random,
        'gcc_targeted': gcc_targeted,
        'f_c': f_c,
        'k_mean': k_mean,
        'k2_mean': k2_mean,
        'kappa': kappa,
        'n_pos_edges': len(pos_edges),
        'n_neg_removed': n_neg_removed,
        'n_nodes': n_total,
    }
