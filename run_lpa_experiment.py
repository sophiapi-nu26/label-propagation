import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os, time
import itertools
import multiprocessing
from tqdm import tqdm
import argparse

###############################################################################
#  1) FASTER SBM GENERATION (VECTORIZED NUMPY)
###############################################################################
def generate_sbm_2_community_fast(n, p, q):
    """
    Vectorized 2-community balanced SBM (undirected, no self-loops).
    Community sizes = n1 and n2 = n - n1.
    """
    n1 = n // 2
    n2 = n - n1
    
    # Create empty adjacency (False = no edge)
    A = np.zeros((n, n), dtype=bool)
    
    # Block for community 0 (n1 x n1)
    block0 = (np.random.rand(n1, n1) < p)
    block0 = np.triu(block0, 1)
    block0 = block0 | block0.T
    A[:n1, :n1] = block0
    
    # Block for community 1 (n2 x n2)
    block1 = (np.random.rand(n2, n2) < p)
    block1 = np.triu(block1, 1)
    block1 = block1 | block1.T
    A[n1:, n1:] = block1
    
    # Cross (n1 x n2) edges
    cross = (np.random.rand(n1, n2) < q)
    A[:n1, n1:] = cross
    A[n1:, :n1] = cross.T
    
    # Remove self-loops
    np.fill_diagonal(A, False)
    
    # Build a NetworkX Graph
    G = nx.from_numpy_array(A)
    
    # Assign community labels as node attributes
    for node in range(n1):
        G.nodes[node]['community'] = 0
    for node in range(n1, n):
        G.nodes[node]['community'] = 1
    
    return G

###############################################################################
#  2) INITIALIZE LABELS
###############################################################################
def initialize_labels(n, G):
    """
    Assign each node a unique label 1..n.
    Force label=1 to be on some node in community=0.
    Return (labels_dict, smallest_label_comm_0, smallest_label_comm_1).
    """
    labels = {}
    nodes_community_0 = [node for node in G.nodes if G.nodes[node]['community'] == 0]
    nodes_community_1 = [node for node in G.nodes if G.nodes[node]['community'] == 1]

    chosen_for_label_1 = np.random.choice(nodes_community_0)
    all_nodes = list(range(n))
    np.random.shuffle(all_nodes)
    
    # Move chosen_for_label_1 to the front
    if all_nodes[0] != chosen_for_label_1:
        idx = all_nodes.index(chosen_for_label_1)
        all_nodes[idx], all_nodes[0] = all_nodes[0], all_nodes[idx]

    for i, node in enumerate(all_nodes):
        labels[node] = i + 1

    smallest_label_comm_0 = min(labels[node] for node in nodes_community_0)
    smallest_label_comm_1 = min(labels[node] for node in nodes_community_1)
    return labels, smallest_label_comm_0, smallest_label_comm_1

###############################################################################
#  3) LABEL PROPAGATION (using adjacency lists or G.neighbors)
###############################################################################
def label_propagation_round(G, current_labels, round_idx):
    """
    At round=0: each node takes the minimum label among neighbors + itself.
    For subsequent rounds: pick the most frequent label among neighbors + itself,
    tie-break on smaller label.
    """
    next_labels = {}
    for node in G.nodes():
        neighbor_labels = [current_labels[nbr] for nbr in G.neighbors(node)]
        neighbor_labels.append(current_labels[node])  # include itself

        if round_idx == 0:
            next_labels[node] = min(neighbor_labels)
        else:
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            max_count = np.max(counts)
            candidates = unique_labels[counts == max_count]
            next_labels[node] = np.min(candidates)  # tie-break
    return next_labels

def run_label_propagation(G, labels, rounds=5):
    """
    Run label propagation for `rounds` rounds.
    Returns a list of label dicts, one per round.
    """
    all_rounds_labels = []
    current_labels = labels.copy()
    for r in range(rounds):
        next_labels = label_propagation_round(G, current_labels, r)
        all_rounds_labels.append(next_labels)
        current_labels = next_labels
    return all_rounds_labels

###############################################################################
#  4) COMPUTE PROPERTIES
###############################################################################
def compute_properties(G, all_rounds_labels, smallest_label_comm_0, smallest_label_comm_1):
    """
    - conv_smallest_in_comm:
      fraction of nodes in each community with that community's "original" smallest label
    - conv_smallest_global:
      fraction of nodes in each community that have label=1
    - fraction_not_changed:
      fraction of nodes in each community that didn't change from previous round
    """
    rounds = len(all_rounds_labels)
    communities = [0, 1]
    comm_nodes = {}
    for c in communities:
        comm_nodes[c] = [node for node in G.nodes if G.nodes[node]['community'] == c]

    correct_label_comm = {0: smallest_label_comm_0, 1: smallest_label_comm_1}
    smallest_global_label = 1

    conv_smallest_in_comm = np.zeros((rounds, 2))
    conv_smallest_global  = np.zeros((rounds, 2))
    fraction_not_changed  = np.zeros((rounds, 2))

    prev_labels = all_rounds_labels[0]
    
    for r in range(rounds):
        current_labels = all_rounds_labels[r]
        
        # 1) Convergence to smallest label in community
        for c_idx, c in enumerate(communities):
            lbls_c = [current_labels[node] for node in comm_nodes[c]]
            frac_correct = np.mean([1 if lbl == correct_label_comm[c] else 0 for lbl in lbls_c])
            conv_smallest_in_comm[r, c_idx] = frac_correct

        # 2) Convergence to smallest global label
        for c_idx, c in enumerate(communities):
            lbls_c = [current_labels[node] for node in comm_nodes[c]]
            frac_global = np.mean([1 if lbl == smallest_global_label else 0 for lbl in lbls_c])
            conv_smallest_global[r, c_idx] = frac_global

        # 3) Fraction not changed
        if r == 0:
            fraction_not_changed[r, :] = 0.0
        else:
            for c_idx, c in enumerate(communities):
                nodes_c = comm_nodes[c]
                not_changed_count = sum(1 for nd in nodes_c if current_labels[nd] == prev_labels[nd])
                fraction_not_changed[r, c_idx] = not_changed_count / len(nodes_c)

        prev_labels = current_labels

    return {
        'conv_smallest_in_comm': conv_smallest_in_comm,
        'conv_smallest_global':  conv_smallest_global,
        'fraction_not_changed':  fraction_not_changed
    }

def compute_cross_label_distribution(G, all_rounds_labels, smallest_label_comm_0, smallest_label_comm_1):
    """
    Example "cross label distribution":
    For each round, for:
      - Comm0 with label=1
      - Comm0 with label=k (smallest_label_comm_1)
      - Comm1 with label=1
      - Comm1 with label=k (smallest_label_comm_1)
    """
    rounds = len(all_rounds_labels)
    comm0_nodes = [node for node in G.nodes if G.nodes[node]['community'] == 0]
    comm1_nodes = [node for node in G.nodes if G.nodes[node]['community'] == 1]

    results_per_round = np.zeros((rounds, 4))
    label1 = 1
    labelk = smallest_label_comm_1

    for r in range(rounds):
        current_labels = all_rounds_labels[r]

        frac_c0_label1 = np.mean([current_labels[n] == label1 for n in comm0_nodes])
        frac_c0_labelk = np.mean([current_labels[n] == labelk for n in comm0_nodes])
        frac_c1_label1 = np.mean([current_labels[n] == label1 for n in comm1_nodes])
        frac_c1_labelk = np.mean([current_labels[n] == labelk for n in comm1_nodes])

        results_per_round[r, 0] = frac_c0_label1
        results_per_round[r, 1] = frac_c0_labelk
        results_per_round[r, 2] = frac_c1_label1
        results_per_round[r, 3] = frac_c1_labelk

    return results_per_round

###############################################################################
#  5) PARALLEL WORKER: run_one_regime
#     This is the function we call in parallel for each (p,q).
###############################################################################
def run_one_regime(args):
    (i, j), p, q, n, rounds, trials, property_names = args
    
    # We'll accumulate sums for each property over 'trials'
    accum_props = {prop: np.zeros((rounds, 2)) for prop in property_names}
    accum_cross = np.zeros((rounds, 4))

    for _ in range(trials):
        # 1) Generate
        G = generate_sbm_2_community_fast(n, p, q)
        # 2) Initialize labels
        labels, smallest_label_comm_0, smallest_label_comm_1 = initialize_labels(n, G)
        # 3) Run LPA
        all_rounds_labels = run_label_propagation(G, labels, rounds=rounds)
        # 4) Compute properties
        props = compute_properties(G, all_rounds_labels, smallest_label_comm_0, smallest_label_comm_1)
        cross_dist = compute_cross_label_distribution(G, all_rounds_labels,
                                                      smallest_label_comm_0, smallest_label_comm_1)
        
        for prop in property_names:
            accum_props[prop] += props[prop]
        accum_cross += cross_dist

    # Average over trials
    for prop in property_names:
        accum_props[prop] /= trials
    accum_cross /= trials
    
    return (i, j, accum_props, accum_cross)

###############################################################################
#  6) MAIN PARALLEL EXPERIMENT
###############################################################################
def run_experiment_parallel(num_params=64,  # for 64^2 grid
                           n=1000,
                           rounds=5,
                           trials=5,
                           property_names=('conv_smallest_in_comm', 
                                           'conv_smallest_global', 
                                           'fraction_not_changed'),
                           num_cores=10):
    """
    - Creates a 64x64 grid of (p,q).
    - Parallelizes over 'num_cores' CPU cores.
    - Collects results in 'results' dict.
    - Plots final heatmaps as PNGs in the current directory.
    - Shows a tqdm progress bar over the entire set of (p,q) tasks.
    """
    print(f"Starting experiment with n={n}, rounds={rounds}, trials={trials}, "
          f"grid={num_params}x{num_params}, using {num_cores} cores.")

    # 1) Prepare (p, q) values: p = n^(-i/num_params), i=1..num_params
    p_values = [n**(-i/num_params) for i in range(1, num_params+1)]
    q_values = [n**(-j/num_params) for j in range(1, num_params+1)]
    
    # 2) Build parameter grid for parallel
    param_grid = []
    for i, p in enumerate(p_values):
        for j, q in enumerate(q_values):
            param_grid.append(((i, j), p, q, n, rounds, trials, property_names))
    
    # 3) Prepare containers to store final results
    results = {}
    for prop in property_names:
        results[prop] = np.zeros((rounds, 2, num_params, num_params))
    results['cross_label_dist'] = np.zeros((rounds, 4, num_params, num_params))

    total_jobs = len(param_grid)  # Should be num_params^2

    # 4) Run parallel using multiprocessing, with a tqdm progress bar
    with multiprocessing.Pool(processes=num_cores) as pool:
        results_list = []
        # Use imap_unordered so results arrive as soon as each job finishes
        for res in tqdm(pool.imap_unordered(run_one_regime, param_grid), 
                        total=total_jobs, 
                        desc="Simulations"):
            results_list.append(res)

    # 5) Assemble final arrays
    for (i, j, accum_props, accum_cross) in results_list:
        for prop in property_names:
            for r in range(rounds):
                for c_idx in range(2):
                    results[prop][r, c_idx, i, j] = accum_props[prop][r, c_idx]
        for r in range(rounds):
            for cross_idx in range(4):
                results['cross_label_dist'][r, cross_idx, i, j] = accum_cross[r, cross_idx]

    print("\nAll parallel jobs completed. Now plotting...")

    # Define how we want to label each property on the plots:
    prop_titles = {
        'conv_smallest_in_comm': "Proportion of nodes w/ locally smallest label",
        'conv_smallest_global':  "Proportion of nodes w/ globally smallest label",
        'fraction_not_changed':  "Proportion of nodes that did not change their label",
    }

    # 6) Plot everything
    exponent_labels = [f"{(k+1)}/{num_params}" for k in range(num_params)]
    
    # Create a helper function that:
    #  1) determines a step size of max(1, num_params // 8),
    #  2) starts from 1 so that, e.g. for num_params=16, we display ticks at 2/16, 4/16, ...
    #  3) places ticks in the middle of each cell by adding 0.5,
    #  4) uses exponent_labels[i] as the tick label.
    step = max(1, num_params // 8)
    def set_custom_ticks(ax):
        # Skip index=0 so the first label is 2/16 for num_params=16, etc.
        indices = list(range(step, num_params, step))
        if indices[-1] != num_params:
            indices.append(num_params)

        # Place ticks at the midpoint of each cell
        xticks = [i - 0.5 for i in indices]
        yticks = [i - 0.5 for i in indices]

        ax.set_xticks(xticks)
        ax.set_xticklabels([exponent_labels[i-1] for i in indices], rotation=45, ha="right")

        ax.set_yticks(yticks)
        ax.set_yticklabels([exponent_labels[i-1] for i in indices], rotation=45, ha="right")
    
    # 6a) Plot the 2-subplot figures for the core properties
    for prop in property_names:
        for r in range(rounds):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for c_idx, ax in enumerate(axes):
                # shape = (num_params, num_params)
                # The slice is results[prop][r, c_idx, :, :]
                # We'll transpose so y->q, x->p
                data_2d = results[prop][r, c_idx, :, :].T  
                sns.heatmap(
                    data_2d, ax=ax, cmap="viridis", vmin=0.0, vmax=1.0,
                    xticklabels=False, 
                    yticklabels=False
                )
                ax.set_title(f"Community {c_idx+1}")
                ax.set_xlabel("a, where p=n^(-a)")
                ax.set_ylabel("b, where q=n^(-b)")

                # Apply our custom ticks
                set_custom_ticks(ax)

            # Use a custom title if it's in prop_titles, else fallback
            title_text = prop_titles.get(prop, prop)
            plt.suptitle(f"{title_text} (Round {r+1})", fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{prop}_round_{r+1}.png")
            plt.close()

    # 6b) Plot cross-label distribution (4 subplots)
    titles = [
        "Community 1, Label=1",
        "Community 1, Label=j1 (smallest in Community 2)",
        "Community 2, Label=1",
        "Community 2, Label=j1 (smallest in Community 2)",
    ]
    for r in range(rounds):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for cross_idx, ax in enumerate(axes.flat):
            data_2d = results['cross_label_dist'][r, cross_idx, :, :].T
            sns.heatmap(
                data_2d, ax=ax, cmap="viridis", vmin=0.0, vmax=1.0,
                xticklabels=False,
                yticklabels=False
            )
            ax.set_title(titles[cross_idx])
            ax.set_xlabel("a, where p=n^(-a)")
            ax.set_ylabel("b, where q=n^(-b)")

            # Apply our custom ticks
            set_custom_ticks(ax)

        # For cross_label_dist, always use the specified plot title:
        plt.suptitle(f"Proportion of nodes w/ given label (Round {r+1})", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"cross_label_dist_round_{r+1}.png")
        plt.close()

    print("Plots saved. Done.")
    return results



###############################################################################
#  7) MAIN / ARGPARSE
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Run parallel label propagation experiments on a 2-community SBM."
    )
    parser.add_argument("--num_params", type=int, default=32,
                        help="Number of p/q exponent steps (grid size).")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of nodes in the graph.")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of label propagation rounds to simulate.")
    parser.add_argument("--trials", type=int, default=4,
                        help="Number of trials per (p,q).")
    parser.add_argument("--num_cores", type=int, default=10,
                        help="Number of CPU cores to use in parallel.")
    parser.add_argument("--prop", dest="properties", action="append",
                        help="Properties to compute. Can repeat --prop for multiple. "
                             "Default: conv_smallest_in_comm, conv_smallest_global, fraction_not_changed")

    args = parser.parse_args()

    if args.properties is None or len(args.properties) == 0:
        property_names = ('conv_smallest_in_comm', 
                          'conv_smallest_global', 
                          'fraction_not_changed')
    else:
        property_names = tuple(args.properties)

    # Call the experiment
    run_experiment_parallel(
        num_params=args.num_params,
        n=args.n,
        rounds=args.rounds,
        trials=args.trials,
        property_names=property_names,
        num_cores=args.num_cores
    )

if __name__ == "__main__":
    main()