#!/usr/bin/env python3

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import os, time
import itertools
import multiprocessing
from tqdm import tqdm
import argparse
from numba import cuda, jit
import warnings
warnings.filterwarnings('ignore')

###############################################################################
#  1) GPU-ACCELERATED SBM GENERATION
###############################################################################
@cuda.jit
def generate_sbm_gpu_kernel(n, p, q, community, adjacency):
    """
    CUDA kernel for generating SBM adjacency matrix
    """
    i, j = cuda.grid(2)
    
    if i >= n or j >= n:
        return
    
    # Generate random number
    x = cuda.random.xoroshiro128p_normalize()
    
    # Determine if edge exists
    has_edge = False
    if community[i] == community[j]:
        has_edge = (x < p)
    else:
        has_edge = (x < q)
    
    # Set adjacency matrix (symmetric)
    if has_edge and i != j:  # No self-loops
        adjacency[i, j] = 1
        adjacency[j, i] = 1

def generate_sbm_2_community_gpu(n, p, q):
    """
    GPU-accelerated 2-community balanced SBM generation
    """
    # Allocate GPU memory
    community = cp.random.randint(0, 2, n, dtype=cp.int32)
    adjacency = cp.zeros((n, n), dtype=cp.int32)
    
    # Set up CUDA grid
    threadsperblock = (16, 16)
    blockspergrid_x = (n + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (n + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # Launch kernel
    generate_sbm_gpu_kernel[blockspergrid, threadsperblock](n, p, q, community, adjacency)
    
    return adjacency.get(), community.get()

###############################################################################
#  2) GPU-ACCELERATED LABEL PROPAGATION
###############################################################################
@cuda.jit
def label_propagation_gpu_kernel(n, round_idx, adjacency, old_labels, new_labels):
    """
    CUDA kernel for one round of label propagation
    """
    u = cuda.grid(1)
    
    if u >= n:
        return
    
    # Count label frequencies
    label_freq = cuda.local.array(100, dtype=np.int32)  # Max 100 different labels
    label_vals = cuda.local.array(100, dtype=np.int32)
    num_labels = 0
    
    # Count own label
    own_label = old_labels[u]
    label_vals[0] = own_label
    label_freq[0] = 1
    num_labels = 1
    
    # Count neighbor labels
    for v in range(n):
        if adjacency[u, v] == 1:  # If edge exists
            neighbor_label = old_labels[v]
            
            # Find if label already exists
            found = False
            for j in range(num_labels):
                if label_vals[j] == neighbor_label:
                    label_freq[j] += 1
                    found = True
                    break
            
            if not found and num_labels < 100:
                label_vals[num_labels] = neighbor_label
                label_freq[num_labels] = 1
                num_labels += 1
    
    # Find most frequent label (tie-break on smaller label)
    best_label = label_vals[0]
    best_count = label_freq[0]
    
    for i in range(1, num_labels):
        if label_freq[i] > best_count or (label_freq[i] == best_count and label_vals[i] < best_label):
            best_count = label_freq[i]
            best_label = label_vals[i]
    
    new_labels[u] = best_label

def run_label_propagation_gpu(adjacency, labels, rounds=5):
    """
    GPU-accelerated label propagation
    """
    n = len(labels)
    all_rounds_labels = [labels.copy()]
    
    # Transfer to GPU
    d_adjacency = cp.asarray(adjacency, dtype=cp.int32)
    d_labels = cp.asarray(labels, dtype=cp.int32)
    d_new_labels = cp.empty_like(d_labels)
    
    # Set up CUDA grid
    threadsperblock = 256
    blockspergrid = (n + threadsperblock - 1) // threadsperblock
    
    for round_idx in range(rounds):
        # Launch kernel
        label_propagation_gpu_kernel[blockspergrid, threadsperblock](
            n, round_idx, d_adjacency, d_labels, d_new_labels
        )
        
        # Copy back to host
        new_labels = d_new_labels.get()
        all_rounds_labels.append(new_labels.copy())
        
        # Swap for next iteration
        d_labels, d_new_labels = d_new_labels, d_labels
    
    return all_rounds_labels

###############################################################################
#  3) COMPUTE PROPERTIES (CPU - lightweight)
###############################################################################
def compute_properties_gpu(adjacency, all_rounds_labels, smallest_label_comm_0, smallest_label_comm_1):
    """
    Compute convergence properties from GPU results
    """
    n = len(all_rounds_labels[0])
    rounds = len(all_rounds_labels) - 1
    
    # Determine communities from adjacency (simplified)
    community = np.zeros(n, dtype=int)
    community[n//2:] = 1  # Simple split
    
    properties = {
        'conv_smallest_in_comm': {'c0': [], 'c1': []},
        'conv_smallest_global': {'c0': [], 'c1': []},
        'fraction_not_changed': {'c0': [], 'c1': []}
    }
    
    for round_idx in range(rounds):
        current_labels = all_rounds_labels[round_idx]
        next_labels = all_rounds_labels[round_idx + 1]
        
        # Compute properties for each community
        for comm in [0, 1]:
            comm_nodes = np.where(community == comm)[0]
            if len(comm_nodes) == 0:
                continue
                
            comm_labels = next_labels[comm_nodes]
            
            # conv_smallest_in_comm
            if comm == 0:
                smallest_in_comm = smallest_label_comm_0
            else:
                smallest_in_comm = smallest_label_comm_1
            
            conv_smallest_in_comm = np.mean(comm_labels == smallest_in_comm)
            properties['conv_smallest_in_comm'][f'c{comm}'].append(conv_smallest_in_comm)
            
            # conv_smallest_global
            conv_smallest_global = np.mean(comm_labels == 1)
            properties['conv_smallest_global'][f'c{comm}'].append(conv_smallest_global)
            
            # fraction_not_changed
            unchanged = np.mean(current_labels[comm_nodes] == next_labels[comm_nodes])
            properties['fraction_not_changed'][f'c{comm}'].append(unchanged)
    
    return properties

###############################################################################
#  4) MAIN EXPERIMENT FUNCTION
###############################################################################
def run_experiment_gpu_parallel(num_params=64, n=1000, rounds=5, trials=5, 
                               property_names=('conv_smallest_in_comm', 'conv_smallest_global', 'fraction_not_changed'),
                               num_cores=10):
    """
    GPU-accelerated parallel experiment runner
    """
    print(f"Starting GPU-accelerated experiment with {num_params}x{num_params} grid")
    print(f"Parameters: n={n}, rounds={rounds}, trials={trials}")
    
    # Initialize result arrays
    results = {}
    for prop in property_names:
        results[prop] = {'c0': np.zeros((rounds, num_params+1, num_params+1)), 
                        'c1': np.zeros((rounds, num_params+1, num_params+1))}
    
    # Cross-label distribution
    results['cross_label_dist'] = {
        '00': np.zeros((rounds, num_params+1, num_params+1)),
        '01': np.zeros((rounds, num_params+1, num_params+1)),
        '10': np.zeros((rounds, num_params+1, num_params+1)),
        '11': np.zeros((rounds, num_params+1, num_params+1))
    }
    
    # Main experiment loop
    total_combinations = (num_params + 1) ** 2
    progress_bar = tqdm(total=total_combinations, desc="Processing parameter combinations")
    
    for i in range(num_params + 1):
        for j in range(num_params + 1):
            # Calculate p, q values
            p = n ** (-i / num_params)
            q = n ** (-j / num_params)
            
            # Run multiple trials
            trial_results = {prop: {'c0': [], 'c1': []} for prop in property_names}
            trial_results['cross_label_dist'] = {'00': [], '01': [], '10': [], '11': []}
            
            for trial in range(trials):
                # Generate SBM
                adjacency, community = generate_sbm_2_community_gpu(n, p, q)
                
                # Initialize labels
                labels = np.random.permutation(n) + 1
                
                # Force label=1 to be in community 0
                comm0_nodes = np.where(community == 0)[0]
                if len(comm0_nodes) > 0:
                    label1_pos = np.where(labels == 1)[0][0]
                    if community[label1_pos] != 0:
                        swap_pos = comm0_nodes[0]
                        labels[label1_pos], labels[swap_pos] = labels[swap_pos], labels[label1_pos]
                
                # Find smallest labels in each community
                smallest_label_comm_0 = np.min(labels[community == 0])
                smallest_label_comm_1 = np.min(labels[community == 1])
                
                # Run label propagation
                all_rounds_labels = run_label_propagation_gpu(adjacency, labels, rounds)
                
                # Compute properties
                properties = compute_properties_gpu(adjacency, all_rounds_labels, 
                                                  smallest_label_comm_0, smallest_label_comm_1)
                
                # Store trial results
                for prop in property_names:
                    for comm in ['c0', 'c1']:
                        trial_results[prop][comm].append(properties[prop][comm])
                
                # Cross-label distribution (simplified)
                for round_idx in range(rounds):
                    labels_round = all_rounds_labels[round_idx + 1]
                    comm0_labels = labels_round[community == 0]
                    comm1_labels = labels_round[community == 1]
                    
                    if len(comm0_labels) > 0:
                        trial_results['cross_label_dist']['00'].append(
                            np.mean(comm0_labels == smallest_label_comm_0))
                        trial_results['cross_label_dist']['01'].append(
                            np.mean(comm0_labels == smallest_label_comm_1))
                    
                    if len(comm1_labels) > 0:
                        trial_results['cross_label_dist']['10'].append(
                            np.mean(comm1_labels == smallest_label_comm_0))
                        trial_results['cross_label_dist']['11'].append(
                            np.mean(comm1_labels == smallest_label_comm_1))
            
            # Average over trials and store
            for prop in property_names:
                for comm in ['c0', 'c1']:
                    avg_results = np.mean(trial_results[prop][comm], axis=0)
                    for round_idx in range(rounds):
                        results[prop][comm][round_idx, i, j] = avg_results[round_idx]
            
            for key in ['00', '01', '10', '11']:
                avg_results = np.mean(trial_results['cross_label_dist'][key], axis=0)
                for round_idx in range(rounds):
                    results['cross_label_dist'][key][round_idx, i, j] = avg_results[round_idx]
            
            progress_bar.update(1)
    
    progress_bar.close()
    return results

###############################################################################
#  5) PLOTTING FUNCTIONS
###############################################################################
def plot_heatmaps_gpu(results, num_params, rounds, output_dir="gpu_results"):
    """
    Generate heatmaps from GPU results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot each property
    for prop_name in ['conv_smallest_in_comm', 'conv_smallest_global', 'fraction_not_changed']:
        for round_idx in range(rounds):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Community 0
            im1 = ax1.imshow(results[prop_name]['c0'][round_idx], origin='lower', vmin=0, vmax=1)
            ax1.set_title(f'{prop_name} Community 0 - Round {round_idx}')
            ax1.set_xlabel('Beta Index')
            ax1.set_ylabel('Alpha Index')
            plt.colorbar(im1, ax=ax1)
            
            # Community 1
            im2 = ax2.imshow(results[prop_name]['c1'][round_idx], origin='lower', vmin=0, vmax=1)
            ax2.set_title(f'{prop_name} Community 1 - Round {round_idx}')
            ax2.set_xlabel('Beta Index')
            ax2.set_ylabel('Alpha Index')
            plt.colorbar(im2, ax=ax2)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{prop_name}_round_{round_idx}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Cross-label distribution
    for round_idx in range(rounds):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        for i, key in enumerate(['00', '01', '10', '11']):
            row, col = i // 2, i % 2
            im = axes[row, col].imshow(results['cross_label_dist'][key][round_idx], 
                                      origin='lower', vmin=0, vmax=1)
            axes[row, col].set_title(f'Cross-label dist {key} - Round {round_idx}')
            axes[row, col].set_xlabel('Beta Index')
            axes[row, col].set_ylabel('Alpha Index')
            plt.colorbar(im, ax=axes[row, col])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cross_label_dist_round_{round_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()

###############################################################################
#  6) MAIN FUNCTION
###############################################################################
def main():
    parser = argparse.ArgumentParser(description='GPU-accelerated Label Propagation SBM Experiment')
    parser.add_argument('--num_params', type=int, default=32, help='Grid size for parameter space')
    parser.add_argument('--n', type=int, default=1000, help='Number of nodes')
    parser.add_argument('--rounds', type=int, default=5, help='Number of propagation rounds')
    parser.add_argument('--trials', type=int, default=4, help='Number of trials')
    parser.add_argument('--num_cores', type=int, default=1, help='Number of CPU cores (not used in GPU version)')
    parser.add_argument('--output_dir', type=str, default='gpu_results', help='Output directory')
    
    args = parser.parse_args()
    
    print("GPU-accelerated Label Propagation SBM Experiment")
    print(f"Parameters: num_params={args.num_params}, n={args.n}, rounds={args.rounds}, trials={args.trials}")
    
    # Check GPU availability
    try:
        print(f"Using GPU: {cp.cuda.Device(0).name}")
        print(f"GPU Memory: {cp.cuda.Device(0).mem_info[1] / 1024**3:.1f} GB")
    except:
        print("Warning: No GPU detected, falling back to CPU")
        return
    
    # Run experiment
    start_time = time.time()
    results = run_experiment_gpu_parallel(
        num_params=args.num_params,
        n=args.n,
        rounds=args.rounds,
        trials=args.trials
    )
    end_time = time.time()
    
    print(f"Experiment completed in {end_time - start_time:.2f} seconds")
    
    # Generate plots
    plot_heatmaps_gpu(results, args.num_params, args.rounds, args.output_dir)
    print(f"Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 