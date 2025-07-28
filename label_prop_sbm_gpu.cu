#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Constants
static const int GRID_SIZE = 64;
static const int DEFAULT_N = 10000;
static const int BLOCK_SIZE = 256;
static const int MAX_DEGREE = 1000;

// Global arrays for results
static double ***conv_smallest_in_comm_c0 = nullptr;
static double ***conv_smallest_in_comm_c1 = nullptr;
static double ***conv_smallest_global_c0 = nullptr;
static double ***conv_smallest_global_c1 = nullptr;
static double ***fraction_not_changed_c0 = nullptr;
static double ***fraction_not_changed_c1 = nullptr;
static double ***cross_label_dist_00 = nullptr;
static double ***cross_label_dist_01 = nullptr;
static double ***cross_label_dist_10 = nullptr;
static double ***cross_label_dist_11 = nullptr;

// CUDA kernel for generating SBM edges
__global__ void generateSBMEdges(
    int n, double p, double q,
    int* community, 
    int* edge_count,
    int* edge_dest,
    curandState* states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    
    int u = idx / n;
    int v = idx % n;
    
    if (u >= v) return; // Only generate upper triangle
    
    curandState localState = states[idx];
    float rand_val = curand_uniform(&localState);
    states[idx] = localState;
    
    bool has_edge = false;
    if (community[u] == community[v]) {
        has_edge = (rand_val < p);
    } else {
        has_edge = (rand_val < q);
    }
    
    if (has_edge) {
        // Add edge u->v
        int edge_idx = atomicAdd(&edge_count[u], 1);
        if (edge_idx < MAX_DEGREE) {
            edge_dest[u * MAX_DEGREE + edge_idx] = v;
        }
        
        // Add edge v->u (undirected)
        edge_idx = atomicAdd(&edge_count[v], 1);
        if (edge_idx < MAX_DEGREE) {
            edge_dest[v * MAX_DEGREE + edge_idx] = u;
        }
    }
}

// CUDA kernel for label propagation
__global__ void labelPropagationKernel(
    int n, int round_idx,
    int* edge_count,
    int* edge_dest,
    int* old_labels,
    int* new_labels
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    
    // Count label frequencies
    int label_freq[10]; // Assume max 10 different labels per node
    int label_vals[10];
    int num_labels = 0;
    
    // Count own label
    int own_label = old_labels[u];
    label_vals[0] = own_label;
    label_freq[0] = 1;
    num_labels = 1;
    
    // Count neighbor labels
    int degree = edge_count[u];
    for (int i = 0; i < degree && i < MAX_DEGREE; i++) {
        int v = edge_dest[u * MAX_DEGREE + i];
        int neighbor_label = old_labels[v];
        
        // Find if label already exists
        bool found = false;
        for (int j = 0; j < num_labels; j++) {
            if (label_vals[j] == neighbor_label) {
                label_freq[j]++;
                found = true;
                break;
            }
        }
        
        if (!found && num_labels < 10) {
            label_vals[num_labels] = neighbor_label;
            label_freq[num_labels] = 1;
            num_labels++;
        }
    }
    
    // Find most frequent label (tie-break on smaller label)
    int best_label = label_vals[0];
    int best_count = label_freq[0];
    
    for (int i = 1; i < num_labels; i++) {
        if (label_freq[i] > best_count || 
            (label_freq[i] == best_count && label_vals[i] < best_label)) {
            best_count = label_freq[i];
            best_label = label_vals[i];
        }
    }
    
    new_labels[u] = best_label;
}

// CUDA kernel for computing statistics
__global__ void computeStatsKernel(
    int n,
    int* labels,
    int* community,
    int* smallest_label_comm_0,
    int* smallest_label_comm_1,
    double* stats
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    
    int label = labels[u];
    int comm = community[u];
    
    // Atomic counters for statistics
    if (comm == 0) {
        if (label == *smallest_label_comm_0) {
            atomicAdd(&stats[0], 1.0); // conv_smallest_in_comm_c0
        }
        if (label == 1) {
            atomicAdd(&stats[2], 1.0); // conv_smallest_global_c0
        }
    } else {
        if (label == *smallest_label_comm_1) {
            atomicAdd(&stats[1], 1.0); // conv_smallest_in_comm_c1
        }
        if (label == 1) {
            atomicAdd(&stats[3], 1.0); // conv_smallest_global_c1
        }
    }
    
    // Cross-label distribution
    if (comm == 0 && label == *smallest_label_comm_0) {
        atomicAdd(&stats[4], 1.0); // cross_label_dist_00
    } else if (comm == 0 && label == *smallest_label_comm_1) {
        atomicAdd(&stats[5], 1.0); // cross_label_dist_01
    } else if (comm == 1 && label == *smallest_label_comm_0) {
        atomicAdd(&stats[6], 1.0); // cross_label_dist_10
    } else if (comm == 1 && label == *smallest_label_comm_1) {
        atomicAdd(&stats[7], 1.0); // cross_label_dist_11
    }
}

// CUDA kernel for counting unchanged labels
__global__ void countUnchangedKernel(
    int n,
    int* old_labels,
    int* new_labels,
    int* community,
    double* unchanged_stats
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    
    if (old_labels[u] == new_labels[u]) {
        int comm = community[u];
        if (comm == 0) {
            atomicAdd(&unchanged_stats[0], 1.0);
        } else {
            atomicAdd(&unchanged_stats[1], 1.0);
        }
    }
}

// Initialize random states for CUDA
__global__ void initRandomStates(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// Allocate 3D array
static double*** allocate3D(int numiterations) {
    double ***arr = new double**[numiterations];
    for (int r = 0; r < numiterations; r++) {
        arr[r] = new double*[GRID_SIZE + 1];
        for (int i = 0; i <= GRID_SIZE; i++) {
            arr[r][i] = new double[GRID_SIZE + 1];
            std::fill_n(arr[r][i], GRID_SIZE + 1, 0.0);
        }
    }
    return arr;
}

// Free 3D array
static void free3D(double ***arr, int numiterations) {
    if (!arr) return;
    for (int r = 0; r < numiterations; r++) {
        for (int i = 0; i <= GRID_SIZE; i++) {
            delete[] arr[r][i];
        }
        delete[] arr[r];
    }
    delete[] arr;
}

// Run one GPU simulation
static void runOneGPUSimulation(
    int n,
    double p, double q,
    int numiterations,
    unsigned long seed,
    double local_csc0[], double local_csc1[],
    double local_csg0[], double local_csg1[],
    double local_fnc0[], double local_fnc1[],
    double local_cld00[], double local_cld01[],
    double local_cld10[], double local_cld11[]
) {
    // Allocate device memory
    thrust::device_vector<int> d_community(n);
    thrust::device_vector<int> d_edge_count(n, 0);
    thrust::device_vector<int> d_edge_dest(n * MAX_DEGREE, -1);
    thrust::device_vector<int> d_labels(n);
    thrust::device_vector<int> d_new_labels(n);
    thrust::device_vector<curandState> d_states(n * n);
    
    // Initialize random states
    int blocks = (n * n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initRandomStates<<<blocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_states.data()), seed);
    
    // Generate random community assignments
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(n),
        d_community.begin(),
        [seed] __device__ (int idx) {
            thrust::default_random_engine rng(seed + idx);
            thrust::uniform_int_distribution<int> dist(0, 1);
            return dist(rng);
        }
    );
    
    // Generate SBM edges
    generateSBMEdges<<<blocks, BLOCK_SIZE>>>(
        n, p, q,
        thrust::raw_pointer_cast(d_community.data()),
        thrust::raw_pointer_cast(d_edge_count.data()),
        thrust::raw_pointer_cast(d_edge_dest.data()),
        thrust::raw_pointer_cast(d_states.data())
    );
    
    // Initialize random unique labels
    thrust::sequence(d_labels.begin(), d_labels.end(), 1);
    thrust::shuffle(d_labels.begin(), d_labels.end(), thrust::default_random_engine(seed));
    
    // Force label=1 to be in community 0
    thrust::host_vector<int> h_labels = d_labels;
    thrust::host_vector<int> h_community = d_community;
    
    int label1_idx = -1, comm0_idx = -1;
    for (int i = 0; i < n; i++) {
        if (h_labels[i] == 1) label1_idx = i;
        if (h_community[i] == 0) comm0_idx = i;
    }
    
    if (label1_idx >= 0 && comm0_idx >= 0 && h_community[label1_idx] != 0) {
        std::swap(h_labels[label1_idx], h_labels[comm0_idx]);
    }
    
    d_labels = h_labels;
    
    // Find smallest labels in each community
    int smallest_comm0 = n + 1, smallest_comm1 = n + 1;
    for (int i = 0; i < n; i++) {
        if (h_community[i] == 0 && h_labels[i] < smallest_comm0) {
            smallest_comm0 = h_labels[i];
        }
        if (h_community[i] == 1 && h_labels[i] < smallest_comm1) {
            smallest_comm1 = h_labels[i];
        }
    }
    
    // Copy to device
    thrust::device_vector<int> d_smallest_comm0(1, smallest_comm0);
    thrust::device_vector<int> d_smallest_comm1(1, smallest_comm1);
    
    // Run label propagation iterations
    for (int round = 0; round < numiterations; round++) {
        // Run label propagation kernel
        int node_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        labelPropagationKernel<<<node_blocks, BLOCK_SIZE>>>(
            n, round,
            thrust::raw_pointer_cast(d_edge_count.data()),
            thrust::raw_pointer_cast(d_edge_dest.data()),
            thrust::raw_pointer_cast(d_labels.data()),
            thrust::raw_pointer_cast(d_new_labels.data())
        );
        
        // Compute statistics
        thrust::device_vector<double> d_stats(8, 0.0);
        computeStatsKernel<<<node_blocks, BLOCK_SIZE>>>(
            n,
            thrust::raw_pointer_cast(d_new_labels.data()),
            thrust::raw_pointer_cast(d_community.data()),
            thrust::raw_pointer_cast(d_smallest_comm0.data()),
            thrust::raw_pointer_cast(d_smallest_comm1.data()),
            thrust::raw_pointer_cast(d_stats.data())
        );
        
        // Count unchanged labels
        thrust::device_vector<double> d_unchanged(2, 0.0);
        countUnchangedKernel<<<node_blocks, BLOCK_SIZE>>>(
            n,
            thrust::raw_pointer_cast(d_labels.data()),
            thrust::raw_pointer_cast(d_new_labels.data()),
            thrust::raw_pointer_cast(d_community.data()),
            thrust::raw_pointer_cast(d_unchanged.data())
        );
        
        // Copy results to host
        thrust::host_vector<double> h_stats = d_stats;
        thrust::host_vector<double> h_unchanged = d_unchanged;
        
        // Count nodes in each community
        int comm0_count = 0, comm1_count = 0;
        for (int i = 0; i < n; i++) {
            if (h_community[i] == 0) comm0_count++;
            else comm1_count++;
        }
        
        // Store results
        local_csc0[round] = h_stats[0] / comm0_count;
        local_csc1[round] = h_stats[1] / comm1_count;
        local_csg0[round] = h_stats[2] / comm0_count;
        local_csg1[round] = h_stats[3] / comm1_count;
        local_fnc0[round] = h_unchanged[0] / comm0_count;
        local_fnc1[round] = h_unchanged[1] / comm1_count;
        local_cld00[round] = h_stats[4] / comm0_count;
        local_cld01[round] = h_stats[5] / comm0_count;
        local_cld10[round] = h_stats[6] / comm1_count;
        local_cld11[round] = h_stats[7] / comm1_count;
        
        // Swap labels for next iteration
        d_labels = d_new_labels;
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Write matrix to CSV (same as CPU version)
static void writeMatrixToCSV(
    const std::string &filename,
    double ***data,
    int roundIdx
) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    
    for (int i = 0; i <= GRID_SIZE; i++) {
        for (int j = 0; j <= GRID_SIZE; j++) {
            ofs << data[roundIdx][i][j];
            if (j < GRID_SIZE) ofs << ",";
        }
        ofs << "\n";
    }
    ofs.close();
}

int main(int argc, char** argv) {
    // Parse command line arguments
    int N = (argc > 1) ? std::atoi(argv[1]) : DEFAULT_N;
    double minalpha = (argc > 2) ? std::atof(argv[2]) : -1.0;
    double maxalpha = (argc > 3) ? std::atof(argv[3]) : 0.0;
    double minbeta = (argc > 4) ? std::atof(argv[4]) : -1.0;
    double maxbeta = (argc > 5) ? std::atof(argv[5]) : 0.0;
    int numiterations = (argc > 6) ? std::atoi(argv[6]) : 10;
    int numtrials = (argc > 7) ? std::atoi(argv[7]) : 4;
    
    std::cout << "GPU Label Propagation SBM Simulation" << std::endl;
    std::cout << "N = " << N << ", iterations = " << numiterations 
              << ", trials = " << numtrials << std::endl;
    
    // Allocate result arrays
    conv_smallest_in_comm_c0 = allocate3D(numiterations);
    conv_smallest_in_comm_c1 = allocate3D(numiterations);
    conv_smallest_global_c0 = allocate3D(numiterations);
    conv_smallest_global_c1 = allocate3D(numiterations);
    fraction_not_changed_c0 = allocate3D(numiterations);
    fraction_not_changed_c1 = allocate3D(numiterations);
    cross_label_dist_00 = allocate3D(numiterations);
    cross_label_dist_01 = allocate3D(numiterations);
    cross_label_dist_10 = allocate3D(numiterations);
    cross_label_dist_11 = allocate3D(numiterations);
    
    // Create results directory
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm* tm = std::localtime(&time_t);
    
    std::ostringstream dirname;
    dirname << "results/run_gpu_" << std::put_time(tm, "%Y%m%d_%H%M%S");
    std::filesystem::create_directories(dirname.str());
    
    // Save parameters
    std::ofstream paramFile(dirname.str() + "/parameters.txt");
    paramFile << "N = " << N << "\n";
    paramFile << "minalpha = " << minalpha << "\n";
    paramFile << "maxalpha = " << maxalpha << "\n";
    paramFile << "minbeta = " << minbeta << "\n";
    paramFile << "maxbeta = " << maxbeta << "\n";
    paramFile << "numiterations = " << numiterations << "\n";
    paramFile << "numtrials = " << numtrials << "\n";
    paramFile << "GPU_ACCELERATED = true\n";
    paramFile.close();
    
    // Main simulation loop
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int i = 0; i <= GRID_SIZE; i++) {
        double alpha = minalpha + (maxalpha - minalpha) * i / GRID_SIZE;
        double p = std::pow(N, alpha);
        
        for (int j = 0; j <= GRID_SIZE; j++) {
            double beta = minbeta + (maxbeta - minbeta) * j / GRID_SIZE;
            double q = std::pow(N, beta);
            
            std::cout << "Processing (i,j) = (" << i << "," << j 
                      << ") -> (p,q) = (" << p << "," << q << ")" << std::endl;
            
            // Run multiple trials
            for (int trial = 0; trial < numtrials; trial++) {
                unsigned long seed = gen();
                
                std::vector<double> local_csc0(numiterations),
                                   local_csc1(numiterations),
                                   local_csg0(numiterations),
                                   local_csg1(numiterations),
                                   local_fnc0(numiterations),
                                   local_fnc1(numiterations),
                                   local_cld00(numiterations),
                                   local_cld01(numiterations),
                                   local_cld10(numiterations),
                                   local_cld11(numiterations);
                
                runOneGPUSimulation(N, p, q, numiterations, seed,
                                   local_csc0.data(), local_csc1.data(),
                                   local_csg0.data(), local_csg1.data(),
                                   local_fnc0.data(), local_fnc1.data(),
                                   local_cld00.data(), local_cld01.data(),
                                   local_cld10.data(), local_cld11.data());
                
                // Accumulate results
                for (int round = 0; round < numiterations; round++) {
                    conv_smallest_in_comm_c0[round][i][j] += local_csc0[round];
                    conv_smallest_in_comm_c1[round][i][j] += local_csc1[round];
                    conv_smallest_global_c0[round][i][j] += local_csg0[round];
                    conv_smallest_global_c1[round][i][j] += local_csg1[round];
                    fraction_not_changed_c0[round][i][j] += local_fnc0[round];
                    fraction_not_changed_c1[round][i][j] += local_fnc1[round];
                    cross_label_dist_00[round][i][j] += local_cld00[round];
                    cross_label_dist_01[round][i][j] += local_cld01[round];
                    cross_label_dist_10[round][i][j] += local_cld10[round];
                    cross_label_dist_11[round][i][j] += local_cld11[round];
                }
            }
            
            // Average over trials
            for (int round = 0; round < numiterations; round++) {
                conv_smallest_in_comm_c0[round][i][j] /= numtrials;
                conv_smallest_in_comm_c1[round][i][j] /= numtrials;
                conv_smallest_global_c0[round][i][j] /= numtrials;
                conv_smallest_global_c1[round][i][j] /= numtrials;
                fraction_not_changed_c0[round][i][j] /= numtrials;
                fraction_not_changed_c1[round][i][j] /= numtrials;
                cross_label_dist_00[round][i][j] /= numtrials;
                cross_label_dist_01[round][i][j] /= numtrials;
                cross_label_dist_10[round][i][j] /= numtrials;
                cross_label_dist_11[round][i][j] /= numtrials;
            }
        }
    }
    
    // Write results to CSV files
    std::filesystem::create_directories(dirname.str() + "/conv_smallest_in_comm");
    std::filesystem::create_directories(dirname.str() + "/conv_smallest_global");
    std::filesystem::create_directories(dirname.str() + "/fraction_not_changed");
    std::filesystem::create_directories(dirname.str() + "/cross_label_distribution");
    
    for (int round = 0; round < numiterations; round++) {
        writeMatrixToCSV(dirname.str() + "/conv_smallest_in_comm/conv_smallest_in_comm_c0_round_" + std::to_string(round) + ".csv",
                        conv_smallest_in_comm_c0, round);
        writeMatrixToCSV(dirname.str() + "/conv_smallest_in_comm/conv_smallest_in_comm_c1_round_" + std::to_string(round) + ".csv",
                        conv_smallest_in_comm_c1, round);
        writeMatrixToCSV(dirname.str() + "/conv_smallest_global/conv_smallest_global_c0_round_" + std::to_string(round) + ".csv",
                        conv_smallest_global_c0, round);
        writeMatrixToCSV(dirname.str() + "/conv_smallest_global/conv_smallest_global_c1_round_" + std::to_string(round) + ".csv",
                        conv_smallest_global_c1, round);
        writeMatrixToCSV(dirname.str() + "/fraction_not_changed/fraction_not_changed_c0_round_" + std::to_string(round) + ".csv",
                        fraction_not_changed_c0, round);
        writeMatrixToCSV(dirname.str() + "/fraction_not_changed/fraction_not_changed_c1_round_" + std::to_string(round) + ".csv",
                        fraction_not_changed_c1, round);
        writeMatrixToCSV(dirname.str() + "/cross_label_distribution/cross_label_dist_00_round_" + std::to_string(round) + ".csv",
                        cross_label_dist_00, round);
        writeMatrixToCSV(dirname.str() + "/cross_label_distribution/cross_label_dist_01_round_" + std::to_string(round) + ".csv",
                        cross_label_dist_01, round);
        writeMatrixToCSV(dirname.str() + "/cross_label_distribution/cross_label_dist_10_round_" + std::to_string(round) + ".csv",
                        cross_label_dist_10, round);
        writeMatrixToCSV(dirname.str() + "/cross_label_distribution/cross_label_dist_11_round_" + std::to_string(round) + ".csv",
                        cross_label_dist_11, round);
    }
    
    // Cleanup
    free3D(conv_smallest_in_comm_c0, numiterations);
    free3D(conv_smallest_in_comm_c1, numiterations);
    free3D(conv_smallest_global_c0, numiterations);
    free3D(conv_smallest_global_c1, numiterations);
    free3D(fraction_not_changed_c0, numiterations);
    free3D(fraction_not_changed_c1, numiterations);
    free3D(cross_label_dist_00, numiterations);
    free3D(cross_label_dist_01, numiterations);
    free3D(cross_label_dist_10, numiterations);
    free3D(cross_label_dist_11, numiterations);
    
    std::cout << "GPU simulation completed. Results saved to: " << dirname.str() << std::endl;
    return 0;
} 