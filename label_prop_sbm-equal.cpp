#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <numeric>      // for iota
#include <chrono>
#include <ctime>        // for localtime
#include <filesystem>   // C++17 for creating directories
#include <fstream>      // for file I/O
#include <sstream>      // for building filenames
#include <cmath>        // for pow

#ifdef _OPENMP
#include <omp.h>
#endif

//------------------------------------------------------------------------------------
// We'll handle up to 65 points (0..64) for alpha & beta, i.e. GRID_SIZE=64 => 65 steps.
static const int GRID_SIZE = 64;  

// We will parse N, but let's provide a fallback default:
static const int DEFAULT_N = 10000;

//------------------------------------------------------------------------------------
// Global arrays to hold the *averaged* results across trials (for each (p,q) pair).
// We'll set the dimension for i,j to GRID_SIZE+1 (i.e., 65).
//
// For each of the 4 properties, we keep sub-arrays for each round and community/etc.
// We'll dynamically parse numiterations from the command line, but to store data we
// can do so either dynamically or with a max dimension. Here, for clarity, let's
// just store them in vectors allocated after we know numiterations.
//
// (Alternatively, we could store them in a static 3D array if we fix a maximum possible
//  iteration count. But let's do dynamic vectors for generality.)
//
// We'll define these *later* after we parse numiterations, so we can do
// e.g. std::vector<std::vector<std::vector<double>>> conv_smallest_in_comm_c0;
// at global scope we need pointers or static arrays, but let's do "static" pointers
// that we allocate once we know numiterations.
//
// For clarity, we'll store them in static variables, but allocate in main().

static double ***conv_smallest_in_comm_c0  = nullptr;
static double ***conv_smallest_in_comm_c1  = nullptr;
static double ***conv_smallest_global_c0   = nullptr;
static double ***conv_smallest_global_c1   = nullptr;
static double ***fraction_not_changed_c0    = nullptr;
static double ***fraction_not_changed_c1    = nullptr;
static double ***cross_label_dist_00       = nullptr;
static double ***cross_label_dist_01       = nullptr;
static double ***cross_label_dist_10       = nullptr;
static double ***cross_label_dist_11       = nullptr;

// We'll store for each node: which community it belongs to (0 or 1).
static std::vector<int> community;

//------------------------------------------------------------------------------------
// A helper function to allocate a 3D array [numiterations][GRID_SIZE+1][GRID_SIZE+1].
static double*** allocate3D(int numiterations) {
    double ***arr = new double**[numiterations];
    for (int r = 0; r < numiterations; r++) {
        arr[r] = new double*[GRID_SIZE + 1];
        for (int i = 0; i <= GRID_SIZE; i++) {
            arr[r][i] = new double[GRID_SIZE + 1];
            // Initialize to 0.0
            std::fill_n(arr[r][i], GRID_SIZE + 1, 0.0);
        }
    }
    return arr;
}

// A helper to free a 3D array allocated above.
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

//------------------------------------------------------------------------------------
// Generate a 2-community SBM with random 0/1 membership.
// Probability p for edges within a community, q for edges across communities.
static std::vector<std::vector<int>> generateSBMRandomSplit(
    int n, double p, double q, std::mt19937 &gen)
{

    // Resize community vector to n nodes.
    community.resize(n);

    // Create a vector of node indices [0, 1, ..., n-1]
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle the indices randomly
    std::shuffle(indices.begin(), indices.end(), gen);

    // If n is even, assign exactly n/2 nodes per community.
    // If n is odd, one community will have one extra node.
    int half = n / 2;
    for (int k = 0; k < n; k++) {
        if (k < half)
            community[indices[k]] = 0;
        else
            community[indices[k]] = 1;
    }


    // Bernoulli for edges
    std::bernoulli_distribution distIntra(p);
    std::bernoulli_distribution distInter(q);

    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            bool same = (community[i] == community[j]);
            bool edge = (same ? distIntra(gen) : distInter(gen));
            if (edge) {
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }
    return adj;
}

//------------------------------------------------------------------------------------
// Initialize each node's label to a unique random label in [1..n].
static std::vector<int> initializeUniqueRandomLabels(int n, std::mt19937 &gen) {
    std::vector<int> labels(n);
    std::iota(labels.begin(), labels.end(), 1);
    std::shuffle(labels.begin(), labels.end(), gen);
    return labels;
}

//------------------------------------------------------------------------------------
// Synchronous label propagation iteration.
static void labelPropagationIteration(
    const std::vector<std::vector<int>> &adj,
    const std::vector<int> &oldLabels,
    std::vector<int> &newLabels)
{
    int n = (int)adj.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int u = 0; u < n; u++) {
        std::unordered_map<int,int> freq;
        freq[oldLabels[u]]++;

        for (int v : adj[u]) {
            freq[oldLabels[v]]++;
        }

        int bestLabel = -1;
        int bestCount = -1;
        for (auto &kv : freq) {
            int lbl = kv.first;
            int cnt = kv.second;
            if (cnt > bestCount) {
                bestCount = cnt;
                bestLabel = lbl;
            } else if (cnt == bestCount && lbl < bestLabel) {
                bestLabel = lbl;
            }
        }
        newLabels[u] = bestLabel;
    }
}

//------------------------------------------------------------------------------------
// Run one simulation for a given (p, q) and fill local arrays for each round.
// We fix the smallest initial label in each community (initMinLabelC0, initMinLabelC1)
// right after random initialization, and do not update them in subsequent rounds.
static void runOneSimulation(
    int n,
    double p, double q,
    int numiterations,
    std::mt19937 &gen,
    // local arrays to fill for each round r in [0..numiterations-1]
    double local_csc0[], double local_csc1[],
    double local_csg0[], double local_csg1[],
    double local_fnc0[], double local_fnc1[],
    double local_cld00[], double local_cld01[],
    double local_cld10[], double local_cld11[]
) {
    // 1) Generate adjacency
    auto adj = generateSBMRandomSplit(n, p, q, gen);

    // 2) Random unique labels in [1..n]
    auto labels = initializeUniqueRandomLabels(n, gen);

    // -- NEW PART: Ensure that at least one node in community 0 has label=1 --
    {
        // Find the node that has label=1
        int idxOfLabel1 = -1;
        for (int u = 0; u < n; u++) {
            if (labels[u] == 1) {
                idxOfLabel1 = u;
                break;
            }
        }
        // If that node is in community 1, swap labels with some node in community 0
        if (idxOfLabel1 >= 0 && community[idxOfLabel1] == 1) {
            int idxInC0 = -1;
            for (int v = 0; v < n; v++) {
                if (community[v] == 0) {
                    idxInC0 = v;
                    break;
                }
            }
            if (idxInC0 >= 0) {
                int temp = labels[idxInC0];
                labels[idxInC0] = 1;
                labels[idxOfLabel1] = temp;
            }
        }
    }
    // --------------------------------------------------------------------------

    // 3) Determine smallest initial label in c0, c1
    int initMinLabelC0 = INT32_MAX;
    int initMinLabelC1 = INT32_MAX;

    // Count nodes in each community
    int c0count = 0, c1count = 0;
    for (int u = 0; u < n; u++) {
        if (community[u] == 0) c0count++;
        else                   c1count++;
    }

    for (int u = 0; u < n; u++) {
        if (community[u] == 0 && labels[u] < initMinLabelC0) {
            initMinLabelC0 = labels[u];
        }
        if (community[u] == 1 && labels[u] < initMinLabelC1) {
            initMinLabelC1 = labels[u];
        }
    }
    if (c0count == 0) initMinLabelC0 = -1;
    if (c1count == 0) initMinLabelC1 = -1;

    // We'll track old labels to detect changes
    std::vector<int> prevLabels(n, -1);
    std::vector<int> newLabels(n);

    // A helper lambda to compute stats after each round
    auto computeRoundStats = [&](int roundIdx) {
        int count_sm_in_comm_c0 = 0, count_sm_in_comm_c1 = 0; 
        int count_sm_global_c0  = 0, count_sm_global_c1  = 0;
        int count_not_changed_c0 = 0, count_not_changed_c1 = 0;

        int c0_minC0 = 0, c0_minC1 = 0, c1_minC0 = 0, c1_minC1 = 0;

        for (int u = 0; u < n; u++) {
            int lbl = labels[u];
            int comm = community[u];

            // conv_smallest_in_comm
            if (lbl == initMinLabelC0) count_sm_in_comm_c0++;
            if (lbl == initMinLabelC1) count_sm_in_comm_c1++;

            // conv_smallest_global
            if (lbl == 1 && comm == 0) count_sm_global_c0++;
            if (lbl == 1 && comm == 1) count_sm_global_c1++;

            // fraction_not_changed
            if (roundIdx > 0 && lbl == prevLabels[u]) {
                if (comm == 0) count_not_changed_c0++;
                else           count_not_changed_c1++;
            }

            // cross_label_distribution
            if (comm == 0) {
                if (lbl == initMinLabelC0) c0_minC0++;
                if (lbl == initMinLabelC1) c0_minC1++;
            } else {
                if (lbl == initMinLabelC0) c1_minC0++;
                if (lbl == initMinLabelC1) c1_minC1++;
            }
        }

        double denomAll = double(n);
        local_csc0[roundIdx] =
            (denomAll > 0 && initMinLabelC0 != -1)
            ? double(count_sm_in_comm_c0) / denomAll : 0.0;
        local_csc1[roundIdx] =
            (denomAll > 0 && initMinLabelC1 != -1)
            ? double(count_sm_in_comm_c1) / denomAll : 0.0;

        double dc0 = double(c0count);
        double dc1 = double(c1count);

        local_csg0[roundIdx] =
            (dc0 > 0) ? double(count_sm_global_c0) / dc0 : 0.0;
        local_csg1[roundIdx] =
            (dc1 > 0) ? double(count_sm_global_c1) / dc1 : 0.0;

        local_fnc0[roundIdx] =
            (roundIdx > 0 && dc0 > 0)
            ? double(count_not_changed_c0)/dc0 : 0.0;
        local_fnc1[roundIdx] =
            (roundIdx > 0 && dc1 > 0)
            ? double(count_not_changed_c1)/dc1 : 0.0;

        local_cld00[roundIdx] =
            (dc0 > 0 && initMinLabelC0 != -1) ? double(c0_minC0)/dc0 : 0.0;
        local_cld01[roundIdx] =
            (dc0 > 0 && initMinLabelC1 != -1) ? double(c0_minC1)/dc0 : 0.0;
        local_cld10[roundIdx] =
            (dc1 > 0 && initMinLabelC0 != -1) ? double(c1_minC0)/dc1 : 0.0;
        local_cld11[roundIdx] =
            (dc1 > 0 && initMinLabelC1 != -1) ? double(c1_minC1)/dc1 : 0.0;
    };

    // Round 0 => stats after initial labeling
    computeRoundStats(0);

    for (int r = 1; r < numiterations; r++) {
        prevLabels = labels;

        // One synchronous iteration
        labelPropagationIteration(adj, prevLabels, newLabels);
        labels.swap(newLabels);

        computeRoundStats(r);
    }
}

//------------------------------------------------------------------------------------
// Helper to write a single [GRID_SIZE+1 x GRID_SIZE+1] matrix for a specific round
// to a CSV file.
static void writeMatrixToCSV(
    const std::string &filename,
    double ***data,    // data[roundIdx][i][j]
    int roundIdx
) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: could not open " << filename << "\n";
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

//------------------------------------------------------------------------------------
// main()
//
// Usage (all arguments optional):
//   ./label_prop_sbm [N=10000] [minalpha=-1] [maxalpha=0]
//                    [minbeta=-1] [maxbeta=0]
//                    [numiterations=10] [numtrials=4]
//
// Example:
//   ./label_prop_sbm 50000 -0.5 0 -0.5 0 8 2
//
// This will create a log in the results folder specifying all parameters.
int main(int argc, char** argv) {
    // 1) Parse arguments with defaults:
    int argIndex = 1;

    // N
    int N = (argc > argIndex) ? std::stoi(argv[argIndex]) : DEFAULT_N;
    argIndex++;

    // minalpha, maxalpha
    double minalpha = (argc > argIndex) ? std::stod(argv[argIndex]) : -1.0;
    argIndex++;
    double maxalpha = (argc > argIndex) ? std::stod(argv[argIndex]) : 0.0;
    argIndex++;

    // minbeta, maxbeta
    double minbeta = (argc > argIndex) ? std::stod(argv[argIndex]) : -1.0;
    argIndex++;
    double maxbeta = (argc > argIndex) ? std::stod(argv[argIndex]) : 0.0;
    argIndex++;

    // numiterations
    int numiterations = (argc > argIndex) ? std::stoi(argv[argIndex]) : 10;
    argIndex++;

    // numtrials
    int numtrials = (argc > argIndex) ? std::stoi(argv[argIndex]) : 4;
    argIndex++;

#ifdef _OPENMP
    std::cout << "Using OpenMP with " << omp_get_max_threads() << " threads.\n";
#endif
    std::cout << "Parameters:\n"
              << "  N = " << N << "\n"
              << "  minalpha = " << minalpha << ", maxalpha = " << maxalpha << "\n"
              << "  minbeta = " << minbeta << ", maxbeta = " << maxbeta << "\n"
              << "  numiterations = " << numiterations << "\n"
              << "  numtrials = " << numtrials << "\n\n";

    // 2) Allocate the 3D arrays [numiterations][GRID_SIZE+1][GRID_SIZE+1]
    conv_smallest_in_comm_c0  = allocate3D(numiterations);
    conv_smallest_in_comm_c1  = allocate3D(numiterations);
    conv_smallest_global_c0   = allocate3D(numiterations);
    conv_smallest_global_c1   = allocate3D(numiterations);
    fraction_not_changed_c0    = allocate3D(numiterations);
    fraction_not_changed_c1    = allocate3D(numiterations);
    cross_label_dist_00       = allocate3D(numiterations);
    cross_label_dist_01       = allocate3D(numiterations);
    cross_label_dist_10       = allocate3D(numiterations);
    cross_label_dist_11       = allocate3D(numiterations);

    // 3) Create a master seed
    unsigned masterSeed = (unsigned)std::chrono::high_resolution_clock::now()
                                        .time_since_epoch().count();

    // 4) Outer loop: we define alpha, beta in [minalpha..maxalpha] & [minbeta..maxbeta]
    //    in equally spaced increments. Then p = N^alpha, q = N^beta.
    // 
    // We have i in [0..GRID_SIZE], alpha = minalpha + i*(maxalpha - minalpha)/GRID_SIZE
    // Similarly for j in [0..GRID_SIZE].
    // That yields 65 x 65 = 4225 possible pairs.

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (int i = 0; i <= GRID_SIZE; i++) {
        for (int j = 0; j <= GRID_SIZE; j++) {
            double alpha = minalpha + (maxalpha - minalpha)*double(i)/double(GRID_SIZE);
            double beta  = minbeta  + (maxbeta  - minbeta )*double(j)/double(GRID_SIZE);

            // p = N^alpha, q = N^beta
            double p = std::pow((double)N, alpha);
            double q = std::pow((double)N, beta);

            // Accumulation arrays for each round
            std::vector<double> sum_csc0(numiterations, 0.0),
                                sum_csc1(numiterations, 0.0),
                                sum_csg0(numiterations, 0.0),
                                sum_csg1(numiterations, 0.0),
                                sum_fnc0(numiterations, 0.0),
                                sum_fnc1(numiterations, 0.0),
                                sum_cld00(numiterations, 0.0),
                                sum_cld01(numiterations, 0.0),
                                sum_cld10(numiterations, 0.0),
                                sum_cld11(numiterations, 0.0);

            // numtrials simulations
            for (int t = 0; t < numtrials; t++) {
                unsigned localSeed = masterSeed + (i*10000) + (j*100) + t;
                std::mt19937 localGen(localSeed);

                // local arrays for each round
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

                // Run one simulation
                runOneSimulation(N, p, q, numiterations, localGen,
                                 local_csc0.data(), local_csc1.data(),
                                 local_csg0.data(), local_csg1.data(),
                                 local_fnc0.data(), local_fnc1.data(),
                                 local_cld00.data(), local_cld01.data(),
                                 local_cld10.data(), local_cld11.data());

                // Accumulate
                for (int r = 0; r < numiterations; r++) {
                    sum_csc0[r] += local_csc0[r];
                    sum_csc1[r] += local_csc1[r];
                    sum_csg0[r] += local_csg0[r];
                    sum_csg1[r] += local_csg1[r];
                    sum_fnc0[r] += local_fnc0[r];
                    sum_fnc1[r] += local_fnc1[r];
                    sum_cld00[r] += local_cld00[r];
                    sum_cld01[r] += local_cld01[r];
                    sum_cld10[r] += local_cld10[r];
                    sum_cld11[r] += local_cld11[r];
                }
            }

            // Average
            for (int r = 0; r < numiterations; r++) {
                conv_smallest_in_comm_c0[r][i][j] = sum_csc0[r] / numtrials;
                conv_smallest_in_comm_c1[r][i][j] = sum_csc1[r] / numtrials;

                conv_smallest_global_c0[r][i][j]  = sum_csg0[r] / numtrials;
                conv_smallest_global_c1[r][i][j]  = sum_csg1[r] / numtrials;

                fraction_not_changed_c0[r][i][j]  = sum_fnc0[r] / numtrials;
                fraction_not_changed_c1[r][i][j]  = sum_fnc1[r] / numtrials;

                cross_label_dist_00[r][i][j] = sum_cld00[r] / numtrials;
                cross_label_dist_01[r][i][j] = sum_cld01[r] / numtrials;
                cross_label_dist_10[r][i][j] = sum_cld10[r] / numtrials;
                cross_label_dist_11[r][i][j] = sum_cld11[r] / numtrials;
            }
        }
    }

    //--------------------------------------------------------------------------------
    // Create a timestamped results folder, plus subfolders, and a parameter log file.
    //--------------------------------------------------------------------------------
    std::time_t now = std::time(nullptr);
    std::tm localTime;
#ifdef _WIN32
    localtime_s(&localTime, &now);
#else
    localtime_r(&now, &localTime);
#endif

    std::ostringstream dirname;
    dirname << "results/run_"
            << (1900 + localTime.tm_year)
            << (localTime.tm_mon + 1)
            << localTime.tm_mday << "_"
            << localTime.tm_hour
            << localTime.tm_min
            << localTime.tm_sec;

    std::string runFolder = dirname.str();
    std::filesystem::create_directories(runFolder);

    // Write parameters to a log file in that folder
    {
        std::ofstream paramLog(runFolder + "/parameters.txt");
        if (paramLog.is_open()) {
            paramLog << "N = " << N << "\n"
                     << "minalpha = " << minalpha << "\n"
                     << "maxalpha = " << maxalpha << "\n"
                     << "minbeta = " << minbeta << "\n"
                     << "maxbeta = " << maxbeta << "\n"
                     << "numiterations = " << numiterations << "\n"
                     << "numtrials = " << numtrials << "\n";
            paramLog.close();
        }
    }

    // Subfolders
    std::string cscPath = runFolder + "/conv_smallest_in_comm";
    std::string csgPath = runFolder + "/conv_smallest_global";
    std::string fncPath = runFolder + "/fraction_not_changed";
    std::string cldPath = runFolder + "/cross_label_distribution";

    std::filesystem::create_directories(cscPath);
    std::filesystem::create_directories(csgPath);
    std::filesystem::create_directories(fncPath);
    std::filesystem::create_directories(cldPath);

    // Write out a CSV for each round in [0..numiterations-1]
    for (int r = 0; r < numiterations; r++) {
        {
            std::ostringstream f0, f1;
            f0 << cscPath << "/conv_smallest_in_comm_c0_round_" << r << ".csv";
            f1 << cscPath << "/conv_smallest_in_comm_c1_round_" << r << ".csv";
            writeMatrixToCSV(f0.str(), conv_smallest_in_comm_c0, r);
            writeMatrixToCSV(f1.str(), conv_smallest_in_comm_c1, r);
        }
        {
            std::ostringstream f0, f1;
            f0 << csgPath << "/conv_smallest_global_c0_round_" << r << ".csv";
            f1 << csgPath << "/conv_smallest_global_c1_round_" << r << ".csv";
            writeMatrixToCSV(f0.str(), conv_smallest_global_c0, r);
            writeMatrixToCSV(f1.str(), conv_smallest_global_c1, r);
        }
        {
            std::ostringstream f0, f1;
            f0 << fncPath << "/fraction_not_changed_c0_round_" << r << ".csv";
            f1 << fncPath << "/fraction_not_changed_c1_round_" << r << ".csv";
            writeMatrixToCSV(f0.str(), fraction_not_changed_c0, r);
            writeMatrixToCSV(f1.str(), fraction_not_changed_c1, r);
        }
        {
            std::ostringstream f00, f01, f10, f11;
            f00 << cldPath << "/cross_label_dist_00_round_" << r << ".csv";
            f01 << cldPath << "/cross_label_dist_01_round_" << r << ".csv";
            f10 << cldPath << "/cross_label_dist_10_round_" << r << ".csv";
            f11 << cldPath << "/cross_label_dist_11_round_" << r << ".csv";
            writeMatrixToCSV(f00.str(), cross_label_dist_00, r);
            writeMatrixToCSV(f01.str(), cross_label_dist_01, r);
            writeMatrixToCSV(f10.str(), cross_label_dist_10, r);
            writeMatrixToCSV(f11.str(), cross_label_dist_11, r);
        }
    }

    std::cout << "Done. Results in folder: " << runFolder << "\n";

    //--------------------------------------------------------------------------------
    // Clean up allocated 3D arrays
    //--------------------------------------------------------------------------------
    free3D(conv_smallest_in_comm_c0,  numiterations);
    free3D(conv_smallest_in_comm_c1,  numiterations);
    free3D(conv_smallest_global_c0,   numiterations);
    free3D(conv_smallest_global_c1,   numiterations);
    free3D(fraction_not_changed_c0,   numiterations);
    free3D(fraction_not_changed_c1,   numiterations);
    free3D(cross_label_dist_00,       numiterations);
    free3D(cross_label_dist_01,       numiterations);
    free3D(cross_label_dist_10,       numiterations);
    free3D(cross_label_dist_11,       numiterations);

    return 0;
}
