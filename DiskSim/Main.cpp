#include "QC_LDPC_Code.h"
#include "Decoder.h"
#include <iostream>
#include <fstream>
#include <bitset>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

// Helper: Generate a random erasure pattern for a codeword of length N with exactly e erasures.
template <std::size_t N>
std::bitset<N> generateErasurePatternExact(int e, std::mt19937& rng) {
    std::bitset<N> ers;
    std::vector<int> indices(N);
    for (int i = 0; i < N; i++) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), rng);
    for (int i = 0; i < e; i++) {
        ers.set(indices[i], true);
    }
    return ers;
}

// New Helper: Generate a probability distribution for k elements using a softmax transformation on normal samples.
// The parameter alpha controls the "sharpness" of the distribution.
std::vector<double> generateSoftmaxDistribution(int k, std::mt19937& rng, double alpha = 1.0) {
    std::normal_distribution<double> norm(0.0, 1.0);
    std::vector<double> samples(k);
    for (int i = 0; i < k; i++) {
        samples[i] = norm(rng);
    }
    std::vector<double> expSamples(k);
    double sumExp = 0.0;
    for (int i = 0; i < k; i++) {
        expSamples[i] = std::exp(alpha * samples[i]);
        sumExp += expSamples[i];
    }
    for (int i = 0; i < k; i++) {
        expSamples[i] /= sumExp;
    }
    return expSamples;
}

// Optimized Helper: Sample an index from a discrete distribution given by probs.
// This function uses a manual cumulative sum rather than constructing a std::discrete_distribution.
int sampleFromDistribution(const std::vector<double>& probs, std::mt19937& rng) {
    double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double cumulative = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i];
        if (r < cumulative)
            return static_cast<int>(i);
    }
    return static_cast<int>(probs.size() - 1);
}

int main() {
    // QC-LDPC parameters:
    // Codeword length n = 156, information length k = 117, circulant block size M = 13.
    // Base matrix W dimensions: (n - k)/M x (n/M) = 3 x 12.
    std::vector<std::vector<int>> W = {
        { 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12},
        { 0,  3,  1,  8,  2,  9, 12,  4, 11,  5,  7,  6},
        { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}
    };

    // Construct the QC-LDPC code.
    QC_LDPC_Code<156, 117, 13> qc_ldpc(W);
    qc_ldpc.calculateGeneratorMatrixFromParityCheckMatrix();

    // Create a decoder instance.
    Decoder<156, 117> decoder(qc_ldpc);

    // Set up a random number generator (for seeding thread-local RNGs).
    std::random_device rd;

    // Simulation parameters.
    const int numSimulations = 10000;  // 10,000 simulations per number of erased disks.
    const int n_val = 156;             // Total disks (codeword length).
    const int k_val = 117;             // Information disks (first k bits).
    const int e_min = 1;
    const int e_max = n_val / 2;       // Up to 0.5 * n.

    // Output CSV with columns:
    // "# disks switched off,Peeling_random,GE_random,Peeling_normal_distribution,GE_normal_distribution"
    std::ofstream outFile("gen_results.csv");
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open output file.\n";
        return 1;
    }
    outFile << "# disks switched off,Peeling_random,GE_random,Peeling_normal_distribution,GE_normal_distribution\n";

    // Loop over e = number of erased disks.
    for (int e = e_min; e <= e_max; e++) {
        // Counters for Scenario 1 (Random erasures).
        uint64_t totalRequests_s1 = 0;
        uint64_t peelingFailures_s1 = 0;
        uint64_t geFailures_s1 = 0;
        // Counters for Scenario 2 (Softmax-based distribution).
        uint64_t totalRequests_s2 = 0;
        uint64_t peelingFailures_s2 = 0;
        uint64_t geFailures_s2 = 0;

        // Parallelize the simulation loop.
        #pragma omp parallel for reduction(+: totalRequests_s1, peelingFailures_s1, geFailures_s1, totalRequests_s2, peelingFailures_s2, geFailures_s2) schedule(static)
        for (int sim = 0; sim < numSimulations; sim++) {
            // Create a thread-local RNG.
            std::mt19937 localRng(rd() + sim);

            // -------------- Scenario 1: Random Erasure Pattern --------------
            // Generate a full-length erasure pattern with exactly e erasures.
            std::bitset<n_val> ers_s1 = generateErasurePatternExact<n_val>(e, localRng);
            // Process only the first k_val disks.
            for (int i = 0; i < k_val; i++) {
                totalRequests_s1++;
                if (!ers_s1.test(i))
                    continue;
                bool peelingSuccess = decoder.peeling_decode_single_bit(ers_s1, i);
                if (!peelingSuccess) {
                    peelingFailures_s1++;
                    bool geSuccess = decoder.gaussian_elimination_decode_single_bit(ers_s1, i);
                    if (!geSuccess)
                        geFailures_s1++;
                }
            }

            // -------------- Scenario 2: Softmax Distribution-Based Erasure --------------
            // Generate a probability distribution for the first k_val disks using softmax.
            std::vector<double> probs = generateSoftmaxDistribution(k_val, localRng, 1.0);
            // Create a vector of (index, probability) pairs for the first k_val disks.
            std::vector<std::pair<int, double>> infoDisks;
            for (int i = 0; i < k_val; i++) {
                infoDisks.push_back({ i, probs[i] });
            }
            // Sort by increasing probability.
            std::sort(infoDisks.begin(), infoDisks.end(), [](auto& a, auto& b) {
                return a.second < b.second;
                });
            // Create a full-length erasure pattern (of size n_val).
            std::bitset<n_val> ers_s2;
            ers_s2.reset();
            // Mark the e lowest-probability disks among the first k_val as erased.
            for (int i = 0; i < e && i < k_val; i++) {
                ers_s2.set(infoDisks[i].first, true);
            }
            // Simulate k_val requests according to the softmax distribution.
            for (int req = 0; req < k_val; req++) {
                totalRequests_s2++;
                int requested = sampleFromDistribution(probs, localRng);
                if (!ers_s2.test(requested))
                    continue;
                bool peelingSuccess = decoder.peeling_decode_single_bit(ers_s2, requested);
                if (!peelingSuccess) {
                    peelingFailures_s2++;
                    bool geSuccess = decoder.gaussian_elimination_decode_single_bit(ers_s2, requested);
                    if (!geSuccess)
                        geFailures_s2++;
                }
            }
        } // end parallel simulation loop

        double peeling_random_ratio = static_cast<double>(peelingFailures_s1) / totalRequests_s1;
        double ge_random_ratio = static_cast<double>(geFailures_s1) / totalRequests_s1;
        double peeling_normal_ratio = static_cast<double>(peelingFailures_s2) / totalRequests_s2;
        double ge_normal_ratio = static_cast<double>(geFailures_s2) / totalRequests_s2;

        outFile << e << "," << peeling_random_ratio << "," << ge_random_ratio << ","
            << peeling_normal_ratio << "," << ge_normal_ratio << "\n";
        std::cout << "Completed simulation for " << e << " disks switched off.\n";
    }
    outFile.close();
    std::cout << "Simulation complete. Results saved to gen_results.csv\n";
    return 0;
}
