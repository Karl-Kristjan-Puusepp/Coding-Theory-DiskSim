#include "QC_LDPC_Code.h"
#include "Decoder.h"
#include <iostream>
#include <fstream>
#include <bitset>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

// Helper: Generate a random erasure pattern for a codeword of length N,
// where each bit is erased with probability p.
template <std::size_t N>
std::bitset<N> generateErasurePattern(double p, std::mt19937& rng) {
    std::bitset<N> ers;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (std::size_t i = 0; i < N; ++i)
        ers.set(i, (dist(rng) < p));
    return ers;
}

int main() {
    // QC-LDPC parameters:
    // Overall codeword length n = 156, information length k = 117, circulant block size M = 13.
    // Base matrix dimensions: (n-k)/M x (n/M) = 39/13 x 156/13 = 3 x 12.
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

    // Set up a random number generator for overall seeding.
    std::random_device rd;

    // Simulation parameters.
    const int numSimulations = 100000;  // e.g., 100,000 trials per erasure probability.
    const int codeLength = 156;         // Codeword length.
    const double p_min = 0.01;
    const double p_max = 0.50;
    const double p_step = 0.01;
    const int numPoints = static_cast<int>((p_max - p_min) / p_step) + 1;

    // Open CSV output file.
    std::ofstream outFile("gen_results.csv");
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open output file.\n";
        return 1;
    }
    outFile << "Probability,Peeling_BER,GE_BER\n";

    // Loop over erasure probabilities.
    for (int point = 0; point < numPoints; ++point) {
        double p = p_min + point * p_step;
        std::cout << "Starting simulation for p = " << p << "\n";
        uint64_t totalUnrecoveredPeeling = 0;
        uint64_t totalUnrecoveredGE = 0;
        uint64_t totalBits = static_cast<uint64_t>(codeLength) * numSimulations;

        // Parallelized Monte Carlo simulation for the given p.
#pragma omp parallel for reduction(+: totalUnrecoveredPeeling, totalUnrecoveredGE) schedule(static)
        for (int sim = 0; sim < numSimulations; ++sim) {
            // Create a thread-local RNG using a seed based on a global random device and the iteration index.
            std::mt19937 localRng(rd() + sim);
            // Generate a random erasure pattern.
            std::bitset<codeLength> ers = generateErasurePattern<codeLength>(p, localRng);
            // Bits not erased are assumed recovered.

            // Run full peeling decoding.
            std::bitset<codeLength> ersPeeling = ers; // Copy because peeling_decode modifies its argument.
            decoder.peeling_decode(ersPeeling);
            int unrecoveredPeeling = static_cast<int>(ersPeeling.count());

            // GE decoding: if peeling recovered all bits, assume GE does too.
            int unrecoveredGE = 0;
            if (unrecoveredPeeling > 0) {
                // Use the new batch GE decoder to process all unrecovered bits at once.
                unrecoveredGE = decoder.batch_gaussian_elimination_decode(ersPeeling);
            }

            totalUnrecoveredPeeling += unrecoveredPeeling;
            totalUnrecoveredGE += unrecoveredGE;
        }

        // Calculate BER as the ratio of unrecoverable bits to total transmitted bits.
        double peelingBER = static_cast<double>(totalUnrecoveredPeeling) / totalBits;
        double geBER = static_cast<double>(totalUnrecoveredGE) / totalBits;

        // Write the results to the CSV file.
        outFile << p << "," << peelingBER << "," << geBER << "\n";
        std::cout << "Completed simulation for p = " << p << "\n";
    }

    outFile.close();
    std::cout << "Simulation complete. Results saved to gen_results.csv\n";
    return 0;
}
