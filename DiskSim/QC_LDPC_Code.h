#pragma once

#include "Code.h"

// Templated class for a QCLDPC code inheriting from Code<n, k>.
// n: Block length (codeword length)
// k: Dimension (number of information bits)
template <std::size_t n, std::size_t k>
class QC_LDPC_Code : public Code<n, k> {
public:
    // Constructor that takes precomputed generator and parity-check matrices.
    QCLDPCCode(const std::array<std::bitset<n>, k>& genMatrix,
        const std::array<std::bitset<n>, n - k>& parMatrix)
    {
        this->setGeneratorMatrix(genMatrix);
        this->setParityCheckMatrix(parMatrix);
    }

    // If you wish to construct the code from a W matrix that defines the QC structure,
    // you can add another constructor that takes W as input and then computes the matrices.
    // For example:
    // QCLDPCCode(const std::array<std::array<int, wCols>, wRows>& W) { ... }

    // Implementation of the encode function.
    // For linear codes, encoding is typically done via the generator matrix.
    // Here, we compute the codeword as the modulo-2 sum of those rows of G
    // where the corresponding information bit is 1.
    virtual std::bitset<n> encode(const std::bitset<k>& info) const override {
        std::bitset<n> codeword; // Initialized to all 0's.
        // Loop through each bit of the information word.
        for (std::size_t i = 0; i < k; ++i) {
            if (info.test(i)) {
                // XOR the i-th row of the generator matrix if the info bit is 1.
                codeword ^= this->getGeneratorMatrix()[i];
            }
        }
        return codeword;
    }
};
