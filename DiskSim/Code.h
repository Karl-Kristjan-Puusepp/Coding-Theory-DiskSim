#pragma once

#include <bitset>
#include <array>
#include <cstddef>

// Templated abstract base class representing a generic binary linear code.
// Template parameters:
//   n: Block length (codeword length)
//   k: Dimension (number of information bits)
template <std::size_t n, std::size_t k>
class Code {
public:
    virtual ~Code() = default;

    // Pure virtual function to encode a std::bitset of size k into a std::bitset of size n.
    virtual std::bitset<n> encode(const std::bitset<k>& info) const = 0;

    // Pure virtual functions to calculate one matrix from the other.
    // These functions assume that the complementary matrix is already set.
    virtual void calculateGeneratorMatrixFromParityCheckMatrix() = 0;
    virtual void calculateParityCheckMatrixFromGeneratorMatrix() = 0;

    // Returns the block length (n).
    static constexpr std::size_t getBlockLength() { return n; }

    // Returns the code dimension (k).
    static constexpr std::size_t getDimension() { return k; }

    // Returns the code rate (k/n).
    static constexpr double getRate() { return static_cast<double>(k) / n; }

    // Accessors for the internal matrices.
    // Generator matrix (G) is stored as an array of k rows, each a bitset of size n.
    const std::array<std::bitset<n>, k>& getGeneratorMatrix() const { return generatorMatrix; }

    // Parity-check matrix (H) is stored as an array of (n-k) rows, each a bitset of size n.
    const std::array<std::bitset<n>, n - k>& getParityCheckMatrix() const { return parityCheckMatrix; }

    // Setters for the matrices.
    void setGeneratorMatrix(const std::array<std::bitset<n>, k>& genMatrix) {
        generatorMatrix = genMatrix;
    }
    void setParityCheckMatrix(const std::array<std::bitset<n>, n - k>& parMatrix) {
        parityCheckMatrix = parMatrix;
    }

protected:
    // Internal storage for matrices.
    std::array<std::bitset<n>, k> generatorMatrix;
    std::array<std::bitset<n>, n - k> parityCheckMatrix;
};
