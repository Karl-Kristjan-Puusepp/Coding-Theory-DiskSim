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

    // Returns the block length (n).
    static constexpr std::size_t getBlockLength() { return n; }

    // Returns the code dimension (k).
    static constexpr std::size_t getDimension() { return k; }

    // Returns the code rate (k/n).
    static constexpr double getRate() { return static_cast<double>(k) / n; }

    // Accessors for the internal matrices.
    // Returns the generator matrix (G) as an array of std::bitset of size n (k rows).
    const std::array<std::bitset<n>, k>& getGeneratorMatrix() const { return generatorMatrix; }

    // Returns the parity-check matrix (H) as an array of std::bitset of size n ((n-k) rows).
    const std::array<std::bitset<n>, n - k>& getParityCheckMatrix() const { return parityCheckMatrix; }

    // Setter for the generator matrix.
    void setGeneratorMatrix(const std::array<std::bitset<n>, k>& genMatrix) {
        generatorMatrix = genMatrix;
    }

    // Setter for the parity-check matrix.
    void setParityCheckMatrix(const std::array<std::bitset<n>, n - k>& parMatrix) {
        parityCheckMatrix = parMatrix;
    }

protected:
    // Generator matrix G with dimensions k x n.
    std::array<std::bitset<n>, k> generatorMatrix;

    // Parity-check matrix H with dimensions (n-k) x n.
    std::array<std::bitset<n>, n - k> parityCheckMatrix;
};

