#pragma once

#include "Code.h"
#include <vector>
#include <stdexcept>
#include <cstddef>
#include "LinAlgUtil.h"

// Templated class for a QC-LDPC code

template <std::size_t n, std::size_t k, std::size_t M>
class QC_LDPC_Code : public Code<n, k> {
public:
    // Default constructor.
    QC_LDPC_Code() = default;

    // Constructor that accepts a base matrix W.
    // W should have dimensions (n - k) x n, where each entry is either a non-negative integer
    // (indicating the circulant shift for the corresponding block) or a negative value indicating a zero block.
    QC_LDPC_Code(const std::vector<std::vector<int>>& W_matrix) {
        setW(W_matrix);
    }

    virtual std::bitset<n> encode(const std::bitset<k>& info) const override {
        std::bitset<n> codeword; // Initially all zeros.
        for (std::size_t i = 0; i < k; ++i) {
            if (info.test(i)) {
                codeword ^= this->getGeneratorMatrix()[i];
            }
        }
        return codeword;
    }

    // Calculates the generator matrix from the parity‑check matrix.
    // (Assumes that the parity‑check matrix is already set.)
    virtual void calculateGeneratorMatrixFromParityCheckMatrix() override {
        // Verify parity‑check matrix exists
        if (this->parityCheckMatrix.empty()) {
            throw std::logic_error("Parity-check matrix has not been initialized.");
        }

        constexpr std::size_t totalRows = n - k;
        constexpr std::size_t totalCols = n;

        if (this->parityCheckMatrix.size() != totalRows) {
            throw std::logic_error("Parity-check matrix dimensions do not match expected size.");
        }

        // Copy the parity-check matrix into a fixed‑size std::array for LinAlgUtil.
        std::array<std::bitset<totalCols>, totalRows> H_arr{};
        for (std::size_t i = 0; i < totalRows; ++i) {
            H_arr[i] = this->parityCheckMatrix[i];
        }

        // Perform Gaussian elimination over GF(2) to obtain the reduced matrix.
        auto result = LinAlgUtil::gaussianElimination<totalRows, totalCols>(H_arr);
        const auto& reduced = result.reducedMatrix;
        const auto& pivots = result.pivotPositions;

        // Identify free (non‑pivot) columns.
        std::vector<std::size_t> freeCols;
        freeCols.reserve(totalCols - pivots.size());
        for (std::size_t c = 0; c < totalCols; ++c) {
            if (std::find(pivots.begin(), pivots.end(), static_cast<int>(c)) == pivots.end()) {
                freeCols.push_back(c);
            }
        }

        // If there are fewer free columns than expected, we cannot form the generator matrix.
        if (freeCols.size() < k) {
            throw std::logic_error("The parity-check matrix does not have enough free columns to form a generator matrix.");
        }

        std::vector<std::size_t> chosenFreeCols(freeCols.begin(), freeCols.begin() + k);

        // Build the generator matrix (k × n). Each generator vector corresponds to one chosen free variable.
        std::array<std::bitset<totalCols>, k> G_arr{};
        for (std::size_t i = 0; i < k; ++i) {
            std::bitset<totalCols> row;
            row.reset();
            // Set the free variable corresponding to this chosen column.
            row.set(chosenFreeCols[i]);

            // For each pivot row in the reduced H, set the corresponding pivot column in the generator vector
            // if that reduced row has a 1 in the chosen free column.
            for (std::size_t r = 0; r < pivots.size(); ++r) {
                if (reduced[r].test(chosenFreeCols[i])) {
                    row.set(pivots[r]);
                }
            }
            G_arr[i] = row;
        }

        this->generatorMatrix = G_arr;
    }


    // Calculates the parity-check matrix from the generator matrix.
    // (Assumes that the generator matrix is already set.)
    virtual void calculateParityCheckMatrixFromGeneratorMatrix() override {
        throw std::logic_error("calculateParityMatrix() not implemented"); // TODO
    }

    // Sets the base matrix W, which defines the QC-LDPC structure.
    // Automatically calculates the overall parity-check matrix H from W.
    void setW(const std::vector<std::vector<int>>& W_matrix) {
        // Validate dimensions: expect W_matrix to have (n - k)/M rows and n columns.
        if (W_matrix.size() != (n - k) / M) {
            throw std::invalid_argument("W_matrix row size must equal n - k");
        }
        for (const auto& row : W_matrix) {
            if (row.size() != n / M) {
                throw std::invalid_argument("Each row of W_matrix must have n elements");
            }
        }
        W = W_matrix;
        calculateParityMatrixFromW();  // Automatically update H based on W.
    }

    // Returns the base matrix W.
    const std::vector<std::vector<int>>& getW() const {
        return W;
    }

private:
    // The base matrix W defining the QC-LDPC code structure.
    std::vector<std::vector<int>> W;

    // Helper function that calculates the overall parity-check matrix H from the base matrix W.
    // The overall H is a block matrix with (n - k) rows and n columns.
    // Each entry W[i][j] (if non-negative) defines an M x M circulant matrix whose first row has a 1 at position 'shift'
    // (with the rest of the block determined by cyclically shifting this row).
    void calculateParityMatrixFromW() {
        // First, clear all bits in the overall parity-check matrix.
        for (std::size_t row = 0; row < (n - k); ++row) {
            this->parityCheckMatrix[row].reset();
        }

        for (std::size_t blockRow = 0; blockRow < (n - k) / M; ++blockRow) {
            for (std::size_t blockCol = 0; blockCol < n / M; ++blockCol) {
                int shift = W[blockRow][blockCol];
                if (shift < 0) continue;

                for (std::size_t innerRow = 0; innerRow < M; ++innerRow) {
                    int totalRow = blockRow * M + innerRow;
                    int shiftedColPos = (innerRow + shift) % M;
                    shiftedColPos += M * blockCol;
                    this->parityCheckMatrix[totalRow].set(shiftedColPos, true);
                }
            }
        }
    }
};
