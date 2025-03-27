#pragma once

#include "Code.h"
#include <array>
#include <bitset>
#include <vector>
#include <stdexcept>

template <std::size_t n, std::size_t k>
class Decoder {
public:
    explicit Decoder(const Code<n, k>& c) : code(c) {}

    bool peeling_decode(std::bitset<n>& ers) const {
        bool progress = true;
        while (progress) {
            progress = false;
            for (const auto& row : code.getParityCheckMatrix()) {
                std::bitset<n> intersection = row & ers;
                if (intersection.count() == 1) {
                    for (std::size_t i = 0; i < n; ++i) {
                        if (intersection.test(i)) {
                            ers.reset(i);
                            progress = true;
                            break;
                        }
                    }
                }
            }
        }
        return (!ers.any());
    }

    bool peeling_decode_single_bit(std::bitset<n> ers, std::size_t idx) const {
        if (!ers.test(idx)) {
            return true;
        }
        bool progress = true;
        while (progress) {
            progress = false;
            for (const auto& row : code.getParityCheckMatrix()) {
                std::bitset<n> intersection = row & ers;
                if (intersection.count() == 1) {
                    for (std::size_t i = 0; i < n; ++i) {
                        if (intersection.test(i)) {
                            ers.reset(i);
                            progress = true;
                            if (i == idx) {
                                return true;
                            }
                            break;
                        }
                    }
                }
            }
        }
        return false;
    }

    bool gaussian_elimination_decode(std::bitset<n>& ers) const {
        std::vector<std::size_t> erasedIndices;
        for (std::size_t i = 0; i < n; ++i) {
            if (ers.test(i)) {
                erasedIndices.push_back(i);
            }
        }
        const std::size_t numErased = erasedIndices.size();
        if (numErased == 0) {
            return true;
        }
        constexpr std::size_t m = n - k;
        const auto& H = code.getParityCheckMatrix();
        std::vector<std::vector<int>> A(m, std::vector<int>(numErased, 0));
        std::vector<int> b(m, 0);
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < numErased; ++j) {
                A[i][j] = H[i].test(erasedIndices[j]) ? 1 : 0;
            }
        }
        std::size_t rank = 0;
        std::vector<std::size_t> pivotColumns;
        for (std::size_t col = 0; col < numErased && rank < m; ++col) {
            std::size_t pivotRow = rank;
            while (pivotRow < m && A[pivotRow][col] == 0) {
                ++pivotRow;
            }
            if (pivotRow == m) {
                continue;
            }
            if (pivotRow != rank) {
                std::swap(A[rank], A[pivotRow]);
                std::swap(b[rank], b[pivotRow]);
            }
            pivotColumns.push_back(col);
            for (std::size_t r = 0; r < m; ++r) {
                if (r != rank && A[r][col] == 1) {
                    for (std::size_t j = col; j < numErased; ++j) {
                        A[r][j] ^= A[rank][j];
                    }
                    b[r] ^= b[rank];
                }
            }
            ++rank;
        }
        for (std::size_t j : pivotColumns) {
            ers.reset(erasedIndices[j]);
        }
        return !ers.any();
    }

    // New method: batch_gaussian_elimination_decode
    // Given a current erasure pattern (ers), this method computes (in batch)
    // the number of bits that remain unrecoverable after GE decoding.
    // (It does not modify the input ers.)
    int batch_gaussian_elimination_decode(const std::bitset<n>& ers) const {
        // 1. Collect indices for unrecovered bits.
        std::vector<std::size_t> unrecoveredIndices;
        for (std::size_t i = 0; i < n; ++i) {
            if (ers.test(i)) {
                unrecoveredIndices.push_back(i);
            }
        }
        const std::size_t numErased = unrecoveredIndices.size();
        if (numErased == 0) return 0; // All bits recovered.

        // 2. Build the submatrix A of size m x numErased.
        constexpr std::size_t m = n - k;
        const auto& H = code.getParityCheckMatrix();
        std::vector<std::vector<int>> A(m, std::vector<int>(numErased, 0));
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < numErased; ++j) {
                A[i][j] = H[i].test(unrecoveredIndices[j]) ? 1 : 0;
            }
        }

        // 3. Perform Gaussian elimination (mod 2) 
        std::size_t currentRows = m;  // effective number of rows (may decrease)
        std::vector<int> pivotRowForCol(numErased, -1);  // For each column, record pivot row or -1 if free.
        std::size_t i = 0;   // current pivot row index
        std::size_t nc = 0;  // candidate column index
        while (i < currentRows && nc < numErased) {
            // Compress out any all-zero row at position i.
            while (i < currentRows) {
                int rowSum = 0;
                for (std::size_t j = 0; j < numErased; ++j) {
                    rowSum += A[i][j];
                }
                if (rowSum == 0) {
                    A[i] = A[currentRows - 1];
                    currentRows--;
                }
                else {
                    break;
                }
            }
            if (i >= currentRows) break;

            // Find a pivot in column nc starting at row i.
            std::size_t pivot = i;
            while (pivot < currentRows && A[pivot][nc] == 0) {
                ++pivot;
            }
            if (pivot == currentRows) {
                nc++;
                continue;
            }
            if (pivot != i) {
                std::swap(A[i], A[pivot]);
            }
            pivotRowForCol[nc] = static_cast<int>(i);
            for (std::size_t r = 0; r < currentRows; ++r) {
                if (r != i && A[r][nc] == 1) {
                    for (std::size_t j = nc; j < numErased; ++j) {
                        A[r][j] ^= A[i][j];
                    }
                }
            }
            i++;
            nc++;
        }

        // 4. Determine the number of unrecoverable bits.
        std::vector<bool> isPivot(numErased, false);
        for (std::size_t j = 0; j < numErased; ++j) {
            if (pivotRowForCol[j] != -1)
                isPivot[j] = true;
        }
        int unrecovered = 0;
        for (std::size_t j = 0; j < numErased; ++j) {
            if (pivotRowForCol[j] == -1) {
                unrecovered++;
            }
            else {
                int r = pivotRowForCol[j];
                bool clean = true;
                for (std::size_t k = 0; k < numErased; ++k) {
                    if (!isPivot[k] && A[r][k] == 1) {
                        clean = false;
                        break;
                    }
                }
                if (!clean)
                    unrecovered++;
            }
        }
        return unrecovered;
    }

    bool gaussian_elimination_decode_single_bit(std::bitset<n> ers, std::size_t idx) const {
        if (!ers.test(idx)) {
            return true;
        }
        std::vector<std::size_t> erasedIndices;
        for (std::size_t i = 0; i < n; ++i) {
            if (ers.test(i)) {
                erasedIndices.push_back(i);
            }
        }
        const std::size_t numErased = erasedIndices.size();
        std::size_t targetPos = 0;
        bool found = false;
        for (std::size_t i = 0; i < numErased; ++i) {
            if (erasedIndices[i] == idx) {
                targetPos = i;
                found = true;
                break;
            }
        }
        if (!found) {
            return true;
        }
        constexpr std::size_t m = n - k;
        const auto& H = code.getParityCheckMatrix();
        std::vector<std::vector<int>> A(m, std::vector<int>(numErased, 0));
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < numErased; ++j) {
                A[i][j] = H[i].test(erasedIndices[j]) ? 1 : 0;
            }
        }
        std::size_t currentRows = m;
        std::vector<int> pivotRowForCol(numErased, -1);
        std::size_t i = 0;
        std::size_t nc = 0;
        while (i < currentRows && nc < numErased) {
            while (i < currentRows) {
                int rowSum = 0;
                for (std::size_t j = 0; j < numErased; ++j) {
                    rowSum += A[i][j];
                }
                if (rowSum == 0) {
                    A[i] = A[currentRows - 1];
                    currentRows--;
                }
                else {
                    break;
                }
            }
            if (i >= currentRows) break;
            std::size_t pivot = i;
            while (pivot < currentRows && A[pivot][nc] == 0) {
                ++pivot;
            }
            if (pivot == currentRows) {
                nc++;
                continue;
            }
            if (pivot != i) {
                std::swap(A[i], A[pivot]);
            }
            pivotRowForCol[nc] = static_cast<int>(i);
            for (std::size_t r = 0; r < currentRows; ++r) {
                if (r != i && A[r][nc] == 1) {
                    for (std::size_t j = nc; j < numErased; ++j) {
                        A[r][j] ^= A[i][j];
                    }
                }
            }
            i++;
            nc++;
        }
        if (pivotRowForCol[targetPos] == -1)
            return false;
        int pivotRow = pivotRowForCol[targetPos];
        for (std::size_t j = 0; j < numErased; ++j) {
            if (pivotRowForCol[j] == -1 && A[pivotRow][j] == 1) {
                return false;
            }
        }
        return true;
    }

private:
    const Code<n, k>& code;
};
