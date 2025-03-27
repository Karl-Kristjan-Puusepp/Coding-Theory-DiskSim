#pragma once

#include <array>
#include <bitset>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace LinAlgUtil {

    template <std::size_t ROWS, std::size_t COLS>
    struct GaussResult {
        std::array<std::bitset<COLS>, ROWS> reducedMatrix; 
        std::vector<int> pivotPositions;
    };

    // Perform Gaussian elimination over GF(2) on a ROWS×COLS binary matrix.
    template <std::size_t ROWS, std::size_t COLS>
    GaussResult<ROWS, COLS> gaussianElimination(
        const std::array<std::bitset<COLS>, ROWS>& inputMatrix
    ) {
        auto mat = inputMatrix; 
        std::vector<int> pivots;
        std::size_t pivot_row = 0;

        for (std::size_t col = 0; col < COLS && pivot_row < ROWS; ++col) {
            std::size_t pivot = pivot_row;
            while (pivot < ROWS && !mat[pivot].test(col)) {
                ++pivot;
            }
            if (pivot == ROWS) continue;

            if (pivot != pivot_row) {
                std::swap(mat[pivot], mat[pivot_row]);
            }

            pivots.push_back(static_cast<int>(col));
            for (std::size_t r = 0; r < ROWS; ++r) {
                if (r != pivot_row && mat[r].test(col)) {
                    mat[r] ^= mat[pivot_row];
                }
            }
            ++pivot_row;
        }

        return { mat, pivots };
    }
}
