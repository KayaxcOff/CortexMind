//
// Created by muham on 9.11.2025.
//

#ifndef CORTEXMIND_KERNEL_HPP
#define CORTEXMIND_KERNEL_HPP

#include <CortexMind/Utils/params.hpp>
#include <vector>

namespace cortex::tools {
    class MindKernel {
    public:
        explicit MindKernel(const std::vector<std::vector<float32>>& kernel) {
            this->kernelMatrix = kernel;
        }

        [[nodiscard]] std::vector<std::vector<float32>> apply(const std::vector<std::vector<float32>>& input) const {
            if (input.empty() || input[0].empty()) return {};

            const auto rows = static_cast<int32>(input.size());
            const auto cols = static_cast<int32>(input[0].size());
            const auto kRows = static_cast<int32>(this->kernelMatrix.size());
            const auto kCols = static_cast<int32>(this->kernelMatrix[0].size());
            const auto kCenterX = kCols / 2;
            const auto kCenterY = kRows / 2;

            std::vector output(rows, std::vector<float32>(cols, 0));

            for (int32 i = 0; i < rows; ++i) {
                for (int32 j = 0; j < cols; ++j) {
                    float32 sum = 0;
                    for (int32 k = 0; k < kRows; ++k) {
                        for (int32 l = 0; l < kCols; ++l) {
                            const int32 x = j + l - kCenterX;
                            if (const int32 y = i + k - kCenterY; x >= 0 && x < cols && y >= 0 && y < rows) {
                                sum += input[y][x] * this->kernelMatrix[k][l];
                            }
                        }
                    }
                    output[i][j] = sum;
                }
            }

            return output;
        }

    private:
        std::vector<std::vector<float32>> kernelMatrix;
    };
}

#endif //CORTEXMIND_KERNEL_HPP