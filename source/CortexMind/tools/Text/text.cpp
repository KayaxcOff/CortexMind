//
// Created by muham on 21.12.2025.
//

#include "CortexMind/tools/Text/text.hpp"
#include <CortexMind/framework/Tools/Debug/catch.hpp>
#include <fstream>
#include <sstream>
#include <vector>

using namespace cortex::tools;
using namespace cortex;

tensor TextVec::to_tensor(const std::string &path) {
    std::ifstream file(path);
    std::string line;
    std::vector<float> values;
    int rows = 0;
    int cols = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        int current_col = 0;
        while (std::getline(ss, cell, ',')) {
            values.push_back(std::stof(cell));
            current_col++;
        }
        if (cols == 0) cols = current_col;
        else if (cols != current_col) CXM_ASSERT(true, "Wrong number of columns");
        rows++;
    }

    tensor output(1, 1, cols, rows);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output.at(0, 0, i, j) = values[i * cols + j];
        }
    }

    return output;
}
