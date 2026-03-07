//
// Created by muham on 2.03.2026.
//

#include "CortexMind/utils/Data/file.hpp"
#include <CortexMind/core/Tools/err.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <vector>

using namespace cortex::utils;
using namespace cortex;

FileNet::FileNet() = default;

FileNet::~FileNet() = default;

tensor FileNet::loadFromCSV(const string &path, bool header, bool _requires_grad) {
    std::ifstream ifs(path);
    CXM_ASSERT(ifs.is_open(), "cortex::_fw::FileNet::loadFromCSV()", "Cannot opened file");

    std::vector<float32> values;
    string line;
    size_t cols = 0;
    size_t rows = 0;

    if (header) std::getline(ifs, line);

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        string cell;
        size_t current_cols = 0;

        while (std::getline(ss, cell, ',')) {
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
            values.push_back(std::stof(cell));
            current_cols++;
        }

        if (cols == 0) cols = current_cols;
        else CXM_ASSERT(cols == current_cols, "cortex::utils::FileNet::loadFromCSV()", "CSV is not rectangular.");
        rows++;
    }
    CXM_ASSERT(rows > 0 && cols > 0, "cortex::utils::FileNet::loadFromCSV()", "CSV is empty.");

    return {{static_cast<int64>(rows), static_cast<int64>(cols)}, values.data(), _requires_grad};
}

tensor FileNet::loadFromJSON(const string &path, bool _requires_grad) {
    std::ifstream ifs(path);
    CXM_ASSERT(ifs.is_open(), "cortex::_fw::FileNet::loadFromJSON()", "Cannot opened file");

    nlohmann::json j;
    ifs >> j;

    CXM_ASSERT(j.contains("shape"), "cortex::utils::FileNet::loadFromJSON()", "Missing 'shape' key.");
    CXM_ASSERT(j.contains("data"),  "cortex::utils::FileNet::loadFromJSON()", "Missing 'data' key.");

    const std::vector<int64> shape = j["shape"].get<std::vector<int64>>();
    const std::vector<float32> data = j["data"].get<std::vector<float32>>();

    size_t expected = 1;
    for (const auto s : shape) expected *= static_cast<size_t>(s);
    CXM_ASSERT(expected == data.size(), "cortex::utils::FileNet::loadFromJSON()", "Shape and data size mismatch.");

    return {shape, data.data(), _requires_grad};
}
