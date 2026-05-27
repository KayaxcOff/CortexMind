//
// Created by muham on 27.05.2026.
//

#include "CortexMind/utility/DataFrame/frame.hpp"

using namespace cortex::_fw;
using namespace cortex::utils;

Series &DataFrame::operator[](const std::string &idx) {
    CXM_ASSERT(!this->series.contains(idx), "Column not found: " + idx);
    return this->series[idx];
}

const Series &DataFrame::operator[](const std::string &idx) const {
    CXM_ASSERT(!this->series.contains(idx), "Column not found: " + idx);
    return this->series.at(idx);
}