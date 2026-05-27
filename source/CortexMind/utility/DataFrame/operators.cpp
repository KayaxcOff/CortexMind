//
// Created by muham on 27.05.2026.
//

#include "CortexMind/utility/DataFrame/column.hpp"
#include "CortexMind/utility/DataFrame/frame.hpp"

using namespace cortex::utils;
using namespace cortex;

float32 &Column::operator[](const size_t i) {
    return this->m_data[i];
}

const float32 &Column::operator[](const size_t i) const {
    return this->m_data[i];
}

Column DataFrame::operator[](const std::string& col) {
    return Column(this->m_data.at(col), col);
}

Column DataFrame::operator[](const std::string& col) const {
    return Column(
        const_cast<std::vector<float32>&>(this->m_data.at(col)),
        col
    );
}