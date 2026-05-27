//
// Created by muham on 27.05.2026.
//

#include "CortexMind/framework/Tools/series.hpp"

using namespace cortex::_fw;

Series::Series(std::vector<f32> &data) : m_data(data) {}

Series::~Series() = default;

void Series::normalize(f32 value) {

}

void Series::scale() {

}

std::vector<f32>& Series::data() {
    return this->m_data;
}