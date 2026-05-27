//
// Created by muham on 27.05.2026.
//

#include "CortexMind/framework/Tools/series.hpp"
#include <CortexMind/framework/Tools/err.hpp>
#include <algorithm>

using namespace cortex::_fw;

Series::Series(std::vector<f32> data) : m_data(std::move(data)) {}

Series::~Series() = default;

void Series::normalize(const f32 value) {
    CXM_ASSERT(value == 0, "Value must be non-zero.");
    for (auto& item : this->m_data) {
        item /= value;
    }
}

void Series::scale() {
    CXM_ASSERT(this->m_data.empty(), "Data is empty");

    auto [min_it, max_it] = std::minmax_element(this->m_data.begin(), this->m_data.end());

    const f32 min = *min_it;
    const f32 max = *max_it;

    if ((max - min) == 0.0f) {
        return;
    }

    for (auto& item : this->m_data) {
        item = (item - min) / (max - min);
    }
}

std::vector<f32> &Series::data() {
    return this->m_data;
}

const std::vector<f32> &Series::data() const {
    return this->m_data;
}

f32 Series::operator[](const size_t index) const {
    CXM_ASSERT(index >= this->m_data.size(), "Out of range");
    return this->m_data[index];
}
