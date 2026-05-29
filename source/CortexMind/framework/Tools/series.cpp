//
// Created by muham on 27.05.2026.
//

#include "CortexMind/framework/Tools/series.hpp"
#include <CortexMind/framework/Tools/err.hpp>
#include <algorithm>

using namespace cortex::_fw;

Series::Series(std::vector<f32> data) : m_data(std::move(data)), m_dtype(DType::Float32) {}

Series::Series(std::vector<bool> data) : m_data(std::move(data)), m_dtype(DType::Bool) {}

Series::Series(std::vector<std::string> data) : m_data(std::move(data)), m_dtype(DType::String) {}

Series::~Series() = default;

DType Series::dtype() const {
    return this->m_dtype;
}

size_t Series::size() const {
    return std::visit([](const auto& item) { return item.size(); }, this->m_data);
}

void Series::normalize(const f32 value) {
    CXM_ASSERT(this->m_dtype != DType::Float32, "Only float32 types can be normalized.");
    CXM_ASSERT(value == 0.0f, "Value must be equal to zero.");
    for (auto& item : as<f32>()) {
        item /= value;
    }
}

void Series::scale() {
    CXM_ASSERT(m_dtype != DType::Float32, "Scale only float32 types can be normalized.");
    CXM_ASSERT(empty(), "Data must not be empty.");

    auto& vec = as<f32>();
    auto [min_it, max_it] = std::minmax_element(vec.begin(), vec.end());
    const f32 mn = *min_it, mx = *max_it;
    if ((mx - mn) == 0.0f) return;
    for (auto& item : vec) item = (item - mn) / (mx - mn);
}

bool Series::empty() const {
    return this->size() == 0;
}

std::vector<f32>& Series::data() {
    CXM_ASSERT(this->m_dtype != DType::Float32, "Data() is only for float32");
    return as<f32>();
}

const std::vector<f32>& Series::data() const {
    CXM_ASSERT(this->m_dtype != DType::Float32, "Data() is only for float32");
    return as<f32>();
}

f32 Series::operator[](const size_t index) const {
    CXM_ASSERT(this->m_dtype != DType::Float32, "Operator is only for float32");
    const auto& vec = as<f32>();
    CXM_ASSERT(index >= vec.size(), "Out of indes");
    return vec[index];
}