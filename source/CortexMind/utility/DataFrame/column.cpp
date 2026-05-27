//
// Created by muham on 27.05.2026.
//

#include "CortexMind/utility/DataFrame/column.hpp"
#include <utility>

using namespace cortex::utils;
using namespace cortex;

Column::Column(std::vector<float32> &data, std::string name) : m_data(data), m_name(std::move(name)) {}

Column::~Column() = default;

size_t Column::size() const {
    return this->m_data.size();
}

const std::string &Column::name() const {
    return this->m_name;
}

tensor Column::toTensor(_fw::sys::DeviceType dev) const {
    return {{1}, this->m_data.data(), dev};
}