//
// Created by muham on 31.05.2026.
//

#include "CortexMind/framework/Net/metric.hpp"
#include <type_traits>

using namespace cortex::_fw;

MetricBase::MetricBase(std::string name) : m_name(std::move(name)) {}

MetricBase::~MetricBase() = default;

const std::string &MetricBase::name() const {
    return this->m_name;
}