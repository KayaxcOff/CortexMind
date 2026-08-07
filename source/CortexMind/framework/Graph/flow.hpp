//
// Created by muham on 7.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_GRAPH_FLOW_HPP
#define CORTEXMIND_FRAMEWORK_GRAPH_FLOW_HPP

#include <tlx/string.hpp>
#include <string_view>

namespace cortex::_fw {
    class Tensor;
    class TensorDebug;

    namespace meta {
        struct GradientFlow {
            explicit GradientFlow(tlx::vstring name);
            virtual ~GradientFlow();

            virtual void backward(const Tensor& _grad) = 0;
            [[nodiscard]]
            std::string_view ToString() const noexcept;

            friend class cortex::_fw::TensorDebug;
        private:
            tlx::vstring m_name;
        };
    } //namespace meta
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_GRAPH_FLOW_HPP