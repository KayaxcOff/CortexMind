//
// Created by muham on 4.02.2026.
//

#ifndef CORTEXMIND_CORE_GRAPH_FLOW_FLOW_HPP
#define CORTEXMIND_CORE_GRAPH_FLOW_FLOW_HPP

namespace cortex::_fw {
    class MindTensor;
} // namespace cortex::_fw

namespace cortex::_fw::meta {
    struct GradientFlow {
        virtual ~GradientFlow() = default;

        virtual void backward(MindTensor& grad_output) = 0;
    };
} // namespace cortex::_fw::meta

#endif //CORTEXMIND_CORE_GRAPH_FLOW_FLOW_HPP