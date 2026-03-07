//
// Created by muham on 22.02.2026.
//

#ifndef CORTEXMIND_CORE_GRAPH_FLOW_HPP
#define CORTEXMIND_CORE_GRAPH_FLOW_HPP

#include <vector>

namespace cortex::_fw {
    class MindTensor;
} // namespace cortex::_fw

namespace cortex::_fw::meta {

    struct GradientFlow {
        virtual ~GradientFlow() = default;

        virtual void backward(MindTensor& _grad) = 0;
        virtual std::vector<MindTensor*> inputs() = 0;
        [[nodiscard]]
        size_t count();
    };
} // namespace cortex::_fw::meta

#endif //CORTEXMIND_CORE_GRAPH_FLOW_HPP