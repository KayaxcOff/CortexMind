//
// Created by muham on 18.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_GRADIENT_FLOW_HPP
#define CORTEXMIND_FRAMEWORK_GRADIENT_FLOW_HPP

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw {
    class MindTensor;
} //namespace cortex::_fw

namespace cortex::_fw::meta {
    struct GradientFlow {
        explicit GradientFlow(i32 id);
        virtual ~GradientFlow();

        virtual void backward(MindTensor* _grad) = 0;
        [[nodiscard]]
        size_t count() const;
    private:
        i32 id;
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRADIENT_FLOW_HPP