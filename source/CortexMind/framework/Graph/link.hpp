//
// Created by muham on 16.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_GRAPH_LINK_HPP
#define CORTEXMIND_FRAMEWORK_GRAPH_LINK_HPP

#include <CortexMind/framework/Graph/flow.hpp>
#include <memory>

namespace cortex::_fw::meta {
    /**
     * @brief Links a tensor to its gradient flow.
     *
     * Associates the specified tensor with a gradient flow responsible for
     * propagating the derivative of the operation that produced the tensor.
     *
     * This is primarily used by activation and other differentiable layers
     * to attach the derivative operation of an output tensor to the
     * corresponding gradient computation graph.
     */
    class GradientLink {
    public:
        /**
         * @brief Associates a tensor with a gradient flow.
         *
         * Assigns the specified gradient flow to the tensor, establishing the
         * connection required for gradient propagation during backpropagation.
         *
         * @param t Tensor whose gradient flow will be associated.
         * @param flow Gradient flow representing the derivative operation
         *             associated with the tensor.
         */
        GradientLink(Tensor& t, const std::shared_ptr<GradientFlow> &flow);
        GradientLink(const GradientLink&) = delete;
        GradientLink(GradientLink&&) = delete;
        ~GradientLink();

        GradientLink& operator=(const GradientLink&) = delete;
        GradientLink& operator=(GradientLink&&) = delete;
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRAPH_LINK_HPP