//
// Created by muham on 15.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_GRADIENT_FLOW_HPP
#define CORTEXMIND_FRAMEWORK_GRADIENT_FLOW_HPP

#include <CortexMind/framework/Gradient/pack.hpp>
#include <string>

namespace cortex::_fw {
    class Tensor;
} //namespace cortex::_fw

namespace cortex::_fw::meta {
    /**
     * @brief Abstract base class for gradient flow tracking in the autograd system.
     *
     * Every differentiable operation creates a `GradientFlow` node that knows
     * how to compute gradients during the backward pass.
     */
    struct GradientFlow {
        /**
         * @brief Constructs a new gradient flow node.
         *
         * @param _op_name Name of the operation (for debugging and visualization)
         */
        explicit GradientFlow(std::string _op_name);
        virtual ~GradientFlow();

        /**
         * @brief Computes the gradients of inputs given the output gradient.
         *
         * This is the core method of the backward pass. Each derived class
         * must implement how to propagate gradients to its inputs.
         *
         * @param _grad Gradient of the output tensor (incoming gradient)
         */
        virtual void backward(const Tensor& _grad) = 0;

        /**
         * @brief Returns the name of the operation.
         *
         * Useful for debugging, graph visualization, and profiling.
         */
        [[nodiscard]]
        const std::string& name() const;

        /**
         * Writes name of op to console
         */
        void print() const;
    private:
        std::string m_name;        ///< Operation name (e.g. "Add", "MatMul", "ReLU")
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRADIENT_FLOW_HPP