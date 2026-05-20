//
// Created by muham on 15.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_GRADIENT_FLOW_HPP
#define CORTEXMIND_FRAMEWORK_GRADIENT_FLOW_HPP

#include <CortexMind/framework/Gradient/pack.hpp>
#include <CortexMind/framework/Tools/types.hpp>
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
         * @param ID Unique identifier for this operation node (-1 means auto-assign)
         */
        explicit GradientFlow(std::string _op_name, i32 ID = -1);
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
         * @brief Returns the unique ID of this gradient node.
         */
        [[nodiscard]]
        i32 count() const;
        /**
         * @brief Returns the name of the operation.
         *
         * Useful for debugging, graph visualization, and profiling.
         */
        [[nodiscard]]
        const std::string& name() const;
    private:
        i32 ID;                    ///< Unique operation ID
        std::string m_name;        ///< Operation name (e.g. "Add", "MatMul", "ReLU")
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRADIENT_FLOW_HPP