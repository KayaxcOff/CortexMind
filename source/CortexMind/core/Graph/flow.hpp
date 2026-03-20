//
// Created by muham on 15.03.2026.
//

#ifndef CORTEXMIND_CORE_GRAPH_FLOW_HPP
#define CORTEXMIND_CORE_GRAPH_FLOW_HPP

#include <CortexMind/core/Tools/params.hpp>
#include <atomic>
#include <vector>

namespace cortex::_fw {
    class MindTensor;
} // namespace cortex::_fw

namespace cortex::_fw::meta {
    /**
     * @brief Global index list for gradient structs
     */
    inline std::atomic global_counter{0};

    /**
     * @brief   Abstract base class for gradient flow / backward computation of an operation
     *
     * Each instance represents one node in the computational graph that performed
     * a differentiable operation. Concrete subclasses implement the specific
     * gradient formulas for that operation.
     *
     * @note    Instances are usually created and owned by MindTensor objects
     * @note    Backward pass is triggered recursively from the root (loss) tensor
     * @note    idx is used for debugging / ordering (topological sort)
     */
    struct GradientFlow {
        /**
         * @brief   Constructor – assigns unique operation index
         */
        explicit
        GradientFlow();
        /**
         * @brief   Virtual destructor – ensures proper cleanup of derived classes
         */
        virtual ~GradientFlow();

        /**
         * @brief   Computes gradients w.r.t. inputs and accumulates them
         * @param   _grad   Incoming gradient from downstream (∂L/∂output)
         *
         * @note    Must be implemented by concrete operations (AddFlow, MatMulFlow, etc.)
         * @note    Typically computes ∂L/∂input_i = _grad × ∂output/∂input_i
         * @note    Accumulates gradients into input tensors' .grad field
         */
        virtual void backward(MindTensor& _grad) = 0;
        /**
         * @brief   Returns list of input tensors that this operation depends on
         * @return  Vector of raw pointers to input MindTensor objects
         *
         * @note    Used during graph traversal to recurse backward
         * @note    Order should match the order of inputs in forward pass
         */
        virtual std::vector<MindTensor*> inputs() = 0;
        /**
         * @brief   Returns the operation index assigned at construction
         * @return  Unique int index
         *
         * @note    Useful for debugging, topological sorting, or graph visualization
         */
        [[nodiscard]]
        i32 count() const;
    private:
        i32 idx; ///< Unique operation index in the graph
    };
} // namespace cortex::_fw::meta

#endif //CORTEXMIND_CORE_GRAPH_FLOW_HPP