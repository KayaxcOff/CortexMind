//
// Created by muham on 19.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_GRADIENT_PACK_HPP
#define CORTEXMIND_FRAMEWORK_GRADIENT_PACK_HPP

#include <memory>

namespace cortex::_fw {
    class Tensor;
    struct TensorStorage;
    struct TensorShape;

    namespace meta {
        struct GradientFlow;
    } //namespace meta

} //namespace cortex::_fw

namespace cortex::_fw::meta {
    /**
     * @brief Packed representation of a tensor's state for gradient flow.
     *
     * Used internally by the autograd system to store and reconstruct tensor
     * information needed during the backward pass.
     */
    struct GradientPacked {
        /** @brief Shared pointer to the underlying tensor storage. */
        std::shared_ptr<TensorStorage> stor;

        /** @brief Auto-grad function */
        std::shared_ptr<GradientFlow> flow;

        /** @brief Gradient tensor (if requires_grad is true). */
        std::shared_ptr<Tensor> gradient;

        /** @brief Shape of the tensor. */
        TensorShape shape;

        /** @brief Whether this tensor requires gradient computation. */
        bool has_gradient{false};
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRADIENT_PACK_HPP