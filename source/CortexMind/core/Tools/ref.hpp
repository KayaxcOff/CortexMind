//
// Created by muham on 6.02.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_REF_HPP
#define CORTEXMIND_CORE_TOOLS_REF_HPP

#include <CortexMind/core/Engine/Tensor/tensor.hpp>

namespace cortex::_fw {
    /**
     * @brief Non-owning reference alias for tensors used in the framework.
     *
     * This alias is a thin wrapper around `std::reference_wrapper<MindTensor>`
     * and is primarily used to expose parameters and gradients from layers
     * without copying underlying tensor data.
     *
     * @param T class
     *
     * @details
     * The main motivation for using `ref` instead of returning tensors by value:
     * - Avoids unnecessary tensor copies
     * - Allows optimizers to modify parameters in-place
     * - Preserves clear ownership (the layer owns the tensor)
     * - Enables safe storage inside STL containers
     *
     * Typical usage includes returning collections of parameters or gradients:
     * @code
     * std::vector<ref> parameters();
     * std::vector<ref> gradients();
     * @endcode
     *
     * @note
     * This type does not manage lifetime. The referenced tensor must outlive
     * any `ref` instance that refers to it.
     */
    template <class  T>
    using ref = std::reference_wrapper<T>;
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_TOOLS_REF_HPP