//
// Created by muham on 17.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_GRAPH_PACK_HPP
#define CORTEXMIND_FRAMEWORK_GRAPH_PACK_HPP

#include <CortexMind/framework/Graph/flow.hpp>
#include <CortexMind/framework/Type/dtype.hpp>
#include <CortexMind/runtime/macros.hpp>
#include <tlx/vector.hpp>
#include <memory>

namespace cortex::_fw {
    class TensorStorage;
    class Tensor;
} //namespace cortex::_fw

namespace cortex::_fw::meta {
    struct GradientPacked {
        GradientPacked(bool require_grad, const tlx::vec<std::int64_t, CXM_MAX_DIMS> &_shape, DType _type, const std::shared_ptr<TensorStorage> &_storage, const std::shared_ptr<GradientFlow> &_flow, const std::shared_ptr<Tensor> &_grad);

        friend class ::cortex::_fw::Tensor;
    private:
        std::shared_ptr<TensorStorage> storage;
        std::shared_ptr<GradientFlow> flow;
        std::shared_ptr<Tensor> grad;
        tlx::vec<std::int64_t, CXM_MAX_DIMS> shape;
        DType type;
        bool require_grad;
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRAPH_PACK_HPP