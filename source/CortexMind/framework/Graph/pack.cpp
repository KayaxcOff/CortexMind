//
// Created by muham on 17.08.2026.
//

#include "CortexMind/framework/Graph/pack.hpp"

using namespace cortex::_fw::meta;

GradientPacked::GradientPacked(const bool require_grad, const tlx::vec<std::int64_t, 5> &_shape, const DType _type, const std::shared_ptr<TensorStorage> &_storage, const std::shared_ptr<GradientFlow> &_flow, const std::shared_ptr<Tensor> &_grad) {
    this->shape = _shape;
    this->type = _type;
    this->storage = _storage;
    this->flow = _flow;
    this->grad = _grad;
    this->require_grad = require_grad;
}
