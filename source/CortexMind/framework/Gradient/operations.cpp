//
// Created by muham on 20.04.2026.
//

#include "CortexMind/framework/Gradient/operations.hpp"
#include <CortexMind/framework/Tensor/tensor.hpp>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

addition::addition(const std::weak_ptr<TensorStorage> &tx_stor, const std::weak_ptr<TensorStorage> &ty_stor, const std::weak_ptr<TensorStorage> &tx_grad_stor, const std::weak_ptr<TensorStorage> &ty_grad_stor, const std::weak_ptr<GradientFlow> &tx_flow, const std::weak_ptr<GradientFlow> &ty_flow) : GradientFlow(1) {
    this->tx_stor = tx_stor;
    this->ty_stor = ty_stor;

    this->tx_grad_stor = tx_grad_stor;
    this->ty_grad_stor = ty_grad_stor;

    this->tx_flow = tx_flow;
    this->ty_flow = ty_flow;
}

void addition::backward(MindTensor *_grad) {
    const auto tx_storage = this->tx_stor.lock();
    const auto tx_grad    = this->tx_grad_stor.lock();
    const auto tx_fl      = this->tx_flow.lock();

    if (tx_storage && tx_grad) {
        MindTensor tx(tx_storage, tx_grad, tx_fl);
        tx.grad() += (*_grad);

        if (tx_fl != nullptr) {
            tx.backward(tx.grad());
        }
    }

    const auto ty_storage = this->ty_stor.lock();
    const auto ty_grad    = this->ty_grad_stor.lock();
    const auto ty_fl      = this->ty_flow.lock();

    if (ty_storage && ty_grad) {
        MindTensor ty(ty_storage, ty_grad, ty_fl);

        ty.grad() += (*_grad);

        if (ty_fl != nullptr) {
            ty.backward(ty.grad());
        }
    }
}

subtraction::subtraction(const std::weak_ptr<TensorStorage> &tx_stor, const std::weak_ptr<TensorStorage> &ty_stor, const std::weak_ptr<TensorStorage> &tx_grad_stor, const std::weak_ptr<TensorStorage> &ty_grad_stor, const std::weak_ptr<GradientFlow> &tx_flow, const std::weak_ptr<GradientFlow> &ty_flow) : GradientFlow(2) {
    this->tx_stor = tx_stor;
    this->ty_stor = ty_stor;

    this->tx_grad_stor = tx_grad_stor;
    this->ty_grad_stor = ty_grad_stor;

    this->tx_flow = tx_flow;
    this->ty_flow = ty_flow;
}

void subtraction::backward(MindTensor *_grad) {
    const auto tx_storage = this->tx_stor.lock();
    const auto ty_storage = this->ty_stor.lock();

    const auto tx_grad    = this->tx_grad_stor.lock();
    const auto ty_grad    = this->ty_grad_stor.lock();

    const auto tx_fl      = this->tx_flow.lock();
    const auto ty_fl      = this->ty_flow.lock();

    MindTensor tx(tx_storage, tx_grad, tx_fl);
    MindTensor ty(ty_storage, ty_grad, ty_fl);

    if (tx_storage && tx_grad) {
        tx.grad() += (*_grad);

        if (tx_fl != nullptr) {
            tx.backward(tx.grad());
        }
    }

    if (ty_storage && ty_grad) {
        ty.grad() -= (*_grad);

        if (ty_fl != nullptr) {
            ty.backward(ty.grad());
        }
    }
}

multiply::multiply(const std::weak_ptr<TensorStorage> &tx_stor, const std::weak_ptr<TensorStorage> &ty_stor, const std::weak_ptr<TensorStorage> &tx_grad_stor, const std::weak_ptr<TensorStorage> &ty_grad_stor, const std::weak_ptr<GradientFlow> &tx_flow, const std::weak_ptr<GradientFlow> &ty_flow) : GradientFlow(3) {
    this->tx_stor = tx_stor;
    this->ty_stor = ty_stor;

    this->tx_grad_stor = tx_grad_stor;
    this->ty_grad_stor = ty_grad_stor;

    this->tx_flow = tx_flow;
    this->ty_flow = ty_flow;
}

void multiply::backward(MindTensor *_grad) {
    const auto tx_storage = this->tx_stor.lock();
    const auto ty_storage = this->ty_stor.lock();

    const auto tx_grad    = this->tx_grad_stor.lock();
    const auto ty_grad    = this->ty_grad_stor.lock();

    const auto tx_fl      = this->tx_flow.lock();
    const auto ty_fl      = this->ty_flow.lock();

    MindTensor tx(tx_storage, tx_grad, tx_fl);
    MindTensor ty(ty_storage, ty_grad, ty_fl);

    if (tx_storage && tx_grad) {
        tx.grad() += (*_grad) * ty;

        if (tx_fl != nullptr) {
            tx.backward(tx.grad());
        }
    }

    if (ty_storage && ty_grad) {
        ty.grad() += (*_grad) * tx;

        if (ty_fl != nullptr) {
            ty.backward(ty.grad());
        }
    }
}

division::division(const std::weak_ptr<TensorStorage> &tx_stor, const std::weak_ptr<TensorStorage> &ty_stor, const std::weak_ptr<TensorStorage> &tx_grad_stor, const std::weak_ptr<TensorStorage> &ty_grad_stor, const std::weak_ptr<GradientFlow> &tx_flow, const std::weak_ptr<GradientFlow> &ty_flow) : GradientFlow(4) {
    this->tx_stor = tx_stor;
    this->ty_stor = ty_stor;

    this->tx_grad_stor = tx_grad_stor;
    this->ty_grad_stor = ty_grad_stor;

    this->tx_flow = tx_flow;
    this->ty_flow = ty_flow;
}

void division::backward(MindTensor *_grad) {
    const auto tx_storage = this->tx_stor.lock();
    const auto ty_storage = this->ty_stor.lock();

    const auto tx_grad    = this->tx_grad_stor.lock();
    const auto ty_grad    = this->ty_grad_stor.lock();

    const auto tx_fl      = this->tx_flow.lock();
    const auto ty_fl      = this->ty_flow.lock();

    MindTensor tx(tx_storage, tx_grad, tx_fl);
    MindTensor ty(ty_storage, ty_grad, ty_fl);

    if (tx_storage && tx_grad) {
        tx.grad() += (*_grad) / ty;

        if (tx_fl != nullptr) {
            tx.backward(tx.grad());
        }
    }

    if (ty_storage && ty_grad) {
        ty.grad() += ((*_grad) * tx / ty.pow(2.0f)) * (-1.0f);

        if (ty_fl != nullptr) {
            ty.backward(ty.grad());
        }
    }
}

scalar_additive::scalar_additive(const std::weak_ptr<TensorStorage>& tx_stor, const std::weak_ptr<TensorStorage>& tx_grad_stor, const std::weak_ptr<GradientFlow>&  tx_flow) : GradientFlow(5) {
    this->tx_stor      = tx_stor;
    this->tx_grad_stor = tx_grad_stor;
    this->tx_flow      = tx_flow;
}

void scalar_additive::backward(MindTensor* _grad) {
    const auto tx_storage = this->tx_stor.lock();
    const auto tx_grad    = this->tx_grad_stor.lock();
    const auto tx_fl      = this->tx_flow.lock();

    if (tx_storage && tx_grad) {
        MindTensor tx(tx_storage, tx_grad, tx_fl);
        tx.grad() += *_grad;

        if (tx_fl != nullptr) {
            tx.backward(tx.grad());
        }
    }
}

scalar_multiply::scalar_multiply(const std::weak_ptr<TensorStorage>& tx_stor, const std::weak_ptr<TensorStorage>& tx_grad_stor, const std::weak_ptr<GradientFlow>&  tx_flow, const f32 c) : GradientFlow(6), c(c) {
    this->tx_stor      = tx_stor;
    this->tx_grad_stor = tx_grad_stor;
    this->tx_flow      = tx_flow;
}

void scalar_multiply::backward(MindTensor* _grad) {
    const auto tx_storage = this->tx_stor.lock();
    const auto tx_grad    = this->tx_grad_stor.lock();
    const auto tx_fl      = this->tx_flow.lock();

    if (tx_storage && tx_grad) {
        MindTensor tx(tx_storage, tx_grad, tx_fl);
        tx.grad() += *_grad * this->c;

        if (tx_fl != nullptr) {
            tx.backward(tx.grad());
        }
    }
}

// dot_product
dot::dot(const std::weak_ptr<TensorStorage> &tx_stor, const std::weak_ptr<TensorStorage> &ty_stor, const std::weak_ptr<TensorStorage> &tx_grad_stor, const std::weak_ptr<TensorStorage> &ty_grad_stor, const std::weak_ptr<GradientFlow> &tx_flow, const std::weak_ptr<GradientFlow> &ty_flow) : GradientFlow(7) {
    this->tx_stor = tx_stor; this->ty_stor = ty_stor;
    this->tx_grad_stor = tx_grad_stor; this->ty_grad_stor = ty_grad_stor;
    this->tx_flow = tx_flow; this->ty_flow = ty_flow;
}

void dot::backward(MindTensor* _grad) {
    const auto tx_storage = this->tx_stor.lock();
    const auto ty_storage = this->ty_stor.lock();
    const auto tx_grad    = this->tx_grad_stor.lock();
    const auto ty_grad    = this->ty_grad_stor.lock();
    const auto tx_fl      = this->tx_flow.lock();
    const auto ty_fl      = this->ty_flow.lock();

    if (tx_storage && tx_grad) {
        MindTensor tx(tx_storage, tx_grad, tx_fl);
        const MindTensor ty(ty_storage, ty_grad, ty_fl);

        // dz/dx = grad * y^T
        tx.grad() += _grad->dot(ty.transpose());
        if (tx_fl) {
            tx.backward(tx.grad());
        }
    }

    if (ty_storage && ty_grad) {
        const MindTensor tx(tx_storage, tx_grad, tx_fl);
        MindTensor ty(ty_storage, ty_grad, ty_fl);

        // dz/dy = x^T * grad
        ty.grad() += tx.transpose().dot(*_grad);
        if (ty_fl) {
            ty.backward(ty.grad());
        }
    }
}

// pow_op
pow::pow(const std::weak_ptr<TensorStorage> &tx_stor, const std::weak_ptr<TensorStorage> &tx_grad_stor, const std::weak_ptr<GradientFlow> &tx_flow, f32 exp) : GradientFlow(8), exp(exp) {
    this->tx_stor = tx_stor;
    this->tx_grad_stor = tx_grad_stor;
    this->tx_flow = tx_flow;
}

void pow::backward(MindTensor* _grad) {
    const auto tx_storage = this->tx_stor.lock();
    const auto tx_grad    = this->tx_grad_stor.lock();
    const auto tx_fl      = this->tx_flow.lock();

    if (tx_storage && tx_grad) {
        MindTensor tx(tx_storage, tx_grad, tx_fl);
        // dz/dx = exp * x^(exp-1) * grad
        tx.grad() += tx.pow(this->exp - 1.0f) * this->exp * (*_grad);
        if (tx_fl) {
            tx.backward(tx.grad());
        }
    }
}

log::log(const std::weak_ptr<TensorStorage> &tx_stor, const std::weak_ptr<TensorStorage> &tx_grad_stor, const std::weak_ptr<GradientFlow> &tx_flow) : GradientFlow(9) {
    this->tx_stor = tx_stor;
    this->tx_grad_stor = tx_grad_stor;
    this->tx_flow = tx_flow;
}

void log::backward(MindTensor* _grad) {
    const auto tx_storage = this->tx_stor.lock();
    const auto tx_grad    = this->tx_grad_stor.lock();
    const auto tx_fl      = this->tx_flow.lock();

    if (tx_storage && tx_grad) {
        MindTensor tx(tx_storage, tx_grad, tx_fl);
        // dz/dx = grad / x
        tx.grad() += (*_grad) / tx;
        if (tx_fl) {
            tx.backward(tx.grad());
        }
    }
}

exp::exp(const std::weak_ptr<TensorStorage> &tx_stor, const std::weak_ptr<TensorStorage> &tx_grad_stor, const std::weak_ptr<GradientFlow> &tx_flow) : GradientFlow(10) {
    this->tx_stor = tx_stor;
    this->tx_grad_stor = tx_grad_stor;
    this->tx_flow = tx_flow;
}

// exp_op
void exp::backward(MindTensor* _grad) {
    const auto tx_storage = this->tx_stor.lock();
    const auto tx_grad    = this->tx_grad_stor.lock();
    const auto tx_fl      = this->tx_flow.lock();

    if (tx_storage && tx_grad) {
        MindTensor tx(tx_storage, tx_grad, tx_fl);
        tx.grad() += (*_grad) * tx.exp();
        if (tx_fl) {
            tx.backward(tx.grad());
        }
    }
}

sum::sum(const std::weak_ptr<TensorStorage> &tx_stor, const std::weak_ptr<TensorStorage> &tx_grad_stor, const std::weak_ptr<GradientFlow> &tx_flow) : GradientFlow(11) {
    this->tx_stor = tx_stor;
    this->tx_grad_stor = tx_grad_stor;
    this->tx_flow = tx_flow;
}

// sum_op
void sum::backward(MindTensor* _grad) {
    const auto tx_storage = this->tx_stor.lock();
    const auto tx_grad    = this->tx_grad_stor.lock();
    const auto tx_fl      = this->tx_flow.lock();

    if (tx_storage && tx_grad) {
        MindTensor tx(tx_storage, tx_grad, tx_fl);

        const f32 grad_val = _grad->get()[0];

        MindTensor ones_t(tx_storage->shape, tx_storage->device());
        ones_t.ones();
        ones_t *= grad_val;

        tx.grad() += ones_t;

        if (tx_fl) {
            tx.backward(tx.grad());
        }
    }
}