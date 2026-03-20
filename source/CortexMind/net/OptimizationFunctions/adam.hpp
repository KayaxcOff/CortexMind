//
// Created by muham on 18.03.2026.
//

#ifndef CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_ADAM_HPP
#define CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_ADAM_HPP

#include <CortexMind/core/Net/optimization.hpp>

namespace cortex::opt {
    class Adam : public _fw::Optimization {
    public:
        explicit Adam(float32 lr, float32 beta1 = 0.9f, float32 beta2 = 0.999f, float32 eps = 1e-8f);
        ~Adam() override;

        void update() override;
        //void setParams(std::vector<_fw::ref<tensor>> _params, std::vector<_fw::ref<tensor>> _grads);
    private:
        float32 beta1, beta2, eps;
        int64 step;
        std::vector<tensor> m_state;
        std::vector<tensor> v_state;
    };
} // namespace cortex::opt

#endif // CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_ADAM_HPP