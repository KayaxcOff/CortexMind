//
// Created by muham on 5.03.2026.
//

#ifndef CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_ADAM_HPP
#define CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_ADAM_HPP

#include <CortexMind/core/Net/optimization.hpp>
#include <CortexMind/tools/params.hpp>

namespace cortex::opt {
    class Adam : public _fw::Optimization {
    public:
        explicit
        Adam(float32 _lr = 0.0001f, float32 _beta1 = 0.9f, float32 _beta2 = 0.99f, float32 _eps = 1e-8f);
        ~Adam() override;

        void update() override;
    private:
        float32 beta1;
        float32 beta2;
        float32 eps;

        std::vector<tensor> m;
        std::vector<tensor> v;

        int64 t = 0;

        void Init();
    };
} // namespace cortex::opt

#endif //CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_ADAM_HPP