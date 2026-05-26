//
// Created by muham on 26.05.2026.
//

#ifndef CORTEXMIND_NET_OPTIMIZATION_FUNCTION_ADAM_HPP
#define CORTEXMIND_NET_OPTIMIZATION_FUNCTION_ADAM_HPP

#include <CortexMind/framework/Net/optimization.hpp>
#include <vector>

namespace cortex::opt {
    class Adam : public _fw::OptimizationBase {
    public:
        explicit Adam(float32 lr = 0.001f, float32 beta1 = 0.9f, float32 beta2 = 0.999f, float32 eps = 1e-8f);
        ~Adam() override;

        void update() override;
    private:
        float32 beta1;
        float32 beta2;
        float32 eps;
        int32 t;

        std::vector<tensor> m;
        std::vector<tensor> v;

        void Init();
        bool flag;
    };
} //namespace cortex::opt

#endif //CORTEXMIND_NET_OPTIMIZATION_FUNCTION_ADAM_HPP