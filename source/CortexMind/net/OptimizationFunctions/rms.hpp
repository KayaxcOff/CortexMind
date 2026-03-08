//
// Created by muham on 8.03.2026.
//

#ifndef CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_RMS_HPP
#define CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_RMS_HPP

#include <CortexMind/core/Net/optimization.hpp>
#include <vector>

namespace cortex::opt {
    class RMSProp : public _fw::Optimization {
    public:
        explicit
        RMSProp(float32 _lr = 0.001f, float32 _decay = 0.9f, float32 eps = 1e-8f);
        ~RMSProp() override;

        void update() override;
    private:
        float32 decay;
        float32 eps;
        std::vector<tensor> cache;
    };
} // namespace cortex::opt

#endif //CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_RMS_HPP