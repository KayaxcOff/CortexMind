//
// Created by muham on 21.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_NET_OPTIMIZATION_HPP
#define CORTEXMIND_FRAMEWORK_NET_OPTIMIZATION_HPP

#include <CortexMind/tools/params.hpp>
#include <string>
#include <vector>

namespace cortex::_fw {
    class OptimizationBase {
    public:
        explicit OptimizationBase(std::string name, float32 _lr = 0.001f);
        virtual ~OptimizationBase();

        virtual void update() = 0;

        void setParams(std::vector<tensor> params);
        void setZeroGradient() const;
        [[nodiscard]]
        const std::string& getName() const;
    private:
        std::vector<tensor> kGradients;
        std::string kName;
        float32 kLearningRate;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_NET_OPTIMIZATION_HPP