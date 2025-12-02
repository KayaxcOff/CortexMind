//
// Created by muham on 2.12.2025.
//

#ifndef CORTEXMIND_CEL_HPP
#define CORTEXMIND_CEL_HPP

#include <CortexMind/framework/NetBase/loss.hpp>

namespace cortex::loss {
    class CrossEntropy : public Loss {
    public:
        explicit CrossEntropy(double _epsilon = 1e-9);
        ~CrossEntropy() override = default;

        tensor forward(const tensor &predictions, const tensor &targets) override;
        tensor backward(const tensor &predictions, const tensor &targets) override;
    private:
        double epsilon;
    };
}

#endif //CORTEXMIND_CEL_HPP