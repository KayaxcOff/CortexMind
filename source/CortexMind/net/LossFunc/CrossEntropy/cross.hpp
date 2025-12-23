//
// Created by muham on 23.12.2025.
//

#ifndef CORTEXMIND_CROSS_HPP
#define CORTEXMIND_CROSS_HPP

#include <CortexMind/framework/Net/loss.hpp>

namespace cortex::net {
    class CrossEntropy : public _fw::Loss {
    public:
        CrossEntropy();
        ~CrossEntropy() override;

        [[nodiscard]] tensor forward(const tensor& predictions, const tensor& targets) const override;
        [[nodiscard]] tensor backward(const tensor& predictions, const tensor& targets) const override;
    };
}

#endif //CORTEXMIND_CROSS_HPP