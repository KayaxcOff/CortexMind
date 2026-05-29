//
// Created by muham on 29.05.2026.
//

#ifndef CORTEXMIND_NET_LOSS_FUNCTION_CCE_HPP
#define CORTEXMIND_NET_LOSS_FUNCTION_CCE_HPP

#include <CortexMind/framework/Net/loss.hpp>

namespace cortex::loss {
    /**
     * @brief Categorical Cross Entropy (CCE) loss function.
     *
     * Used for multi-class classification problems. Computes the cross-entropy
     * loss between predicted probability distributions and one-hot encoded target labels.
     *
     * Formula (with numerical stability clipping):
     *
     *     CCE = - (1/N) * Σ ( y_i * log(p_i) )
     *
     * where `p` is the predicted probability distribution (usually after softmax).
     */
    class CategoricalCrossEntropy : public _fw::LossBase {
    public:
        /**
         * @brief Constructs a Categorical Cross Entropy loss.
         *
         * @param eps Small value to prevent log(0) (default: 1e-7)
         */
        explicit CategoricalCrossEntropy(float32 eps = 1e-7f);
        ~CategoricalCrossEntropy() override;

        /**
         * @brief Computes Categorical Cross Entropy loss.
         *
         * @param predict Predicted probabilities (usually after softmax)
         * @param target  One-hot encoded target labels
         * @return Scalar tensor containing the average CCE loss
         */
        [[nodiscard]]
        tensor forward(const tensor &predict, const tensor &target) override;
    private:
        float32 eps;
    };
} //namespace cortex::loss

#endif //CORTEXMIND_NET_LOSS_FUNCTION_CCE_HPP