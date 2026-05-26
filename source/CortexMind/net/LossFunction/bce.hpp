//
// Created by muham on 25.05.2026.
//

#ifndef CORTEXMIND_NET_LOSS_FUNCTION_BCE_HPP
#define CORTEXMIND_NET_LOSS_FUNCTION_BCE_HPP

#include <CortexMind/framework/Net/loss.hpp>

namespace cortex::loss {
    /**
     * @brief Binary Cross Entropy (BCE) loss function.
     *
     * Commonly used for binary classification tasks. Computes the cross-entropy
     * loss between predicted probabilities and target labels (0 or 1).
     *
     * Formula (with numerical stability clipping):
     *
     *     BCE = - ( y * log(p) + (1 - y) * log(1 - p) )
     *
     * where `p` is clamped between `eps` and `1 - eps` to avoid log(0).
     */
    class BinaryCrossEntropy : public _fw::LossBase {
    public:
        /**
         * @brief Constructs a Binary Cross Entropy loss.
         *
         * @param eps Small value to prevent log(0) (default: 1e-7)
         */
        explicit BinaryCrossEntropy(float32 eps = 1e-7f);
        ~BinaryCrossEntropy() override;

        /**
         * @brief Computes Binary Cross Entropy loss.
         *
         * @param predict Predicted probabilities (usually after sigmoid)
         * @param target  Ground truth binary labels (0 or 1)
         * @return Scalar tensor containing the average BCE loss
         */
        [[nodiscard]]
        tensor forward(const tensor &predict, const tensor &target) override;
    private:
        float32 eps;
    };
} //namespace cortex::loss

#endif //CORTEXMIND_NET_LOSS_FUNCTION_BCE_HPP