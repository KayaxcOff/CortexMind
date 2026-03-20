//
// Created by muham on 18.03.2026.
//

#ifndef CORTEXMIND_CORE_NET_LOSS_HPP
#define CORTEXMIND_CORE_NET_LOSS_HPP

#include <CortexMind/tools/params.hpp>
#include <string>

namespace cortex::_fw {
    /**
     * @brief   Abstract base class for all loss (objective) functions
     *
     * Every concrete loss (MSELoss, CrossEntropyLoss, BCELoss, etc.) must inherit
     * from this class and implement the forward pass that computes the scalar loss.
     */
    class Loss {
    public:
        /**
         * @brief   Constructs a loss function with optional name
         * @param   name    Human-readable name (for logging/debugging/serialization)
         */
        explicit Loss(std::string name);
        Loss(const Loss&) = delete;
        Loss(Loss&&) noexcept ;
        virtual ~Loss();

        /**
         * @brief   Computes the loss between predictions and targets
         * @param   predicted   Model output tensor (logits, probabilities, regression values, etc.)
         * @param   target      Ground-truth tensor (labels, one-hot, regression targets, etc.)
         * @return  Scalar tensor containing the loss value (shape {1} or empty)
         *
         * @note    Output is always reduced to a scalar (mean or sum over batch)
         * @note    Shape compatibility and reduction logic are loss-specific
         * @note    Many losses expect floating-point tensors; integer targets may need casting
         */
        [[nodiscard]]
        virtual tensor forward(tensor& predicted, tensor& target) = 0;

        /**
         * @brief   Returns the loss function's name (for logging / configuration)
         */
        [[nodiscard]]
        const std::string& config() const;
    private:
        std::string name;
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_NET_LOSS_HPP