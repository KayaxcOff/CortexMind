//
// Created by muham on 19.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_NET_LOSS_HPP
#define CORTEXMIND_FRAMEWORK_NET_LOSS_HPP

#include <CortexMind/tools/types.hpp>
#include <string>

namespace cortex::_fw {
    /**
     * @brief Abstract base class for loss functions.
     *
     * This class serves as the foundation for all loss functions in the framework.
     * It provides a unified interface for computing the loss between predictions
     * and target values, and supports naming for better debugging and logging.
     */
    class LossBase {
    public:
        /**
         * @brief Constructs a new loss function.
         *
         * @param name Human-readable name of the loss function (e.g. "CrossEntropy", "MSE")
         */
        explicit LossBase(std::string name);
        virtual ~LossBase();

        /**
         * @brief Computes the loss between predictions and target values.
         *
         * @param predict Predicted values (model output)
         * @param target  Ground truth / target values
         * @return Computed loss value as a scalar tensor
         */
        [[nodiscard]]
        virtual tensor forward(const tensor& predict, const tensor& target) = 0;

        /**
         * @brief Returns the name of the loss function.
         *
         * Used for logging, debugging, and model summary.
         */
        [[nodiscard]]
        const std::string& name() const;
    private:
        std::string m_name;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_NET_LOSS_HPP