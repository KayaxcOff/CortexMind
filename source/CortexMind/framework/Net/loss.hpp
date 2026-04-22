//
// Created by muham on 21.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_NET_LOSS_HPP
#define CORTEXMIND_FRAMEWORK_NET_LOSS_HPP

#include <CortexMind/tools/params.hpp>
#include <string>

namespace cortex::_fw {
    /**
     * @brief Base class for all loss functions.
     *
     * Provides a common interface for loss computation in neural network training.
     * All concrete loss functions (MSE, CrossEntropy, etc.) should inherit from this class.
     */
    class LossBase {
    public:
        /**
         * @brief Constructs a loss function with a given name.
         * @param name Name of the loss function (used for logging and identification)
         */
        explicit LossBase(std::string name);
        virtual ~LossBase();

        /**
         * @brief Computes the loss between prediction and target.
         *
         * @param prediction Model's output (predicted values)
         * @param target     Ground truth / target values
         * @return Loss tensor (usually a scalar tensor containing the loss value)
         */
        [[nodiscard]]
        virtual tensor forward(tensor& prediction, tensor& target) = 0;

        /**
         * @brief Returns the name of the loss function.
         */
        [[nodiscard]]
        const std::string& name() const;
    private:
        std::string kName;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_NET_LOSS_HPP