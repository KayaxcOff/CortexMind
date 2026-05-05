//
// Created by muham on 21.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_NET_OPTIMIZATION_HPP
#define CORTEXMIND_FRAMEWORK_NET_OPTIMIZATION_HPP

#include <CortexMind/framework/Tools/ref.hpp>
#include <CortexMind/tools/params.hpp>
#include <string>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief Base class for all optimizers (SGD, Adam, RMSprop, etc.).
     *
     * Provides common functionality such as parameter registration,
     * zeroing gradients, and learning rate management.
     *
     * All concrete optimizers should inherit from this class and implement
     * the `update()` method.
     */
    class OptimizationBase {
    public:
        /**
         * @brief Constructs an optimizer with a name and learning rate.
         * @param name Name of the optimizer (used for logging/identification)
         * @param _lr  Initial learning rate (default: 0.001)
         */
        explicit OptimizationBase(std::string name, float32 _lr = 0.001f);
        virtual ~OptimizationBase();

        /**
         * @brief Performs one optimization step (parameter update).
         *
         * Must be implemented by derived classes.
         */
        virtual void update() = 0;

        /**
         * @brief Registers the parameters (and their gradients) to be optimized.
         * @param _params Vector of references to trainable tensors
         */
        void setParams(std::vector<ref<tensor>>& _params);
        /**
         * @brief Sets all registered gradients to zero.
         *
         * Typically called before the backward pass in each training iteration.
         */
        void zero_grad() const;
        /**
         * @brief Returns the name of the optimizer.
         */
        [[nodiscard]]
        const std::string& name() const;
    protected:
        std::vector<ref<tensor>> params;
        float32 learning_rate;
    private:
        std::string m_name;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_NET_OPTIMIZATION_HPP