//
// Created by muham on 19.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_NET_OPTIMIZATION_HPP
#define CORTEXMIND_FRAMEWORK_NET_OPTIMIZATION_HPP

#include <CortexMind/framework/Tools/ref.hpp>
#include <CortexMind/tools/types.hpp>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief Abstract base class for all optimizers.
     *
     * This class provides the foundation for optimization algorithms.
     * It manages parameters, learning rate, and gradient zeroing.
     */
    class OptimizationBase {
    public:
        /**
         * @brief Constructs a new optimizer.
         *
         * @param name Name of the optimizer (e.g. "SGD", "Adam")
         * @param _lr  Learning rate (default: 0.001)
         */
        explicit OptimizationBase(std::string name, float32 _lr = 0.001);
        virtual ~OptimizationBase();

        /**
         * @brief Performs a single optimization step.
         *
         * This is the core method that updates the model parameters based on their gradients.
         * Must be implemented by all derived optimizers.
         */
        virtual void update() = 0;

        /**
         * @brief Sets the parameters to be optimized.
         *
         * @param params List of tensors (usually model parameters) to optimize
         */
        void set_params(const std::vector<ref<tensor>>& params);
        /**
         * @brief Sets all parameter gradients to zero.
         *
         * Should be called before each backward pass.
         */
        void zero_grad() const;
        /**
         * @brief Returns the list of parameters being optimized.
         */
        [[nodiscard]]
        const std::vector<ref<tensor>>& parameters();
        /**
         * @brief Returns the current learning rate.
         */
        [[nodiscard]]
        float32 lr() const;
        /**
         * @brief Returns the name of the optimizer.
         */
        [[nodiscard]]
        const std::string& name();
    private:
        std::vector<ref<tensor>> m_params;
        float32 m_lr;
        std::string m_name;
        bool is_initialized;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_NET_OPTIMIZATION_HPP