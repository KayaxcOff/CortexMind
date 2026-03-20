//
// Created by muham on 18.03.2026.
//

#ifndef CORTEXMIND_CORE_NET_OPTIMIZATION_HPP
#define CORTEXMIND_CORE_NET_OPTIMIZATION_HPP

#include <CortexMind/tools/params.hpp>
#include <CortexMind/core/Tools/ref.hpp>
#include <string>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief   Abstract base class for gradient-based optimizers
     *
     * Every concrete optimizer (SGD, Adam, RMSprop, etc.) must inherit from
     * this class and implement the `update()` method that applies one optimization step.
     */
    class Optimization {
    public:
        /**
         * @brief   Constructs optimizer with learning rate and optional name
         * @param   _lr     Initial learning rate (positive value)
         * @param   name    Optional human-readable name (for logging/debugging)
         */
        explicit Optimization(float32 _lr, std::string name);
        Optimization(const Optimization&) = delete;
        Optimization(Optimization&&) noexcept;
        virtual ~Optimization();

        /**
         * @brief   Performs one optimization step: updates parameters using gradients
         *
         * @note    Must be called after backward pass (gradients accumulated)
         * @note    Concrete implementations apply their specific update rule
         *         (e.g. SGD: param -= lr * grad, Adam: momentum + adaptive lr, etc.)
         */
        virtual void update() = 0;

        /**
         * @brief   Sets the list of parameters and corresponding gradients to optimize
         * @param   _params   Non-owning references to trainable tensors
         * @param   _grads    Non-owning references to their gradient tensors
         *
         * @pre     _params.size() == _grads.size()
         * @pre     Each param has matching gradient (same shape/device)
         * @note    References are stored internally — valid as long as tensors exist
         */
        void setParams(std::vector<ref<tensor>> _params, std::vector<ref<tensor>> _grads);
        /**
         * @brief   Sets all gradients to zero
         *
         * @note    Convenience method — usually called after update()
         * @note    Skips tensors without gradient buffer
         */
        void zero_grad() const;
        /**
         * @brief   Returns optimizer name (for logging / configuration)
         */
        [[nodiscard]]
        const std::string& config() const;
    protected:
        /**
         * @brief   Non-owning references to gradients (updated by backward pass)
         */
        std::vector<ref<tensor>> grads;
        /**
         * @brief   Non-owning references to trainable parameters
         */
        std::vector<ref<tensor>> params;
        /**
         * @brief   Current learning rate (can be modified externally)
         */
        float32 lr;
    private:
        std::string name;
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_NET_OPTIMIZATION_HPP