//
// Created by muham on 18.03.2026.
//

#ifndef CORTEXMIND_CORE_NET_LAYER_HPP
#define CORTEXMIND_CORE_NET_LAYER_HPP

#include <CortexMind/tools/params.hpp>
#include <CortexMind/core/Tools/ref.hpp>
#include <string>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief   Abstract base class for all neural network layers
     *
     * Defines the core interface that every layer must implement:
     *   - Forward computation
     *   - Access to trainable parameters
     *   - Access to accumulated gradients
     *   - Training / evaluation mode control
     *   - Optional name for debugging / configuration
     */
    class Layer {
    public:
        /**
         * @brief   Constructs a layer with given training mode and name
         * @param   train   Whether the layer starts in training mode
         * @param   name    Optional human-readable name (for debugging/logging)
         */
        Layer(bool train, std::string name);
        Layer(const Layer&) = delete;
        Layer(Layer&&) noexcept;
        virtual ~Layer();

        /**
         * @brief   Performs forward pass: computes output from input
         * @param   input   Input tensor (may be moved or referenced)
         * @return  Output tensor (newly allocated or view)
         *
         * @note    Input tensor may be consumed (moved-from state allowed)
         * @note    Output tensor ownership is transferred to caller
         * @note    Shape inference / broadcasting handled by concrete layer
         */
        [[nodiscard]]
        virtual tensor forward(tensor& input) = 0;
        /**
         * @brief   Returns non-owning references to all trainable parameters
         * @return  Vector of reference_wrapper<tensor> pointing to parameters
         *
         * @note    Used by optimizers to update weights
         * @note    References to remain valid as long as layer exists
         */
        [[nodiscard]]
        virtual std::vector<ref<tensor>> parameters() = 0;
        /**
         * @brief   Returns non-owning references to all parameter gradients
         * @return  Vector of reference_wrapper<tensor> pointing to gradients
         *
         * @note    Gradients are accumulated during backward pass
         * @note    Used by optimizers to apply updates
         */
        [[nodiscard]]
        virtual std::vector<ref<tensor>> gradients() = 0;

        /**
         * @brief   Returns the layer's configuration name (for logging/debugging)
         */
        [[nodiscard]]
        const std::string& config() const;
        /**
         * @brief   Switches layer to evaluation mode
         */
        void toEval();
        /**
         * @brief   Switches layer to training mode
         */
        void toTrain();
    protected:
        /**
         * @brief   Returns current training mode flag
         * @return  true if in training mode, false if in evaluation mode
         */
        [[nodiscard]]
        bool is_training() const noexcept;
    private:
        std::string name;
        bool train_flag;
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_NET_LAYER_HPP