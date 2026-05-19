//
// Created by muham on 19.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_NET_LAYER_HPP
#define CORTEXMIND_FRAMEWORK_NET_LAYER_HPP

#include <CortexMind/framework/Tools/ref.hpp>
#include <CortexMind/tools/types.hpp>
#include <string>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief Abstract base class for all neural network layers.
     *
     * This class serves as the foundation for all layers in the framework.
     * It provides common functionality such as naming, training mode management,
     * and abstract interfaces for forward propagation and parameter handling.
     */
    class LayerBase {
    public:
        /**
         * @brief Constructs a new layer.
         *
         * @param name       Name of the layer (used for identification and debugging)
         * @param _train_flag Initial training mode (default: true)
         */
        explicit LayerBase(std::string name, bool _train_flag = true);
        virtual ~LayerBase();

        /**
         * @brief Performs the forward pass of the layer.
         *
         * @param input Input tensor
         * @return Output tensor after applying the layer's transformation
         */
        [[nodiscard]]
        virtual tensor forward(const tensor& input) = 0;
        /**
         * @brief Returns all trainable parameters of the layer.
         *
         * Used by optimizers to update weights during training.
         */
        [[nodiscard]]
        virtual std::vector<ref<tensor>> getParameters() = 0;
        /**
         * @brief Returns gradients of all trainable parameters.
         *
         * Used for gradient clipping, inspection, and optimizer steps.
         */
        [[nodiscard]]
        virtual std::vector<ref<tensor>> getGradients() = 0;

        /**
         * @brief Returns the name of the layer.
         */
        [[nodiscard]]
        const std::string& name() const;
        /**
         * @brief Returns whether the layer is in training mode.
         */
        [[nodiscard]]
        bool flag() const;
        /**
         * @brief Sets the layer to training mode.
         *
         * Enables behaviors like dropout, batch normalization in training mode, etc.
         */
        void TrainMode();
        /**
         * @brief Sets the layer to evaluation mode.
         *
         * Disables training-specific behaviors (e.g., dropout is turned off).
         */
        void EvalMode();
    private:
        std::string m_name;
        bool m_train_flag;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_NET_LAYER_HPP