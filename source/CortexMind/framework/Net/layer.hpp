//
// Created by muham on 21.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_NET_LAYER_HPP
#define CORTEXMIND_FRAMEWORK_NET_LAYER_HPP

#include <CortexMind/framework/Tools/ref.hpp>
#include <CortexMind/tools/params.hpp>
#include <string>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief Base class for all neural network layers.
     *
     * Provides common functionality for layers such as naming, training/evaluation mode,
     * and virtual interfaces for forward pass and parameter access.
     *
     * All concrete layer implementations should inherit from this class.
     */
    class LayerBase {
    public:
        /**
         * @brief Constructs a new layer with a given name.
         * @param name       Name of the layer (used for identification and debugging)
         * @param _train_flag Initial training mode (default: true)
         */
        explicit LayerBase(std::string name, boolean _train_flag = true);
        virtual ~LayerBase();

        /**
         * @brief Performs the forward pass of the layer.
         * @param input Input tensor
         * @return Output tensor after applying the layer's transformation
         */
        [[nodiscard]]
        virtual tensor forward(tensor& input) = 0;
        /**
         * @brief Returns references to all trainable weights/parameters of the layer.
         * @return Vector of reference_wrapper to weight tensors
         */
        [[nodiscard]]
        virtual std::vector<ref<tensor>> getWeight() = 0;
        /**
         * @brief Returns references to the gradients of all trainable parameters.
         * @return Vector of reference_wrapper to gradient tensors
         */
        [[nodiscard]]
        virtual std::vector<ref<tensor>> getGradient() = 0;

        /**
         * @brief Sets the layer to training mode (enables gradient computation, dropout, etc.).
         */
        void TrainMode();
        /**
         * @brief Sets the layer to evaluation mode (disables training-specific behaviors).
         */
        void EvalMode();
        /**
         * @brief Returns the name of the layer.
         */
        [[nodiscard]]
        const std::string& name() const;
    private:
        std::string m_name;
        boolean train_flag;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_NET_LAYER_HPP