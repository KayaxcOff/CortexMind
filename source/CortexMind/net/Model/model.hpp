//
// Created by muham on 24.05.2026.
//

#ifndef CORTEXMIND_NET_MODEL_MODEL_HPP
#define CORTEXMIND_NET_MODEL_MODEL_HPP

#include <CortexMind/framework/Net/layer.hpp>
#include <CortexMind/framework/Net/loss.hpp>
#include <CortexMind/framework/Net/optimization.hpp>
#include <CortexMind/utility/DataFrame/frame.hpp>
#include <concepts>
#include <memory>
#include <vector>
#include <type_traits>

namespace cortex::net {
    /**
     * @brief High-level neural network model.
     *
     * This class acts as a container and trainer for a sequence of layers.
     * It provides a simple interface similar to Keras/PyTorch for building,
     * compiling, training, and evaluating neural networks.
     */
    class Model {
    public:
        /**
         * @brief Constructs a Model.
         *
         * @param name Optional name for the model (used in summary and logging)
         */
        explicit Model(std::string name = "");
        ~Model();

        /**
         * @brief Adds a layer to the model.
         *
         * @tparam T Layer type (must derive from LayerBase)
         * @tparam Args Constructor argument types for the layer
         * @param args Arguments forwarded to the layer's constructor
         */
        template<typename T, typename... Args> requires std::derived_from<T, _fw::LayerBase>
        void add(Args&&... args) {
            this->layers_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
        }

        /**
         * @brief Compiles the model with a loss function and optimizer.
         *
         * @tparam LossT Loss function type (must derive from LossBase)
         * @tparam OptT  Optimizer type (must derive from OptimizationBase)
         * @tparam Args  Constructor arguments for the optimizer
         * @param args   Arguments forwarded to the optimizer constructor
         */
        template<typename LossT, typename OptT, typename... Args>
        void compile(Args&&... args) {
            static_assert(std::is_base_of_v<_fw::LossBase, LossT>);
            static_assert(std::is_base_of_v<_fw::OptimizationBase, OptT>);

            if (this->m_flag) {
                CXM_WARN(true, "Model is already compiled");
                return;
            }

            this->loss_fn_ = std::make_unique<LossT>();
            this->optim_fn_ = std::make_unique<OptT>(std::forward<Args>(args)...);
            this->m_flag = true;
        }

        /**
         * @brief Trains the model using the specified data.
         *
         * @param Xx        Input features
         * @param Xy        Target labels
         * @param epochs    Number of training epochs
         * @param epochIdx  Every inc epoch to console
         */
        void fit(const tensor& Xx, const tensor& Xy, int32 epochs, int32 epochIdx = 1) const;
        /**
         * @brief Prints a summary of the model architecture.
         */
        void summary() const;
        /**
         * @brief Sets all layers to training mode.
         */
        void train() const;
        /**
         * @brief Sets all layers to evaluation mode.
         */
        void eval() const;

        /**
         * @brief Returns whether any layer in the model is trainable.
         */
        [[nodiscard]]
        bool trainable() const;
        /**
         * @brief Performs inference (forward pass) on the input.
         *
         * @param x Input tensor
         * @return Output tensor after passing through all layers
         */
        [[nodiscard]]
        tensor predict(const tensor& x) const;
        /**
         * @brief Returns all trainable parameters in the model.
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> parameters() const;
        /**
         * @brief Returns gradients of all trainable parameters.
         */
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> gradients() const;
    private:
        std::vector<std::unique_ptr<_fw::LayerBase>> layers_;
        std::unique_ptr<_fw::LossBase> loss_fn_;
        std::unique_ptr<_fw::OptimizationBase> optim_fn_;
        bool m_flag;

        std::string m_name;

        /**
         * @brief Computes total number of trainable parameters.
         */
        [[nodiscard]]
        size_t compute_element() const;
    };
} //namespace cortex::net

#endif //CORTEXMIND_NET_MODEL_MODEL_HPP