//
// Created by muham on 24.05.2026.
//

#ifndef CORTEXMIND_NET_MODEL_MODEL_HPP
#define CORTEXMIND_NET_MODEL_MODEL_HPP

#include <CortexMind/framework/Net/layer.hpp>
#include <CortexMind/framework/Net/loss.hpp>
#include <CortexMind/framework/Net/optimization.hpp>
#include <memory>
#include <vector>

namespace cortex::net {
    class Model {
    public:
        Model();
        ~Model();

        template<typename T, typename... Args>
        void add(Args&&... args) {
            static_assert(std::is_base_of_v<_fw::LayerBase, T>, "T must derive from LayerBase");
            this->layers_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
        }

        template<typename LossT, typename OptT, typename... LossArgs, typename... OptimArgs>
        void compile(LossArgs&&... loss, OptimArgs&&... optim) {
            this->loss_fn_ = std::make_unique<LossT>(std::forward<LossArgs>(loss)...);
            this->optimization_fn_ = std::make_unique<OptT>(std::forward<OptimArgs>(optim)...);
            this->flag = true;
        }

        void fit(const tensor& Xx, const tensor& Xy, int32 epochs, int32 batch) const;
        void summary() const;
        void toTrain() const;
        void toEval() const;

        [[nodiscard]]
        bool trainable() const;
        [[nodiscard]]
        tensor predict(const tensor& x) const;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> parameters() const;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> gradients() const;
    private:
        std::vector<std::unique_ptr<_fw::LayerBase>> layers_;
        std::unique_ptr<_fw::LossBase> loss_fn_;
        std::unique_ptr<_fw::OptimizationBase> optimization_fn_;
        bool flag;
    };
} //namespace cortex::net

#endif //CORTEXMIND_NET_MODEL_MODEL_HPP