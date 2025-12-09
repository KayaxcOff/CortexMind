//
// Created by muham on 6.12.2025.
//

#ifndef CORTEXMIND_MODEL_HPP
#define CORTEXMIND_MODEL_HPP

#include <CortexMind/framework/Net/activ.hpp>
#include <CortexMind/framework/Net/layer.hpp>
#include <CortexMind/framework/Net/loss.hpp>

namespace cortex::net {
    class ModelNet {
    public:
        template<typename T, typename... Args>
        void add(Args&&... args) {
            this->layers_.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
        }

        template<typename LossT, typename OptimT>
        void compile(double learning_rate) {
            this->loss_fn_ = std::make_unique<LossT>();
            this->activ_fn_ = std::make_unique<OptimT>(learning_rate);
        }

        void summary() const {
            for (auto& item : this->layers_) {
                std::cout << "\n" << item->config() << std::endl;
            }
        }
    private:
        std::vector<std::unique_ptr<_fw::Layer>> layers_;
        std::unique_ptr<_fw::ActivationFunc> activ_fn_;
        std::unique_ptr<_fw::LossFunc> loss_fn_;
    };
}

#endif //CORTEXMIND_MODEL_HPP