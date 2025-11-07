//
// Created by muham on 4.11.2025.
//

#ifndef CORTEXMIND_MODEL_HPP
#define CORTEXMIND_MODEL_HPP

#include <memory>
#include <type_traits>
#include "../Layers/layer.hpp"
#include "../Loss/loss.hpp"
#include "../Optimizer/optimizer.hpp"

namespace cortex::model {
    class Model {
    public:
        template <typename T, typename... Args>
        void add(Args&&... args) {
            layers_.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
        }


        template<typename L, typename O>
        void compile(double _lr=0.1) {
            std::vector<math::MindVector*> all_params, all_grads;
            for (const auto& layer : layers_) {
                /*auto params = layer->get_parameters();
                auto grads = layer->get_gradients();

                all_params.insert(all_params.end(), params.begin(), params.end());
                all_grads.insert(all_grads.end(), grads.begin(), grads.end());*/
                for (const auto param : layer->get_parameters()) {
                    all_params.push_back(param);
                }
                for (const auto grad : layer->get_gradients()) {
                    all_grads.push_back(grad);
                }

            }
            loss_fn_ = std::make_unique<L>();
            optim_fn_ = std::make_unique<O>(all_params, all_grads, _lr);
        }

        void fit(const std::vector<math::MindVector> &X, const std::vector<math::MindVector> &Y, size_t epochs=1);

        [[nodiscard]] math::MindVector predict(const math::MindVector &X) const;
    private:
        std::vector<std::unique_ptr<layer::Layer>> layers_;
        std::unique_ptr<loss::Loss> loss_fn_;
        std::unique_ptr<optim::Optimizer> optim_fn_;
    };
}

#endif //CORTEXMIND_MODEL_HPP