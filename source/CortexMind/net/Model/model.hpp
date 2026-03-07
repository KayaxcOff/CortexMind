//
// Created by muham on 1.03.2026.
//

#ifndef CORTEXMIND_NET_MODEL_HPP
#define CORTEXMIND_NET_MODEL_HPP

#include <CortexMind/core/Net/layer.hpp>
#include <CortexMind/core/Net/loss.hpp>
#include <CortexMind/core/Net/optimization.hpp>
#include <CortexMind/core/Net/callback.hpp>
#include <CortexMind/tools/defaults.hpp>
#include <CortexMind/tools/params.hpp>
#include <memory>
#include <vector>
#include <type_traits>
#include <utility>

namespace cortex::net {
    class Model {
    public:
        Model();
        ~Model();

        template<typename T, typename... Args>
        void add(Args&&... args) {
            static_assert(std::is_base_of_v<_fw::Layer, T>, "You must use Layer classes");
            this->layers_.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
        }

        template<typename LossT, typename OptimT, typename... ArgsOptim>
        void compile(ArgsOptim&&... optim_args) {
            static_assert(std::is_base_of_v<_fw::Loss, LossT>, "Loss must derive from _fw::Loss");
            static_assert(std::is_base_of_v<_fw::Optimization, OptimT>, "Optim must derive from _fw::Optimization");

            this->loss_fn_ = std::make_unique<LossT>();
            this->optim_fn_ = std::make_unique<OptimT>(std::forward<ArgsOptim>(optim_args)...);
            this->compile_flag = true;
        }

        void fit(tensor& _x, tensor& _y, int64 _epochs = epochs, int64 _batch = batch) const;
        void train_mode() const;
        void eval_mode() const;
        void summary() const;

        [[nodiscard]]
        tensor predict(tensor& test) const;
    private:
        std::vector<std::unique_ptr<_fw::Layer>> layers_;
        std::vector<std::unique_ptr<_fw::Callback>> callbacks_;
        std::unique_ptr<_fw::Optimization> optim_fn_;
        std::unique_ptr<_fw::Loss> loss_fn_;
        boolean compile_flag;

        [[nodiscard]]
        static float32 accuracy(const tensor& predicted, const tensor& target);
        [[nodiscard]]
        std::pair<float32, float32> evaluate(tensor& _x, const tensor& _y) const;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> collect_params() const;
        [[nodiscard]]
        std::vector<_fw::ref<tensor>> collect_grads()  const;
    };
} // namespace cortex::net

#endif //CORTEXMIND_NET_MODEL_HPP