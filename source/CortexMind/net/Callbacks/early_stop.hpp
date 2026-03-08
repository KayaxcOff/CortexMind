//
// Created by muham on 5.03.2026.
//

#ifndef CORTEXMIND_NET_CALLBACKS_EARLY_STOP_HPP
#define CORTEXMIND_NET_CALLBACKS_EARLY_STOP_HPP

#include <CortexMind/core/Net/callback.hpp>
#include <CortexMind/tools/params.hpp>

namespace cortex::call {
    class EarlyStopping : public _fw::Callback {
    public:
        explicit
        EarlyStopping(int64 _patience = 5, float32 _min_delta = 1e-4f);
        ~EarlyStopping() override;

        void on_batch_end(int64_t batch, float loss) override;
        void on_batch_begin(int64_t batch) override;
        void on_epoch_begin(int64_t epoch) override;
        void on_train_end() override;
        void on_epoch_end(int64_t epoch, float32 loss, float32 acc) override;
        void on_train_begin() override;

        [[nodiscard]]
        bool shouldStop() override;
    private:
        int64 patience;
        float32 min_delta;
        float32 best_loss;
        int64 counter;
        bool stop;
    };
} // namespace cortex::net

#endif //CORTEXMIND_NET_CALLBACKS_EARLY_STOP_HPP