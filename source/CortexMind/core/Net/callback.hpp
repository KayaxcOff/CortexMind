//
// Created by muham on 5.03.2026.
//

#ifndef CORTEXMIND_CORE_NET_CALLBACK_HPP
#define CORTEXMIND_CORE_NET_CALLBACK_HPP

#include <CortexMind/tools/params.hpp>

namespace cortex::_fw {
    class Callback {
    public:
        explicit
        Callback(string  name);
        virtual ~Callback() = default;

        virtual void on_epoch_begin(int64_t epoch) = 0;
        virtual void on_epoch_end(int64_t epoch, float loss, float acc) = 0;
        virtual void on_train_begin() = 0;
        virtual void on_train_end() = 0;
        virtual void on_batch_begin(int64_t batch) = 0;
        virtual void on_batch_end(int64_t batch, float loss) = 0;
        [[nodiscard]]
        virtual bool shouldStop() = 0;

        [[nodiscard]]
        string config() const;
    private:
        string name;
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_NET_CALLBACK_HPP