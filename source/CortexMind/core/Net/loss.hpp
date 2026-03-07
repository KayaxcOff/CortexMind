//
// Created by muham on 25.02.2026.
//

#ifndef CORTEXMIND_CORE_NET_LOSS_HPP
#define CORTEXMIND_CORE_NET_LOSS_HPP

#include <CortexMind/tools/params.hpp>

namespace cortex::_fw {
    class Loss {
    public:
        explicit
        Loss(const string& name);
        Loss(const Loss&)              = delete;
        Loss(Loss&&)                   = default;
        Loss& operator=(const Loss&)   = delete;
        Loss& operator=(Loss&&)        = default;
        virtual ~Loss()                = default;

        [[nodiscard]]
        virtual tensor forward(const tensor& predicted,const tensor& target) = 0;

        [[nodiscard]]
        string config() const;
    private:
        string name;
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_NET_LOSS_HPP