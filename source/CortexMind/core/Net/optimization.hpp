//
// Created by muham on 25.02.2026.
//

#ifndef CORTEXMIND_CORE_NET_OPTIMIZATION_HPP
#define CORTEXMIND_CORE_NET_OPTIMIZATION_HPP

#include <CortexMind/tools/params.hpp>
#include <CortexMind/core/Tools/ref.hpp>
#include <vector>

namespace cortex::_fw {
    class Optimization {
    public:
        explicit
        Optimization(float32 _lr, string info);
        Optimization(const Optimization&) = delete;
        Optimization(Optimization&&) = default;
        virtual ~Optimization() = default;

        virtual void update() = 0;
        void setParams(std::vector<ref<tensor>> _params, std::vector<ref<tensor>> _grads);
        void zero_grad() const;

        [[nodiscard]]
        string config() const;

        Optimization& operator=(const Optimization&) = delete;
        Optimization& operator=(Optimization&&) = default;
    protected:
        std::vector<ref<tensor>> grads;
        std::vector<ref<tensor>> params;
        float32 lr;
    private:
        string info;
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_NET_OPTIMIZATION_HPP