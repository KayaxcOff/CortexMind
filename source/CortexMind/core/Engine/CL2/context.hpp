//
// Created by muham on 21.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CL2_CONTEXT_HPP
#define CORTEXMIND_CORE_ENGINE_CL2_CONTEXT_HPP

#include <CortexMind/core/Engine/CL2/params.hpp>

namespace cortex::_fw::cl2 {
    class runtime {
    public:
        static runtime& get();

        [[nodiscard]]
        const cl::Context& context() const;
        [[nodiscard]]
        const cl::CommandQueue& queue()   const;
        [[nodiscard]]
        const cl::Device& device()  const;

        runtime(const runtime&)            = delete;
        runtime& operator=(const runtime&) = delete;
    private:
        runtime();

        cl::Platform     m_platform;
        cl::Device       m_device;
        cl::Context      m_ctx;
        cl::CommandQueue m_queue;

        static void select_platform(::cl::Platform& out);
        static void select_device(const ::cl::Platform& p, ::cl::Device& out);
    };
} // namespace cortex::_fw::cl2

#endif //CORTEXMIND_CORE_ENGINE_CL2_CONTEXT_HPP