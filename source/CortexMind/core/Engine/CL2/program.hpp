//
// Created by muham on 22.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CL2_PROGRAM_HPP
#define CORTEXMIND_CORE_ENGINE_CL2_PROGRAM_HPP

#include <CortexMind/core/Engine/CL2/context.hpp>
#include <CortexMind/core/Engine/Memory/buffer.hpp>
#include <CortexMind/core/Tools/file_system.hpp>
#include <CortexMind/core/Tools/err.hpp>
#include <unordered_map>
#include <string>

namespace cortex::_fw::cl2 {
    class program {
    public:
        explicit
        program(const fs::path& path);
        explicit
        program(const std::string& source, std::string  name = "<inline>");
        program(program&&) noexcept = default;
        program& operator=(program&&) noexcept = default;
        ~program() = default;

        program(const program&) = delete;
        program& operator=(const program&) = delete;

        [[nodiscard]]
        cl::Kernel& kernel(const std::string& name);

        template<typename... Args>
        void run(const std::string& kernel_name, const cl::NDRange& global, const cl::NDRange& local, Args&&... args) {
            cl::Kernel& k = kernel(kernel_name);
            set_args(k, 0, std::forward<Args>(args)...);

            const cl_int err = runtime::get().queue().enqueueNDRangeKernel(
                k, cl::NullRange, global, local
            );
            CXM_ASSERT(err == CL_SUCCESS, "cortex::_fw::cl2::program::run()", "Can't run kernel: " + kernel_name);

            runtime::get().queue().finish();
        }

        [[nodiscard]]
        const std::string& name() const;
    private:
        std::string  name_;
        cl::Program prog_;
        std::unordered_map<std::string, cl::Kernel> cache_;

        void build() const;

        static void set_args(::cl::Kernel&, size_t) {}

        template<typename T, typename... Rest>
        void set_args(cl::Kernel& k, size_t idx, T&& arg, Rest&&... rest) {
            if constexpr (std::is_same_v<std::decay_t<T>, cl::Buffer>) {
                k.setArg(idx, arg);
            } else if constexpr (std::is_same_v<std::decay_t<T>, sys::buffer>) {
                k.setArg(idx, arg.handle());
            } else {
                k.setArg(idx, arg);
            }
            set_args(k, idx + 1, std::forward<Rest>(rest)...);
        }
    };

} // namespace cortex::_fw::cl2

#endif //CORTEXMIND_CORE_ENGINE_CL2_PROGRAM_HPP