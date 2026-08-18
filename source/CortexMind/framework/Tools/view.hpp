//
// Created by muham on 18.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_VIEW_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_VIEW_HPP

#include <CortexMind/framework/Memory/type.hpp>
#include <CortexMind/framework/Tools/Log/w.hpp>
#include <CortexMind/framework/Type/as_string.hpp>
#include <CortexMind/framework/Type/dtype.hpp>
#include <tlx/types.hpp>
#include <cstddef>

namespace cortex::_fw {
    /**
     * @brief Non-owning view over tensor data.
     *
     * Provides access to an existing data buffer together with its data type,
     * device type, and element count.
     *
     * TensorView does not allocate, deallocate, or otherwise manage the
     * lifetime of the referenced memory.
     */
    class TensorView {
    public:
        /**
         * @brief Creates a view over an existing data buffer.
         *
         * @param data Pointer to the underlying data.
         * @param type Data type of the elements in the buffer.
         * @param device Device on which the data resides.
         * @param size Number of elements in the buffer.
         */
        TensorView(std::byte* data, DType type, DeviceType device, std::size_t size);
        TensorView(TensorView const&) = delete;
        TensorView(TensorView&&) = delete;
        ~TensorView();

        /**
         * @brief Returns a mutable pointer to the underlying data.
         *
         * @return Pointer to the viewed data buffer.
         */
        [[nodiscard]]
        std::byte* data() noexcept;
        /**
         * @brief Returns a read-only pointer to the underlying data.
         *
         * @return Const pointer to the viewed data buffer.
         */
        [[nodiscard]]
        const std::byte* data() const noexcept;
        /**
         * @brief Returns the data type of the viewed elements.
         *
         * @return Data type represented by the view.
         */
        [[nodiscard]]
        DType dtype() const noexcept;
        /**
         * @brief Returns the device on which the data resides.
         *
         * @return Device type represented by the view.
         */
        [[nodiscard]]
        DeviceType device() const noexcept;
        /**
         * @brief Returns the number of elements in the viewed buffer.
         *
         * @return Number of elements.
         */
        [[nodiscard]]
        std::size_t size() const noexcept;

        TensorView& operator=(TensorView const&) = delete;
        TensorView& operator=(TensorView&&) = delete;
    private:
        std::byte* m_data;
        DType m_type;
        DeviceType m_device;
        std::size_t m_size;
    };

    /**
     * @brief Dispatches a callable according to a runtime data type.
     *
     * Maps the specified @c DType to its corresponding C++ element type
     * and invokes the callable with that type as a template argument.
     *
     * The callable is expected to provide a templated call operator
     * accepting no runtime arguments.
     *
     * Supported mappings are:
     * - @c DType::Float32  -> @c float
     * - @c DType::Float16  -> @c tlx::half
     * - @c DType::BFloat16 -> @c tlx::bfloat16
     *
     * @tparam F Callable type.
     * @param type Runtime data type used for dispatch.
     * @param f Callable receiving the resolved type.
     *
     * @return The value returned by the dispatched callable.
     *
     * @note Unsupported data types terminate the application.
     */
    template<typename F>
    decltype(auto) dispatch(const DType type, F&& f) {
        switch (type) {
            case DType::Float32:  return f.template operator()<float>();
            case DType::Float16:  return f.template operator()<tlx::half>();
            case DType::BFloat16: return f.template operator()<tlx::bfloat16>();
            default:
                WLog(LogLevel::ERROR) << "dispatch: unsupported DType " << as_string(type);
                std::abort();
        }
    }
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_VIEW_HPP