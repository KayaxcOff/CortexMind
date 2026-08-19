//
// Created by muham on 19.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_SCRATCH_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_SCRATCH_HPP

#include <cwchar>

namespace cortex::_fw {
    struct Scratch {
        explicit Scratch(std::size_t size);
        Scratch(const Scratch&) = delete;
        Scratch(Scratch&&) = delete;
        ~Scratch();

        [[nodiscard]]
        float* data() noexcept;
        [[nodiscard]]
        const float* data() const noexcept;
        [[nodiscard]]
        std::size_t size() const noexcept;

        Scratch& operator=(const Scratch&) = delete;
        Scratch& operator=(Scratch&&) = delete;
    private:
        float* m_data;
        std::size_t m_size;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_SCRATCH_HPP