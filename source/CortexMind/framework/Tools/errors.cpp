//
// Created by muham on 3.08.2026.
//

#include "CortexMind/framework/Tools/errors.hpp"
#include <CortexMind/runtime/macros.hpp>
#include <iostream>

using namespace cortex::_fw;

namespace {
    constexpr auto RED    = "\033[31m";
    constexpr auto YELLOW = "\033[33m";
    constexpr auto RESET  = "\033[0m";

    enum class LogLevel : std::uint8_t {
        Red = 0,
        Yellow = 1,
    };

    [[nodiscard]]
    constexpr const char* colour(const LogLevel level) {
        switch (level) {
            case LogLevel::Red:
                return RED;
            case LogLevel::Yellow:
                return YELLOW;
        }
        __assume(false);
    }

    class Log {
    public:
        Log(std::ostream& os, const LogLevel level) : m_os(os)  {
            this->m_level = level;
            this->m_os << colour(level);
        }

        ~Log() {
            this->m_os << RESET;
        }

        template<typename T>
        Log& operator<<(const T& value) {
            this->m_os << value;
            return *this;
        }

    private:
        std::ostream& m_os;
        LogLevel m_level;
    };
} //unnamed namespace

void errors::ExitIf(const bool condition, const tlx::vstring msg, const std::source_location &sloc) {
    if (condition) {
        std::cout << "[ERROR]" << std::endl;
        Log(std::cerr, LogLevel::Red) << "[" << sloc.file_name() << ":" << sloc.line() << "] " << msg << "\n";
        std::exit(CXM_ERR_EXIT);
    }
}

void errors::WarnIf(const bool condition, const tlx::vstring msg, const std::source_location &sloc) {
    if (condition) {
        std::cout << "[WARN]" << std::endl;
        Log(std::cout, LogLevel::Yellow) << "[" << sloc.file_name() << ":" << sloc.line() << "] " << msg << "\n";
    }
}

void errors::WrongDevice() {
    Log(std::cerr, LogLevel::Red) << "[DEVICE ERROR] Wrong device.\n";
    std::exit(CXM_ERR_EXIT);
}