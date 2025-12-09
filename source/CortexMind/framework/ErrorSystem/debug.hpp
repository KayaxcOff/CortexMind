//
// Created by muham on 5.12.2025.
//

#ifndef CORTEXMIND_DEBUG_HPP
#define CORTEXMIND_DEBUG_HPP

#include <string>
#include <iostream>

namespace cortex::_fw {
    class SynapticNode {
    public:
        SynapticNode() = default;
        ~SynapticNode() = default;

        static void captureFault(const bool isValid, const std::string &name, const std::string &msg) {
            if (isValid) {
                std::cerr << "[FATAL ERROR] ]" << "\n" << name << "\n" << msg <<std::endl;
                std::exit(1);
            }
        }
    };
}

#endif //CORTEXMIND_DEBUG_HPP