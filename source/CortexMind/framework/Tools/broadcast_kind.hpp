//
// Created by muham on 14.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_BROADCAST_KIND_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_BROADCAST_KIND_HPP

namespace cortex::_fw {
    /**
     * @brief Specifies the type of broadcasting required between two tensors.
     */
    enum class BroadcastKind {
        kNone,      ///< No broadcasting needed (shapes are equal)
        kRow,       ///< Row-wise broadcasting (broadcasting over rows)
        kCol,       ///< Column-wise broadcasting (broadcasting over columns)
        kGeneral    ///< General broadcasting (arbitrary shape broadcasting)
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_BROADCAST_KIND_HPP