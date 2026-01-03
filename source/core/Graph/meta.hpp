//
// Created by muham on 29.12.2025.
//

#ifndef CORTEXMIND_CORE_GRAPH_META_HPP
#define CORTEXMIND_CORE_GRAPH_META_HPP

#include <memory>
#include <vector>
#include <functional>

namespace cortex::_fw::meta {
    template <class T>
    struct AutoDiff {
        bool is_leaf;
        bool requires_grad;
        std::vector<std::shared_ptr<AutoDiff>> parents;
        std::function<void(T&)> backward;
    };
} // namespace cortex::_fw::meta

#endif // CORTEXMIND_CORE_GRAPH_META_HPP