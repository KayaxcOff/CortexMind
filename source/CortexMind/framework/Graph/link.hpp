//
// Created by muham on 16.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_GRAPH_LINK_HPP
#define CORTEXMIND_FRAMEWORK_GRAPH_LINK_HPP

#include <CortexMind/framework/Graph/flow.hpp>
#include <memory>

namespace cortex::_fw::meta {
    class GradientLink {
    public:
        GradientLink(Tensor& t, const std::shared_ptr<GradientFlow> &flow);
        GradientLink(const GradientLink&) = delete;
        GradientLink(GradientLink&&) = delete;
        ~GradientLink();

        GradientLink& operator=(const GradientLink&) = delete;
        GradientLink& operator=(GradientLink&&) = delete;
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRAPH_LINK_HPP