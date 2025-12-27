#ifndef CORTEXMIND_CORE_ENGINE_GRAPH_GRAD_NODE_HPP
#define CORTEXMIND_CORE_ENGINE_GRAPH_GRAD_NODE_HPP

#include <CortexMind/core/Engine/Graph/Node/grad_op_name.hpp>
#include <CortexMind/core/Engine/Tensor/tensor.hpp>

namespace cortex::_fw::meta {
	// / @brief Base class for gradient nodes in the computational graph.
	struct GradNode {
		~GradNode() = default;

		virtual void accumulate_grad(MindTensor& out_grad) = 0;
		virtual const GradOpType name() const noexcept = 0;
	};
} // namespace cortex::_fw::meta

#endif // CORTEXMIND_CORE_ENGINE_GRAPH_GRAD_NODE_HPP