#ifndef CORTEXMIND_CORE_ENGINE_GRAPH_NODE_HPP
#define CORTEXMIND_CORE_ENGINE_GRAPH_NODE_HPP

#include <CortexMind/core/Engine/Graph/Node/grad_op_name.hpp>
#include <CortexMind/core/Engine/Graph/Node/grad_node.hpp>

#include <functional>
#include <vector>
#include <memory>
#include <string>

namespace cortex::_fw {
	// Forward declaration
	class MindTensor;

	namespace meta {
		// @brief Metadata for autograd functionality
		struct AutogradMeta {
			bool requires_grad = false; /// Flag indicating if gradient is required
			bool is_leaf = true;        /// Flag indicating if tensor is a leaf node
			std::shared_ptr<GradNode> grad_node = nullptr; /// Associated gradient node
			std::shared_ptr<MindTensor> grad = nullptr; /// Gradient tensor
			std::vector<std::shared_ptr<MindTensor>> parents; /// Parent nodes in the computation graph
			std::function<void(const MindTensor&)> backward_fn; /// Backward function for gradient computation
			GradOpType _op_type = GradOpType::UNK; /// Type of gradient operation

			AutogradMeta(const AutogradMeta&) = delete;
			AutogradMeta& operator=(const AutogradMeta&) = delete;
			AutogradMeta(AutogradMeta&&) = default;
			AutogradMeta& operator=(AutogradMeta&&) = default;
		};
	} // namespace cortex::_fw::meta
} // namespace cortex::_fw

#endif // CORTEXMIND_CORE_ENGINE_GRAPH_NODE_HPP