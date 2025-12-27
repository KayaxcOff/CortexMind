#ifndef CORTEXMIND_CORE_ENGINE_GRAPH_OPS_ADD_HPP
#define CORTEXMIND_CORE_ENGINE_GRAPH_OPS_ADD_HPP

#include <CortexMind/core/Engine/Graph/Node/grad_node.hpp>

namespace cortex::_fw::meta {
	struct AddOp : public GradNode {
		MindTensor tena, tenb;

		AddOp(const MindTensor& a, const MindTensor& b) : tena(a), tenb(b) {}

		void accumulate_grad(MindTensor& out_grad) override;
		const GradOpType name() const noexcept override {
			return GradOpType::ADD;
		}
	};
} // namespace cortex::_fw::meta

#endif // CORTEXMIND_CORE_ENGINE_GRAPH_OPS_ADD_HPP