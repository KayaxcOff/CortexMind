#ifndef CORTEXMIND_CORE_ENGINE_GRAPH_GRAD_OP_NAME_HPP
#define CORTEXMIND_CORE_ENGINE_GRAPH_GRAD_OP_NAME_HPP

namespace cortex::_fw::meta {
	// @brief Enumeration of gradient operation names
	enum class GradOpType {
		ADD,
		SUB,
		MUL,
		DIV,
		MATMUL,
		RELU,
		SIGMOID,
		TANH,
		SOFTMAX,
		LOG_SOFTMAX,
		CROSS_ENTROPY,
		MSE,
		MEAN,
		SUM,
		MAX,
		MIN,
		CONCAT,
		SPLIT,
		RESHAPE,
		TRANSPOSE,
		FLATTEN,
		UNK
	};
} // namespace cortex::_fw::meta

#endif // CORTEXMIND_CORE_ENGINE_GRAPH_GRAD_OP_NAME_HPP