#ifndef CORTEXMIND_CORE_NET_LAYER_HPP
#define CORTEXMIND_CORE_NET_LAYER_HPP

#include <CortexMind/core/Params/params.hpp>

namespace cortex::_fw {
	// @brief Base class for all neural network layers.
	class Layer {
	public:
		Layer() = default;
		virtual ~Layer() = default;

		// @brief Forward pass through the
		// @param input The input data to the layer.
		virtual tensor forward(const tensor& input) = 0;
	protected:
		tensor weights; ///< Weights of the layer
	};
} // namespace cortex::_fw

#endif // CORTEXMIND_CORE_NET_LAYER_HPP