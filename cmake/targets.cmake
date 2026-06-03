add_library(CortexMind STATIC
        source/CortexMind/dataset/ProcessedDataset/circle.cpp
        source/CortexMind/dataset/ProcessedDataset/spiral.cpp
        source/CortexMind/dataset/ProcessedDataset/two_moons.cpp
        source/CortexMind/framework/Engine/AVX2/activation.cpp
        source/CortexMind/framework/Engine/AVX2/broadcast.cpp
        source/CortexMind/framework/Engine/AVX2/matrix.cpp
        source/CortexMind/framework/Engine/AVX2/partial.cpp
        source/CortexMind/framework/Engine/AVX2/reduce.cpp
        source/CortexMind/framework/Engine/AVX2/scalar.cpp
        source/CortexMind/framework/Engine/AVX2/wise.cpp
        source/CortexMind/framework/Engine/IX/activation.cpp
        source/CortexMind/framework/Engine/IX/compare.cpp
        source/CortexMind/framework/Engine/IX/convolution.cpp
        source/CortexMind/framework/Engine/IX/element_wise.cpp
        source/CortexMind/framework/Engine/IX/matrix.cpp
        source/CortexMind/framework/Engine/IX/random.cpp
        source/CortexMind/framework/Engine/IX/reduce.cpp
        source/CortexMind/framework/Engine/IX/scalar.cpp
        source/CortexMind/framework/Gradient/flow.cpp
        source/CortexMind/framework/Gradient/operations.cpp
        source/CortexMind/framework/Memory/mem.cpp
        source/CortexMind/framework/Net/layer.cpp
        source/CortexMind/framework/Net/loss.cpp
        source/CortexMind/framework/Net/metric.cpp
        source/CortexMind/framework/Net/optimization.cpp
        source/CortexMind/framework/Storage/operators.cpp
        source/CortexMind/framework/Storage/stor.cpp
        source/CortexMind/framework/Tensor/operators.cpp
        source/CortexMind/framework/Tensor/tensor.cpp
        source/CortexMind/framework/Tools/alignment.cpp
        source/CortexMind/framework/Tools/as_string.cpp
        source/CortexMind/framework/Tools/err.cpp
        source/CortexMind/framework/Tools/logger.cpp
        source/CortexMind/framework/Tools/series.cpp
        source/CortexMind/framework/Tools/tensor_debug.cpp
        source/CortexMind/framework/Tools/tensor_meta.cpp
        source/CortexMind/net/LossFunction/bce.cpp
        source/CortexMind/net/LossFunction/cce.cpp
        source/CortexMind/net/LossFunction/cce_with_logit.cpp
        source/CortexMind/net/LossFunction/mae.cpp
        source/CortexMind/net/LossFunction/mse.cpp
        source/CortexMind/net/Metrics/acc.cpp
        source/CortexMind/net/Metrics/approx.cpp
        source/CortexMind/net/Metrics/mse.cpp
        source/CortexMind/net/Metrics/rmse.cpp
        source/CortexMind/net/Model/model.cpp
        source/CortexMind/net/NeuralNetwork/batch_norm.cpp
        source/CortexMind/net/NeuralNetwork/convolution_2d.cpp
        source/CortexMind/net/NeuralNetwork/dense.cpp
        source/CortexMind/net/NeuralNetwork/dropout.cpp
        source/CortexMind/net/NeuralNetwork/flatten.cpp
        source/CortexMind/net/NeuralNetwork/gelu.cpp
        source/CortexMind/net/NeuralNetwork/gelu_exact.cpp
        source/CortexMind/net/NeuralNetwork/global_avg_pool_2d.cpp
        source/CortexMind/net/NeuralNetwork/input.cpp
        source/CortexMind/net/NeuralNetwork/leaky_relu.cpp
        source/CortexMind/net/NeuralNetwork/relu.cpp
        source/CortexMind/net/NeuralNetwork/sigmoid.cpp
        source/CortexMind/net/NeuralNetwork/sigmoid_fast.cpp
        source/CortexMind/net/NeuralNetwork/silu.cpp
        source/CortexMind/net/NeuralNetwork/silu_fast.cpp
        source/CortexMind/net/NeuralNetwork/softmax.cpp
        source/CortexMind/net/NeuralNetwork/tanh.cpp
        source/CortexMind/net/OptimizationFunction/adam.cpp
        source/CortexMind/net/OptimizationFunction/momentum.cpp
        source/CortexMind/net/OptimizationFunction/sgd.cpp
        source/CortexMind/tools/load.cpp
        source/CortexMind/tools/math.cpp
        source/CortexMind/tools/tensor_meta.cpp
        source/CortexMind/tools/version.cpp
        source/CortexMind/utility/Cast/factory.cpp
        source/CortexMind/utility/DataFrame/frame.cpp
        source/CortexMind/utility/DataFrame/operators.cpp
        source/CortexMind/utility/Image/kernel.cpp
        source/CortexMind/utility/Image/vm.cpp
)

if(CXM_CUDA_AVAILABLE)
    target_sources(CortexMind PRIVATE
            source/CortexMind/framework/Engine/CUDA/activation.cu
            source/CortexMind/framework/Engine/CUDA/broadcast.cu
            source/CortexMind/framework/Engine/CUDA/compare.cu
            source/CortexMind/framework/Engine/CUDA/element_wise.cu
            source/CortexMind/framework/Engine/CUDA/matrix.cu
            source/CortexMind/framework/Engine/CUDA/reduce.cu
            source/CortexMind/framework/Engine/CUDA/scalar.cu
            source/CortexMind/framework/Engine/IX/fill.cu
            source/CortexMind/framework/Memory/forge.cu
            source/CortexMind/framework/Memory/transform.cu
            source/CortexMind/framework/Tools/cuda.cu
            source/CortexMind/runtime/curand.cu
            source/CortexMind/runtime/provider.cu
    )
endif()

target_include_directories(CortexMind PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/source
        ${stb_SOURCE_DIR}
)

target_link_libraries(CortexMind PUBLIC
        nlohmann_json::nlohmann_json
)

if(CXM_CUDA_AVAILABLE)
    target_include_directories(CortexMind PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(CortexMind PRIVATE
            CUDA::cudart
            CUDA::cublas
            CUDA::curand
    )
endif()

target_compile_definitions(CortexMind PUBLIC
        $<$<BOOL:${CXM_CUDA_AVAILABLE}>:CXM_IS_CUDA_AVAILABLE=1>
        $<$<BOOL:${CXM_CUDA_AVAILABLE}>:CXM_CUDA_ARCH=86>
)