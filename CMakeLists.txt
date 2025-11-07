cmake_minimum_required(VERSION 4.0)
project(CortexMind VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(STB_DIR "External/STB/stb")
include_directories("${STB_DIR}\\include")

add_library(CortexMind STATIC
        src/CortexMind/Model/Model/model.cpp
        src/CortexMind/Model/Layers/layer.cpp
        src/CortexMind/Model/Layers/Retina/retina.cpp
        src/CortexMind/Model/Layers/Norm/norm.cpp
        src/CortexMind/Model/Layers/Dense/dense.cpp
        src/CortexMind/Model/Loss/MAE/mae.cpp
        src/CortexMind/Model/Loss/MSE/mse.cpp
        src/CortexMind/Model/Optimizer/optimizer.cpp
        src/CortexMind/Model/Optimizer/SGD/sgd.cpp
        src/CortexMind/Utils/ImageData/stb_images.cpp
        src/CortexMind/Utils/ImageData/image_data.cpp
        src/CortexMind/Utils/TextData/tokenizer.cpp
        src/CortexMind/Utils/MathTools/vector/vector.cpp
)

target_include_directories(CortexMind PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

add_executable(MindTest test/main.cpp)
target_link_libraries(MindTest PRIVATE CortexMind)
target_include_directories(MindTest PRIVATE ${CMAKE_SOURCE_DIR}/include)