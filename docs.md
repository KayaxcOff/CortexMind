# Documentation for Cortex Mind

===
VERSION 1.5.0 -
Written by Muhammet Kaya -
LICENSE: MIT License
===

# 1. Overview
------------------------------------------------------------
Cortex Mind is a C++ deep learning library designed to simulate
artificial neural networks efficiently. It provides modular
components such as layers, optimizers, loss functions, and
activation functions that can be easily composed to form
custom neural architectures.
# 2. Features
------------------------------------------------------------
- Modular design for easy customization
- Support for various layer types (Dense, Convolutional,
  Recurrent, etc.)
- Multiple activation functions (ReLU, Sigmoid, Tanh, etc.)
- Built-in optimizers (SGD, Adam, RMSprop, etc.)
- Loss functions (MSE, Cross-Entropy, etc.)
# 3. Installation
------------------------------------------------------------
To install Cortex Mind, clone the repository and build the
project using CMake:
```
git clone https://github.com/KayaxcOff/CortexMind.git
```
Inside the `build` folder, there are `cmake` and `lib` folders 
containing files with the `.cmake` and `.lib` extensions. You can 
use them like this:
```
cmake_minimum_required(VERSION 4.0)
project(YourProjectName)
set(CMAKE_CXX_STANDARD 20)
find_package(CortexMind REQUIRED PATHS "path/to/CortexMind/build/cmake")
add_executable(YourExecutableName main.cpp)
target_link_libraries(YourExecutableName CortexMind::CortexMind)
```
# 4. Basic Usage
------------------------------------------------------------
Here is a simple example of how to create and train a neural
network using Cortex Mind:
```cpp
#include <CortexMind/cortexmind.h>
using namespace cortex;

int main() {
    model::Model neuralNet;
    
    tensor x = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    tensor y = {{0.0}, {1.0}, {1.0}, {0.0}};
    tensor test = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    
    neuralNet.add<nn::Dense>(2, 3);
    neuralNet.add<nn::Dense>(3, 1);
    neuralNet.compile<loss::MeanSquared, optim::StochasticGradient, act::ReLU>(0.001);
    neuralNet.fit(x, y, 10000, 4);
    
    auto predictions = neuralNet.predict(test);
    predictions.print();
    
    return 0;
}
```
# 5. API Reference
------------------------------------------------------------
`cortex` is the main namespace containing all classes and functions. You can explore the following sub-namespaces:
- `cortex::nn`: Contains various neural network layer implementations.
- `cortex::act`: Contains activation function implementations.
- `cortex::loss`: Contains loss function implementations.
- `cortex::optim`: Contains optimizer implementations.
- `cortex::model`: Contains the Model class for building and training neural networks.
- `cortex::tensor`: Contains the tensor class for data representation and manipulation.
- `cortex::tools`: Contains utility functions for data preprocessing and other tasks.