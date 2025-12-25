# CortexMind

`CortexMind` is a library project aiming to be a mini tensor operations and machine learning library in the C++ programming language.

`CortexMind` uses a mathematical engine composed of `AVX2` functions for speed in tensor operations, and its future vision includes using CUDA as well.

`CortexMind`'s API is modeled after `Tensorflow`'s 'Python' API and it can be integrated directly into `CMake` with a "`.lib`" file.

## API

>> `cortex`
Main namespace, the tensor variable, which is a CortexMind-specific variable, can be accessed via the `cortex` namespace.
>> `cortex::nn`
Layer classes are here
>> `cortex::net`
Model class, optimizer function, activation function and loss function classes are here
>> `cortex::ds`
Scale function are here
>> `cortex::tin`
Models are here
>> `cortex::tools`
Helpful tools classes are here
>> `cortex::_fw`
Core of library
