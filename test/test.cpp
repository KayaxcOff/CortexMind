//
// Created by muham on 19.12.2025.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;

int main() {

    tensor input(2, 2, 2, 2);
    input.fill(240);

    const tensor output = ds::TensorScale::scale(input);
    output.print();

    return 0;
}

/*
--- output ---
Tensor shape: [2, 2, 2, 2]
Batch 0:
 Channel 0:
 [ 0.941176 0.941176 ]
 [ 0.941176 0.941176 ]

 Channel 1:
 [ 0.941176 0.941176 ]
 [ 0.941176 0.941176 ]

Batch 1:
 Channel 0:
 [ 0.941176 0.941176 ]
 [ 0.941176 0.941176 ]

 Channel 1:
 [ 0.941176 0.941176 ]
 [ 0.941176 0.941176 ]
 */