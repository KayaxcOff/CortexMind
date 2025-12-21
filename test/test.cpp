//
// Created by muham on 19.12.2025.
//

#include <CortexMind/cortexmind.hpp>
#include <vector>

using namespace cortex;

int main() {
    std::vector<tensor> array;

    for (int i = 0; i < 10; i++) {
        tensor x(1, 1, 2, 2);
        x.uniform_rand();
        array.push_back(x);
    }

    for (int i = 0; i < array.size(); i++) {
        array[i] = ds::TensorScale::scale(array[i]);
    }

    log("3. Index of Array: ");
    array[3].print();

    return 0;
}

/*
---- output ----
[LOG]: 3. Index of Array:
Tensor shape: [1, 1, 2, 2]
Batch 0:
 Channel 0:
 [ 0.00171939 0.00175495 ]
 [ 0.000173517 2.90936e-05 ]
*/