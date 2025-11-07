# 🧠 CortexMind

**CortexMind** is a C++ library designed to emulate the core behavior of artificial neural networks.  
It currently provides a fully functional **Dense layer** implementation and includes modules like **Tokenizer** and **ImageGenerator**, enabling model creation with both text and image data.

---

## 🚀 Features

- Written in **modern C++20**
- Modular architecture for neural network components
- Working **Dense Layer** implementation
- **Tokenizer** for text-based data processing
- **ImageGenerator** for image-based dataset creation
- Easy-to-extend design for future layer and optimizer types

---

## 🧩 Example Usage

```cpp
#include <CortexMind/cortexmind.hpp>

using namespace cortex;
int main() {
  model::Model model;

  math::MindVector X = {{1.0, 0}, {0, 1.0}};
  math::MindVector Y = {{1.0}, {0}};

  constexpr size_t epochs = 2;

  model.add<layer::Dense>(1, 10);
  model.add<layer::Dense>(10, 1);
  model.compile<loss::MeanSquared, optim::StochasticGradient>(0.01);
  model.fit(X, Y, epochs);

  const auto result_test = model.predict(test);

  result_test.print("Result of Test");

  return 0;
}
```
