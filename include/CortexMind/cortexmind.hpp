//
// Created by muham on 4.11.2025.
//

#ifndef CORTEXMIND_CORTEXMIND_HPP
#define CORTEXMIND_CORTEXMIND_HPP
/*---- CortexMind Includes ----*/
// model
#include "Model/Model/model.hpp"
// layers
#include "Model/Layers/layer.hpp"
#include "Model/Layers/Dense/dense.hpp"
#include "Model/Layers/Norm/norm.hpp"
#include "Model/Layers/Retina/retina.hpp"
// optimizer function class
#include "Model/Optimizer/optimizer.hpp"
#include "Model/Optimizer/SGD/sgd.hpp"
// loss function classes
#include "Model/Loss/loss.hpp"
#include "Model/Loss/MSE/mse.hpp"
#include "Model/Loss/MAE/mae.hpp"
// helpful tools classes and function
#include "Utils/ImageData/image_data.hpp"
#include "Utils/TextData/tokenizer.hpp"
#include "Utils/MathTools/random_weight.hpp"
#include "Utils/MathTools/vector/vector.hpp"

#ifdef CORTEXMIND_USE_NAMESPACE_SHORTCUTS
namespace c = cortex;
namespace cm = cortex::model;
namespace cl = cortex::layer;
namespace co = cortex::optim;
namespace ct = cortex::tools;
namespace cs = cortex::loss;
#endif

#endif //CORTEXMIND_CORTEXMIND_HPP