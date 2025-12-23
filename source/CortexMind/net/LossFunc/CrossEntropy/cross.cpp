//
// Created by muham on 23.12.2025.
//

#include "CortexMind/net/LossFunc/CrossEntropy/cross.hpp"
#include <CortexMind/framework/Core/AVX/funcs.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace cortex::net;
using namespace cortex::_fw;
using namespace cortex;

CrossEntropy::CrossEntropy() = default;
CrossEntropy::~CrossEntropy() = default;

tensor CrossEntropy::forward(const tensor &predictions, const tensor &targets) const {
    const int batch = predictions.batch();
    const int num_classes = predictions.width();

    tensor output(batch, 1, 1, 1);

    for (int b = 0; b < batch; ++b) {
        float loss = 0.0f;

        if (targets.width() == 1) {
            const int target_class = static_cast<int>(targets.at(b, 0, 0, 0));

            if (target_class < 0 || target_class >= num_classes) {
                std::cerr << "Target class " << target_class << " out of range [0, "
                          << num_classes << ")" << std::endl;
                continue;
            }

            const float pred = std::max(predictions.at(b, 0, 0, target_class), 1e-7f);
            loss = -std::log(pred);

        } else {
            avx2::reg loss_vec = avx2::zero();

            int c = 0;
            while (c + 8 <= num_classes) {
                float pred_vals[8], target_vals[8];
                for (int i = 0; i < 8; ++i) {
                    pred_vals[i] = std::max(predictions.at(b, 0, 0, c + i), 1e-7f);
                    target_vals[i] = targets.at(b, 0, 0, c + i);
                }

                const avx2::reg pred_vec = avx2::load(pred_vals);
                const avx2::reg target_vec = avx2::load(target_vals);

                const avx2::reg log_pred = avx2::log_approx(pred_vec);
                const avx2::reg prod = avx2::mul(target_vec, log_pred);

                loss_vec = avx2::add(loss_vec, prod);
                c += 8;
            }

            loss -= avx2::h_sum(loss_vec);

            while (c < num_classes) {
                const float pred = std::max(predictions.at(b, 0, 0, c), 1e-7f);
                const float target = targets.at(b, 0, 0, c);
                loss -= target * std::log(pred);
                c++;
            }
        }

        output.at(b, 0, 0, 0) = loss;
    }

    return output;
}

tensor CrossEntropy::backward(const tensor &predictions, const tensor &targets) const {
    const int batch = predictions.batch();
    const int num_classes = predictions.width();

    tensor grad(batch, 1, 1, num_classes);

    for (int b = 0; b < batch; ++b) {
        if (targets.width() == 1) {
            const int target_class = static_cast<int>(targets.at(b, 0, 0, 0));

            if (target_class < 0 || target_class >= num_classes) {
                continue;
            }
            for (int c = 0; c < num_classes; ++c) {
                if (c == target_class) {
                    grad.at(b, 0, 0, c) = predictions.at(b, 0, 0, c) - 1.0f;
                } else {
                    grad.at(b, 0, 0, c) = predictions.at(b, 0, 0, c);
                }
            }

        } else {
            int c = 0;
            while (c + 8 <= num_classes) {
                float pred_vals[8], target_vals[8];
                for (int i = 0; i < 8; ++i) {
                    pred_vals[i] = predictions.at(b, 0, 0, c + i);
                    target_vals[i] = targets.at(b, 0, 0, c + i);
                }

                const avx2::reg pred_vec = avx2::load(pred_vals);
                const avx2::reg target_vec = avx2::load(target_vals);
                const avx2::reg grad_vec = avx2::sub(pred_vec, target_vec);

                float grad_vals[8];
                avx2::store(grad_vals, grad_vec);
                for (int i = 0; i < 8; ++i) {
                    grad.at(b, 0, 0, c + i) = grad_vals[i];
                }

                c += 8;
            }

            while (c < num_classes) {
                grad.at(b, 0, 0, c) = predictions.at(b, 0, 0, c) - targets.at(b, 0, 0, c);
                c++;
            }
        }
    }

    return grad;
}