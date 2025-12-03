//
// Created by muham on 3.12.2025.
//

#include "CortexMind/net/OptimizerFunc/Adam/adam.hpp"

using namespace cortex::optim;
using namespace cortex;

Adam::Adam(const double learning_rate, const double b1, const double b2, const double epsilon) : Optimizer(learning_rate), beta1(b1), beta2(b2), epsilon(epsilon), t(0) {}

void Adam::step() {
    this->t++;

    if (this->v_moments.empty()) {
        for (const auto& [_weights, _grads] : this->params_list) {
            const auto& shape = _weights->get_shape();
            size_t d0 = !shape.empty() ? shape[0] : 1;
            size_t d1 = shape.size() > 1 ? shape[1] : 1;
            size_t d2 = shape.size() > 2 ? shape[2] : 1;

            this->v_moments.emplace_back(d0, d1, d2);
            this->s_moments.emplace_back(d0, d1, d2);
        }
    }

    const double alpha = this->learning_rate;
    const double bias_correction_1 = 1.0 / (1.0 - std::pow(this->beta1, this->t));
    const double bias_correction_2 = 1.0 / (1.0 - std::pow(this->beta2, this->t));

    for (size_t i = 0; i < this->params_list.size(); ++i) {
        tensor* W = this->params_list[i]._weights;
        tensor* G = this->params_list[i]._grads;
        tensor& V = v_moments[i];
        tensor& S = s_moments[i];
        const size_t total_size = W->get_data().size();
        for (size_t j = 0; j < total_size; ++j) {
            const double g = G->get_data()[j];
            double v = V.get_data()[j];
            double s = S.get_data()[j];

            v = this->beta1 * v + (1.0 - this->beta1) * g;

            s = this->beta2 * s + (1.0 - this->beta2) * (g * g);

            const double v_hat = v * bias_correction_1;
            const double s_hat = s * bias_correction_2;

            W->get_data()[j] -= alpha * (v_hat / (std::sqrt(s_hat) + this->epsilon));

            V.get_data()[j] = v;
            S.get_data()[j] = s;
        }
    }
}
