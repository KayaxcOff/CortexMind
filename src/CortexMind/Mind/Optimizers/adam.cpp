//
// Created by muham on 28.11.2025.
//

#include "CortexMind/Mind/OptimizerFunc/adam.hpp"

#include <cmath>

using namespace cortex::optim;

Adam::Adam(const float _lr, const float64 _beta1, const float64 _beta2, const float64 _epsilon) : Optimizer(_lr), beta1(_beta1), beta2(_beta2), epsilon(_epsilon), m(0, 0), v(0, 0), t(0) {}

void Adam::step(tensor &weights, const tensor &gradients) {
    if (weights.get_rows() != gradients.get_rows() || weights.get_cols() != gradients.get_cols()) {
        throw std::invalid_argument("Weights and gradients must have the same shape.");
    }

    if (this->m.get_rows() == 0 && this->m.get_cols() == 0) {
        this->m = tensor(weights.get_rows(), weights.get_cols());
        this->v = tensor(weights.get_rows(), weights.get_cols());
    }

    ++this->t;

    const float64 lt_t = this->learning_rate * std::sqrt(1.0 - std::pow(this->beta2, this->t)) / (1.0 - std::pow(this->beta1, this->t));

    for (size i = 0; i < gradients.get_rows(); ++i) {
        for (size j = 0; j < gradients.get_cols(); ++j) {
            const float64 g = gradients(i, j);

            this->m(i, j) = this->beta1 * this->m(i, j) + (1.0 - this->beta1) * g;
            this->v(i, j) = this->beta2 * this->v(i, j) + (1.0 - this->beta2) * (g * g);

            const float64 m_hat = this->m(i, j) / (1.0 - std::pow(this->beta1, this->t));
            const float64 v_hat = this->v(i, j) / (1.0 - std::pow(this->beta2, this->t));

            weights(i, j) -= lt_t * m_hat / (std::sqrt(v_hat) + this->epsilon);
        }
    }
}