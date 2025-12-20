//
// Created by muham on 20.12.2025.
//

#include "CortexMind/models/LinearRegression/linear.hpp"

using namespace cortex::tin;
using namespace cortex;

LinearRegression::LinearRegression(const int input_dim, const float lr) : weights(1, 1, 1, input_dim), bias(0), lr(lr) {}

LinearRegression::~LinearRegression() = default;

void LinearRegression::fit(std::vector<tensor> &X, std::vector<tensor> &Y) {
    constexpr int epochs = 100;

    for (int i = 0; i < epochs; ++i) {
        for (size_t j = 0; j < X.size(); ++j) {
            tensor& x = X[j];
            tensor& y = Y[j];

            tensor y_pred = this->predict(x);
            tensor grad = y_pred - y;

            for (int k = 0; k < this->weights.width(); ++k) {
                float sum = 0;
                for (int l = 0; l < this->weights.batch(); ++l) {
                    sum += grad.at(l, 0, 0, 0) * x.at(l, 0, 0, k);
                }
                this->weights.at(0, 0, 0, k) -= this->lr * sum / static_cast<float>(x.batch());
            }

            float bias_grad = 0.0f;
            for (int l = 0; l < x.batch(); ++l) {
                bias_grad += grad.at(l, 0, 0, 0);
            }
            this->bias -= this->lr * bias_grad / static_cast<float>(x.batch());
        }
    }
}

tensor LinearRegression::predict(const tensor &pred) {
    tensor y_pred(pred.batch(), 1, 1, 1);
    for (int i = 0; i < pred.batch(); ++i) {
        float sum = this->bias;
        for (int j = 0; j < pred.width(); ++j) {
            sum += pred.at(i, 0, 0, j) * this->weights.at(0, 0, 0, j);
        }
        y_pred.at(i, 0, 0, 0) = sum;
    }
    return y_pred;
}
