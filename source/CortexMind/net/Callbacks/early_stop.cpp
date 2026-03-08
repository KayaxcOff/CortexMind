//
// Created by muham on 5.03.2026.
//

#include "CortexMind/net/Callbacks/early_stop.hpp"
#include <string>
#include <limits>

using namespace cortex::call;
using namespace cortex::_fw;

EarlyStopping::EarlyStopping(const int64 _patience, const float32 _min_delta) : Callback("EarlyStopping(" + std::to_string(_patience) + ")"), patience(_patience), min_delta(_min_delta), best_loss(std::numeric_limits<float32>::max()), counter(0), stop(false) {}

EarlyStopping::~EarlyStopping() = default;

void EarlyStopping::on_batch_end(int64_t batch, float loss) {

}

void EarlyStopping::on_batch_begin(int64_t batch) {

}

void EarlyStopping::on_epoch_begin(int64_t epoch) {

}

void EarlyStopping::on_train_end() {

}

void EarlyStopping::on_epoch_end(int64_t epoch, const float32 loss, float32 acc) {
    if (loss < this->best_loss - this->min_delta) {
        this->best_loss = loss;
        this->counter = 0;
    } else {
        ++this->counter;
        if (this->counter > this->patience) {
            this->stop = true;
        }
    }
}

void EarlyStopping::on_train_begin() {
    this->best_loss = std::numeric_limits<float32>::max();
    this->counter = 0;
    this->stop = false;
}

bool EarlyStopping::shouldStop() {
    return this->stop;
}
