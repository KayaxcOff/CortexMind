//
// Created by muham on 31.05.2026.
//

#include "CortexMind/net/Metrics/acc.hpp"

using namespace cortex::_fw;
using namespace cortex::metric;
using namespace cortex;

Accuracy::Accuracy() : MetricBase("Accuracy") {}

Accuracy::~Accuracy() = default;

float32 Accuracy::forward(const tensor &predict, const tensor &target) {
    const size_t batch = predict.shape()[0];
    const size_t class_count = predict.shape()[1]; // Sınıf sayısını alıyoruz (Örn: 6)

    const float32* p = predict.get();
    const float32* t = target.get();

    size_t correct = 0;

    for (size_t i = 0; i < batch; ++i) {
        size_t row_offset = i * class_count;

        // 1. Modelin tahmini için Argmax bul (En yüksek değerli sınıfın indisi)
        size_t pred_argmax = 0;
        float32 max_pred_val = p[row_offset];

        for (size_t c = 1; c < class_count; ++c) {
            if (p[row_offset + c] > max_pred_val) {
                max_pred_val = p[row_offset + c];
                pred_argmax = c;
            }
        }

        // 2. Gerçek hedef (Target) için Argmax bul (One-hot vektöründeki 1'in indisi)
        size_t true_argmax = 0;
        float32 max_true_val = t[row_offset];

        for (size_t c = 1; c < class_count; ++c) {
            if (t[row_offset + c] > max_true_val) {
                max_true_val = t[row_offset + c];
                true_argmax = c;
            }
        }

        // 3. İki indeks eşleşiyorsa tahmin doğrudur
        if (pred_argmax == true_argmax) {
            ++correct;
        }
    }

    // Doğru tahmin sayısını toplam batch sayısına bölerek gerçek metriği döndür
    return static_cast<float32>(correct) / static_cast<float32>(batch);
}