//
// Created by muham on 26.05.2026.
//

#include "CortexMind/net/OptimizationFunction/adam.hpp"
#include <cmath>
#include <string>

#include <iostream>

using namespace cortex::_fw;
using namespace cortex::opt;
using namespace cortex;

Adam::Adam(const float32 lr, const float32 beta1, const float32 beta2, const float32 eps) : OptimizationBase("Adam(" + std::to_string(lr) + ")", lr) {
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->eps = eps;
    this->t = 0;
    this->flag = false;
}

Adam::~Adam() = default;
/*
void Adam::update() {
    */
    /*
    if (!this->flag) {
        this->Init();
    }

    this->t++;

    const float32 bc1 = 1.0f - std::pow(this->beta1, static_cast<float32>(this->t));
    const float32 bc2 = 1.0f - std::pow(this->beta2, static_cast<float32>(this->t));
    const float32 alpha = this->m_lr * std::sqrt(bc2) / bc1;

    for (size_t i = 0; i < this->m_params.size(); ++i) {
        tensor& param = this->m_params[i].get();
        tensor& g = param.grad();

        this->m[i] = this->m[i] * this->beta1 + g * (1.0f - this->beta1);
        std::cout << "Before" << std::endl;
        std::cout << "v[" << i << "] min:" << this->v[i].min() << " max:" << this->v[i].max() << std::endl;
        std::cout << "v[" << i << "] mean:" << this->v[i].mean() << std::endl;
        std::cout << "v[" << i << "] variance:" << this->v[i].variance() << std::endl;
        this->v[i].ones();
        this->v[i] = this->v[i] * this->beta2 + (g * g) * (1.0f - this->beta2);
        std::cout << "After" << std::endl;
        std::cout << "v[" << i << "] min:" << this->v[i].min() << " max:" << this->v[i].max() << std::endl;
        std::cout << "v[" << i << "] mean:" << this->v[i].mean() << std::endl;
        std::cout << "v[" << i << "] variance:" << this->v[i].variance() << std::endl;


        //const tensor v_old = this->v[i].detach().clone();
        //const tensor new_v_data = v_old * beta2 + g * g * (1.0f - beta2);

        //this->v[i] = new_v_data.clamp(0.0f, std::numeric_limits<f32>::max());

        param -= this->m[i] * alpha / (this->v[i].sqrt() + this->eps);
    }
    */
    /*
        const tensor g_copy = this->m_params[i].get().grad().clone();  // tamamen bağımsız kopya
        const tensor g_sq = g_copy * g_copy;

        const tensor v_new = this->v[i].clone() * this->beta2
                           + g_sq * (1.0f - this->beta2);
        this->v[i].SetData(v_new.get());  // storage değiştirmeden sadece veri yaz

        const tensor m_new = this->m[i].clone() * this->beta1
                           + g_copy * (1.0f - this->beta1);
        this->m[i].SetData(m_new.get());

        const tensor step = m_new / (v_new.sqrt() + this->eps) * alpha;
        const tensor param_new = this->m_params[i].get().detach().clone() - step;
        this->m_params[i].get().SetData(param_new.get());
    }
    */
/*
    if (!this->flag) {
        this->Init();
    }

    this->t++;

    const float32 bc1 = 1.0f - std::pow(this->beta1, static_cast<float32>(this->t));
    const float32 bc2 = 1.0f - std::pow(this->beta2, static_cast<float32>(this->t));
    const float32 alpha = this->m_lr * std::sqrt(bc2) / bc1;

    for (size_t i = 0; i < this->m_params.size(); ++i) {
        tensor& param = this->m_params[i].get();
        const tensor g = param.grad().clone();

        const tensor m_new = this->m[i].clone() * this->beta1 + g * (1.0f - this->beta1);
        const tensor v_new = this->v[i].clone() * this->beta2 + (g * g) * (1.0f - this->beta2);

        this->m[i].SetData(m_new.get());
        this->v[i].SetData(v_new.get());

        const tensor step = this->m[i] * alpha / (this->v[i].sqrt() + this->eps);

        param.SetData((param.detach().clone() - step).get());
    }

}
*/
/*
void Adam::update() {
    if (!this->flag) {
        this->Init();
    }

    this->t++;

    const float32 bc1 = 1.0f - std::pow(this->beta1, static_cast<float32>(this->t));
    const float32 bc2 = 1.0f - std::pow(this->beta2, static_cast<float32>(this->t));
    const float32 alpha = this->m_lr * std::sqrt(bc2) / bc1;

    for (size_t i = 0; i < this->m_params.size(); ++i) {
        tensor& param = this->m_params[i].get();
        const tensor g = param.grad().clone();

        auto mc = this->m[i];
        auto vc = this->v[i].clone();

        mc = mc.mul(this->beta1);
        mc = mc.add(g * (1.0f - this->beta1));

        vc = vc.mul(this->beta2);
        vc = vc.add((g * g) * (1.0f - this->beta2));

        int32 idx2 = 0;
        for(size_t j = 0; j < vc.len(); ++j) {
            if(vc.get()[j] < 0.0f) {
                idx2 += 1;
            }
        }
        std::cout << "Number of negative element of v: " << idx2 << std::endl;

        const tensor step = mc * alpha / (vc.sqrt() + this->eps);

        if (!std::isfinite(step.mean())) {
            std::cout << "ADAM STEP BROKEN at param " << i << std::endl;
        }

        param = param - step;
    }
}
*/
/*
void Adam::update() {
    if (!this->flag) {
        this->Init();
    }

    this->t++;

    const float32 bc1 = 1.0f - std::pow(this->beta1, static_cast<float32>(this->t));
    const float32 bc2 = 1.0f - std::pow(this->beta2, static_cast<float32>(this->t));
    const float32 alpha = this->m_lr * std::sqrt(bc2) / bc1;

    for (size_t i = 0; i < this->m_params.size(); ++i) {
        tensor& param = this->m_params[i].get();
        const tensor g = param.grad().clone();

        // ✅ RAW POINTER ACCESS - Tensor operators'ı bypass et
        float32* m_ptr = this->m[i].get();
        float32* v_ptr = this->v[i].get();
        const float32* g_ptr = g.get();
        float32* param_ptr = param.get();

        size_t len = this->m[i].len();

        // Element-wise update
        for (size_t j = 0; j < len; ++j) {
            float32 g_val = g_ptr[j];
            float32 g_sq = g_val * g_val;

            // m[i][j] = m[i][j] * beta1 + g[j] * (1 - beta1)
            m_ptr[j] = m_ptr[j] * this->beta1 + g_val * (1.0f - this->beta1);

            // v[i][j] = v[i][j] * beta2 + g²[j] * (1 - beta2)
            v_ptr[j] = v_ptr[j] * this->beta2 + g_sq * (1.0f - this->beta2);

            if (g_sq < 0) {
                std::cerr << "WARNING: g_sq < 0" << std::endl;
            }

            // Check negative v
            if (v_ptr[j] < 0.0f) {
                std::cout << "NEGATIVE V DETECTED at [" << i << "][" << j << "]: "
                          << v_ptr[j] << std::endl;
            }
        }

        // Parameter güncelleme - yine raw access
        for (size_t j = 0; j < len; ++j) {
            float32 step = m_ptr[j] * alpha / (std::sqrt(v_ptr[j]) + this->eps);

            if (!std::isfinite(step)) {
                std::cout << "ADAM STEP BROKEN at param " << i << " element " << j << std::endl;
            }

            param_ptr[j] -= step;
        }
    }
}
*/

void Adam::update() {
    if (!this->flag) {
        this->Init();
    }

    this->t++;

    const float32 bc1 = 1.0f - std::pow(this->beta1, static_cast<float32>(this->t));
    const float32 bc2 = 1.0f - std::pow(this->beta2, static_cast<float32>(this->t));
    const float32 alpha = this->m_lr * std::sqrt(bc2) / bc1;

    for (size_t i = 0; i < this->m_params.size(); ++i) {
        tensor& param = this->m_params[i].get();

        // Dikkat: grad() fonksiyonunun ömrünün döngü boyunca sürdüğünden emin olmalıyız.
        // Eğer grad() geçici bir nesne dönüyorsa clone() şarttır.
        tensor g = param.grad().clone();

        float32* m_ptr = this->m[i].get();
        float32* v_ptr = this->v[i].get();
        const float32* g_ptr = g.get();
        float32* param_ptr = param.get();

        size_t len = this->m[i].len();

        for (size_t j = 0; j < len; ++j) {
            float32 g_val = g_ptr[j];

            // Eğer gradyan NaN veya Inf ise ağ patlamıştır, bunu loglayalım
            if (!std::isfinite(g_val)) {
                // Küçük bir koruma: NaN gradyanı sıfıra çekelim ki bellek tamamen çökmesin
                g_val = 0.0f;
            }

            float32 g_sq = g_val * g_val;

            // Adam Formülleri
            m_ptr[j] = m_ptr[j] * this->beta1 + g_val * (1.0f - this->beta1);
            v_ptr[j] = v_ptr[j] * this->beta2 + g_sq * (1.0f - this->beta2);

            // Bellek sızıntısı/bozulması koruması (Zorunlu önlem)
            if (v_ptr[j] < 0.0f) {
                v_ptr[j] = 0.0f;
            }

            // Parametre Güncelleme
            float32 step = m_ptr[j] * alpha / (std::sqrt(v_ptr[j]) + this->eps);

            if (std::isfinite(step)) {
                param_ptr[j] -= step;
            }
        }
    }
}

void Adam::Init() {
    for (auto& item : this->m_params) {
        const tensor& x = item.get();
        this->m.emplace_back(x.shape(), x.device());
        this->m.back().zero();
        this->v.emplace_back(x.shape(), x.device());
        this->v.back().zero();
    }
    this->flag = true;
}