//
// Created by muham on 26.05.2026.
//

#include "CortexMind/net/OptimizationFunction/adam.hpp"
#include <cmath>
#include <string>

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

void Adam::update() {
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

        const tensor step = this->m[i].clone() * alpha / (this->v[i].clone().sqrt() + this->eps);
        param.SetData((param.detach().clone() - step).get());
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
