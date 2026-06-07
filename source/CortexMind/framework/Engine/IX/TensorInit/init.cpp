//
// Created by muham on 5.06.2026.
//

#include "CortexMind/framework/Engine/IX/TensorInit/init.hpp"
#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <CortexMind/framework/Tools/err.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Engine/CUDA/scalar.cuh>
    #include <CortexMind/runtime/curand.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <random>

using namespace cortex::_fw::ix;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

void TensorInit::rand(TensorStorage *x) {
    CXM_ASSERT(x == nullptr, "Storage is null");
    CXM_ASSERT(!x->isValid(), "Storage is invalid");
    CXM_ASSERT(x->isEmpty(), "Storage is empty");

    const size_t num = x->size();

    #if CXM_IS_CUDA_AVAILABLE
        if (x->device() == DeviceType::kHOST) {
            std::mt19937 generator(static_cast<unsigned int>(0.1f));
            std::uniform_real_distribution uniform(0.000001f, 1.0f);

            i64 i = 0;
            const i64 safe_limit = static_cast<i64>(num - (num % 16));

            const avx2::vec8f vec_minus_two = avx2::set1(-2.0f);
            const avx2::vec8f vec_two_pi    = avx2::set1(2.0f * 3.1415926535f);

            for (; i < safe_limit; i += 16) {
                alignas(32) f32 u1[8];
                alignas(32) f32 u2[8];

                for (i32 j = 0; j < 8; ++j) {
                    u1[j] = uniform(generator);
                    u2[j] = uniform(generator);
                }

                const avx2::vec8f v1 = avx2::load(u1);
                const avx2::vec8f v2 = avx2::load(u2);

                const avx2::vec8f v3 = avx2::log(v1);
                const avx2::vec8f v4 = avx2::mul(vec_minus_two, v3);
                const avx2::vec8f v5   = avx2::sqrt(v4);

                const avx2::vec8f v_theta = avx2::mul(vec_two_pi, v2);

                const avx2::vec8f v_cos = avx2::cos(v_theta);
                const avx2::vec8f v_sin = avx2::sin(v_theta);

                const avx2::vec8f res_cos = avx2::mul(v5, v_cos);
                const avx2::vec8f res_sin = avx2::mul(v5, v_sin);

                avx2::store(&x->data()[i], res_cos);
                avx2::storeu(&x->data()[i + 8], res_sin);
            }

            if (i < num) {
                std::normal_distribution normal_dist(0.0f, 1.0f);
                for (; i < num; ++i) {
                    x->data()[i] = normal_dist(generator);
                }
            }
        } else {
            curandGenerator_t gen = runtime::Curand::instance().rand_handle;
            const curandStatus_t status = curandGenerateNormal(gen, x->data(), num, 0.0f, 1.0f);
            CXM_ASSERT(status != CURAND_STATUS_SUCCESS, "curandGenerateNormal() GPU'da başarısız");
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        std::mt19937 generator(static_cast<unsigned int>(0.1f));
        std::uniform_real_distribution uniform(0.000001f, 1.0f);

        i64 i = 0;
        const i64 safe_limit = static_cast<i64>(num - (num % 16));

        const avx2::vec8f vec_minus_two = avx2::set1(-2.0f);
        const avx2::vec8f vec_two_pi    = avx2::set1(2.0f * 3.1415926535f);

        for (; i < safe_limit; i += 16) {
            alignas(32) f32 u1[8];
            alignas(32) f32 u2[8];

            for (i32 j = 0; j < 8; ++j) {
                u1[j] = uniform(generator);
                u2[j] = uniform(generator);
            }

            const avx2::vec8f v1 = avx2::load(u1);
            const avx2::vec8f v2 = avx2::load(u2);

            const avx2::vec8f v3 = avx2::log(v1);
            const avx2::vec8f v4 = avx2::mul(vec_minus_two, v3);
            const avx2::vec8f v5   = avx2::sqrt(v4);

            const avx2::vec8f v_theta = avx2::mul(vec_two_pi, v2);

            const avx2::vec8f v_cos = avx2::cos(v_theta);
            const avx2::vec8f v_sin = avx2::sin(v_theta);

            const avx2::vec8f res_cos = avx2::mul(v5, v_cos);
            const avx2::vec8f res_sin = avx2::mul(v5, v_sin);

            avx2::store(&x->data()[i], res_cos);
            avx2::storeu(&x->data()[i + 8], res_sin);
        }

        if (i < num) {
            std::normal_distribution normal_dist(0.0f, 1.0f);
            for (; i < num; ++i) {
                x->data()[i] = normal_dist(generator);
            }
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorInit::uniform(TensorStorage *x, const f32 min, const f32 max) {
    CXM_ASSERT(x == nullptr, "Storage is null");
    CXM_ASSERT(!x->isValid(), "Storage is invalid");

    const size_t num = x->size();

    #if CXM_IS_CUDA_AVAILABLE
        if (x->device() == DeviceType::kHOST) {
            thread_local std::mt19937 rng{std::random_device{}()};
            std::uniform_real_distribution dist(min, max);
            for (size_t i = 0; i < num; ++i) {
                x->data()[i] = dist(rng);
            }
        } else {
            CXM_ASSERT(
                curandGenerateUniform(runtime::Curand::instance().rand_handle, x->data(), num)
                != CURAND_STATUS_SUCCESS,
                "curandGenerateUniform() başarısız"
            );

            if (const f32 range = max - min; range != 1.0f) {
                cuda::ScalarKernel::mul(x->data(), range, num);
            }
            if (min != 0.0f) {
                cuda::ScalarKernel::add(x->data(), min, num);
            }
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        thread_local std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution dist(min, max);
        for (size_t i = 0; i < num; ++i) {
            x->data()[i] = dist(rng);
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

void TensorInit::fill(TensorStorage *x, const f32 value) {
    CXM_ASSERT(x == nullptr, "Storage is null");
    CXM_ASSERT(!x->isValid(), "Storage is invalid");

    const size_t num = x->size();

    #if CXM_IS_CUDA_AVAILABLE
        if (x->device() == DeviceType::kHOST) {
            const auto val = avx2::set1(value);
            size_t i = 0;
            for (; i + 8 <= num; i += 8) {
                avx2::storeu(x->data() + i, val);
            }
            for (; i < num; ++i) {
                x->data()[i] = value;
            }
        } else {
            //cuda::ScalarKernel::fill(x->data(), value, num);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        const auto val = avx2::set1(value);
        size_t i = 0;
        for (; i + 8 <= num; i += 8) {
            avx2::storeu(x->data() + i, val);
        }
        for (; i < num; ++i) {
            x->data()[i] = value;
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}