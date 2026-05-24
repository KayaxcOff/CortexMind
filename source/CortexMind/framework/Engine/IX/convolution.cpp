//
// Created by muham on 24.05.2026.
//

#include "CortexMind/framework/Engine/IX/convolution.hpp"
#include <CortexMind/framework/Tools/err.hpp>
#include <cstring>

using namespace cortex::_fw::ix;
using namespace cortex::_fw::sys;
using namespace cortex::_fw;

namespace {
    void unfold_cpu(const f32* input, f32* col, const i64 N, const i64 C, const i64 H, const i64 W, const i64 kH, const i64 kW, const i64 sH, const i64 sW, const i64 pH, const i64 pW, const i64 oH, const i64 oW) {
        const i64 col_cols = N * oH * oW;

        for (i64 c = 0; c < C; ++c) {
            for (i64 kh = 0; kh < kH; ++kh) {
                for (i64 kw = 0; kw < kW; ++kw) {
                    const i64 row = c * kH * kW + kh * kW + kw;
                    f32* col_row = col + row * col_cols;

                    for (i64 n = 0; n < N; ++n) {
                        const f32* input_n = input + n * C * H * W;

                        for (i64 oh = 0; oh < oH; ++oh) {
                            const i64 ih = oh * sH - pH + kh;

                            for (i64 ow = 0; ow < oW; ++ow) {
                                const i64 iw = ow * sW - pW + kw;
                                const i64 col_idx = n * oH * oW + oh * oW + ow;

                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    col_row[col_idx] = input_n[c * H * W + ih * W + iw];
                                } else {
                                    col_row[col_idx] = 0.0f;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void fold_cpu(const f32* col, f32* input_grad, const i64 N, const i64 C, const i64 H, const i64 W, const i64 kH, const i64 kW, const i64 sH, const i64 sW, const i64 pH, const i64 pW, const i64 oH, const i64 oW) {
        const i64 col_cols = N * oH * oW;

        std::memset(input_grad, 0, N * C * H * W * sizeof(f32));

        for (i64 c = 0; c < C; ++c) {
            for (i64 kh = 0; kh < kH; ++kh) {
                for (i64 kw = 0; kw < kW; ++kw) {
                    const i64 row = c * kH * kW + kh * kW + kw;
                    const f32* col_row = col + row * col_cols;

                    for (i64 n = 0; n < N; ++n) {
                        f32* grad_n = input_grad + n * C * H * W;

                        for (i64 oh = 0; oh < oH; ++oh) {
                            const i64 ih = oh * sH - pH + kh;

                            for (i64 ow = 0; ow < oW; ++ow) {
                                const i64 iw = ow * sW - pW + kw;
                                const i64 col_idx = n * oH * oW + oh * oW + ow;

                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    grad_n[c * H * W + ih * W + iw] += col_row[col_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
} //unnamed namespace

void Convolution::unfold(const f32 *input, f32 *col, const i64 N, const i64 C, const i64 H, const i64 W, const i64 kH, const i64 kW, const i64 sH, const i64 sW, const i64 pH, const i64 pW, const i64 oH, const i64 oW, const DeviceType device) {
    CXM_ASSERT(N == 0 || C == 0, "N and C must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (device == DeviceType::kCUDA) {
            CXM_ASSERT(true, "unfold CUDA not implemented yet");
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE

    unfold_cpu(input, col, N, C, H, W, kH, kW, sH, sW, pH, pW, oH, oW);
}

void Convolution::fold(const f32 *col, f32 *input_grad, const i64 N, const i64 C, const i64 H, const i64 W, const i64 kH, const i64 kW, const i64 sH, const i64 sW, const i64 pH, const i64 pW, const i64 oH, const i64 oW, const DeviceType device) {
    CXM_ASSERT(N == 0 || C == 0, "N and C must be non-zero");

    #if CXM_IS_CUDA_AVAILABLE
        if (device == DeviceType::kCUDA) {
            CXM_ASSERT(true, "fold CUDA not implemented yet");
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE

    fold_cpu(col, input_grad, N, C, H, W,
             kH, kW, sH, sW, pH, pW, oH, oW);
}
