//
// Created by muham on 22.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CL2_FUNCS_HPP
#define CORTEXMIND_CORE_ENGINE_CL2_FUNCS_HPP

#include <CortexMind/core/Engine/CL2/program.hpp>
#include <CortexMind/core/Engine/Memory/buffer.hpp>
#include <CortexMind/core/Tools/err.hpp>
#include <memory>

namespace cortex::_fw::cl2 {
    class registry {
    public:
        static registry& get();

        [[nodiscard]]
        program& elementwise() const {
            return *this->elem_;
        }
        [[nodiscard]]
        program& matmul() const {
            return *this->matmul_;
        }
        [[nodiscard]]
        program& reduce() const {
            return *this->reduce_;
        }

        registry(const registry&)            = delete;
        registry& operator=(const registry&) = delete;

    private:
        explicit registry(const fs::path& kernel_dir);

        std::unique_ptr<program> elem_;
        std::unique_ptr<program> matmul_;
        std::unique_ptr<program> reduce_;

        static fs::path resolve_kernel_dir();
    };

    inline constexpr size_t BLOCK_SIZE = 256;
    inline constexpr size_t TILE_SIZE  = 16;

    [[nodiscard]] inline
    size_t align_to(const size_t n, const size_t align) {
        return ((n + align - 1) / align) * align;
    }

    inline void add(const sys::buffer& x, const sys::buffer& y, sys::buffer& z) {
        CXM_ASSERT(x.count() == y.count() && x.count() == z.count(),
                   "cortex::_fw::cl2::add()", "Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "add",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, y, z, n
        );
    }

    inline void sub(const sys::buffer& x, const sys::buffer& y, sys::buffer& z) {
        CXM_ASSERT(x.count() == y.count() && x.count() == z.count(),
                   "cortex::_fw::cl2::sub()", "Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "sub",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, y, z, n
        );
    }

    inline void mul(const sys::buffer& x, const sys::buffer& y, sys::buffer& z) {
        CXM_ASSERT(x.count() == y.count() && x.count() == z.count(),
                   "cortex::_fw::cl2::mul()", "Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "mul",
            cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            cl::NDRange(BLOCK_SIZE),
            x, y, z, n
        );
    }

    inline void div(const sys::buffer& x, const sys::buffer& y, sys::buffer& z) {
        CXM_ASSERT(x.count() == y.count() && x.count() == z.count(),
                   "cortex::_fw::cl2::div()", "Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "div",
            cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            cl::NDRange(BLOCK_SIZE),
            x, y, z, n
        );
    }

    inline void fma(const sys::buffer& x, const sys::buffer& y, const sys::buffer& z, sys::buffer& m) {
        CXM_ASSERT(x.count() == y.count() &&
                   x.count() == z.count() &&
                   x.count() == m.count(),
                   "cortex::_fw::cl2::fma()", "Buffer Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "fma",
            cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            cl::NDRange(BLOCK_SIZE),
            x, y, z, m, n
        );
    }

    inline void add(const sys::buffer& x, const f32 s, sys::buffer& z) {
        CXM_ASSERT(x.count() == z.count(), "cortex::_fw::cl2::add()", "Buffer Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "add_scalar",
            cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            cl::NDRange(BLOCK_SIZE),
            x, s, z, n
        );
    }

    inline void mul(const sys::buffer& x, const f32 s, sys::buffer& z) {
        CXM_ASSERT(x.count() == z.count(), "cortex::_fw::cl2::mul()", "Buffer Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "mul_scalar",
            cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            cl::NDRange(BLOCK_SIZE),
            x, s, z, n
        );
    }

    inline void neg(const sys::buffer& x, sys::buffer& z) {
        CXM_ASSERT(x.count() == z.count(), "cortex::_fw::cl2::neg()", "Buffer Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "neg",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, z, n
        );
    }

    inline void abs(const sys::buffer& x, sys::buffer& z) {
        CXM_ASSERT(x.count() == z.count(), "cortex::_fw::cl2::abs()", "Buffer Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "abs_val",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, z, n
        );
    }

    inline void sqrt(const sys::buffer& x, sys::buffer& z) {
        CXM_ASSERT(x.count() == z.count(), "cortex::_fw::cl2::sqrt()", "Buffer Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "sqrt_val",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, z, n
        );
    }

    inline void exp(const sys::buffer& x, sys::buffer& z) {
        CXM_ASSERT(x.count() == z.count(), "cortex::_fw::cl2::exp()", "Buffer Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "exp_val",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, z, n
        );
    }

    inline void log(const sys::buffer& x, sys::buffer& z) {
        CXM_ASSERT(x.count() == z.count(), "cortex::_fw::cl2::log()", "Buffer Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "log_val",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, z, n
        );
    }

    inline void pow(const sys::buffer& x, const sys::buffer& y, sys::buffer& z) {
        CXM_ASSERT(x.count() == y.count() && x.count() == z.count(), "cortex::_fw::cl2::pow()", "Buffer Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "pow_val",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, y, z, n
        );
    }

    inline void clamp(const sys::buffer& x, const f32 lo, const f32 hi, sys::buffer& z) {
        CXM_ASSERT(x.count() == z.count(), "cortex::_fw::cl2::clamp()", "Buffer Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "clamp_val",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, lo, hi, z, n
        );
    }

    inline void relu(const sys::buffer& x, sys::buffer& z) {
        CXM_ASSERT(x.count() == z.count(), "cortex::_fw::cl2::relu()", "Buffer Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "relu",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, z, n
        );
    }

    inline void leaky_relu(const sys::buffer& x, const f32 alpha, sys::buffer& z) {
        CXM_ASSERT(x.count() == z.count(), "cortex::_fw::cl2::leaky_relu()", "Buffer Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "leaky_relu",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, alpha, z, n
        );
    }

    inline void sigmoid(const sys::buffer& x, sys::buffer& z) {
        CXM_ASSERT(x.count() == z.count(), "cortex::_fw::cl2::sigmoid()", "Buffer Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "sigmoid",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, z, n
        );
    }

    inline void gelu(const sys::buffer& x, sys::buffer& z) {
        CXM_ASSERT(x.count() == z.count(), "cortex::_fw::cl2::gelu()", "Buffer Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "gelu",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, z, n
        );
    }

    inline void silu(const sys::buffer& x, sys::buffer& z) {
        CXM_ASSERT(x.count() == z.count(), "cortex::_fw::cl2::silu()", "Buffer Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "silu",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, z, n
        );
    }

    inline void matmul(const sys::buffer& x, const sys::buffer& y, sys::buffer& z,
                       const size_t M, const size_t K, const size_t N) {
        CXM_ASSERT(x.count() == M * K, "cortex::_fw::cl2::matmul()", "x size wrong.");
        CXM_ASSERT(y.count() == K * N, "cortex::_fw::cl2::matmul()", "y size wrong.");
        CXM_ASSERT(z.count() == M * N, "cortex::_fw::cl2::matmul()", "z size wrong.");

        registry::get().matmul().run(
            "matmul_tiled",
            ::cl::NDRange(align_to(M, TILE_SIZE), align_to(N, TILE_SIZE)),
            ::cl::NDRange(TILE_SIZE, TILE_SIZE),
            x, y, z,
            static_cast<cl_int>(M),
            static_cast<cl_int>(K),
            static_cast<cl_int>(N)
        );
    }

    inline void matmul_xt(const sys::buffer& x, const sys::buffer& y, sys::buffer& z,
                          const size_t M, const size_t K, const size_t N) {
        CXM_ASSERT(x.count() == M * K, "cortex::_fw::cl2::matmul_xt()", "x size wrong.");
        CXM_ASSERT(y.count() == K * N, "cortex::_fw::cl2::matmul_xt()", "y size wrong.");
        CXM_ASSERT(z.count() == M * N, "cortex::_fw::cl2::matmul_xt()", "z size wrong.");

        registry::get().matmul().run(
            "matmul_xt",
            ::cl::NDRange(align_to(M, TILE_SIZE), align_to(N, TILE_SIZE)),
            ::cl::NDRange(TILE_SIZE, TILE_SIZE),
            x, y, z,
            static_cast<cl_int>(M),
            static_cast<cl_int>(K),
            static_cast<cl_int>(N)
        );
    }

    inline void matmul_batched(const sys::buffer& x, const sys::buffer& y, sys::buffer& z,
                               const size_t B, const size_t M,
                               const size_t K, const size_t N) {
        CXM_ASSERT(x.count() == M * K, "cortex::_fw::cl2::matmul_batched()", "x size wrong.");
        CXM_ASSERT(y.count() == K * N, "cortex::_fw::cl2::matmul_batched()", "y size wrong.");
        CXM_ASSERT(z.count() == M * N, "cortex::_fw::cl2::matmul_batched()", "z size wrong.");

        registry::get().matmul().run(
            "matmul_batched",
            ::cl::NDRange(align_to(M, TILE_SIZE), align_to(N, TILE_SIZE), B),
            ::cl::NDRange(TILE_SIZE, TILE_SIZE, 1),
            x, y, z,
            static_cast<cl_int>(M),
            static_cast<cl_int>(K),
            static_cast<cl_int>(N)
        );
    }

    [[nodiscard]] inline
    f32 reduce_sum(const sys::buffer& x) {
        const size_t n          = x.count();
        const size_t num_groups = align_to(n, BLOCK_SIZE) / BLOCK_SIZE;

        sys::buffer partial(num_groups);
        sys::buffer result(1);

        registry::get().reduce().run(
            "reduce_sum_partial",
            ::cl::NDRange(align_to(n, BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, partial,
            ::cl::Local(BLOCK_SIZE * sizeof(cl_float)),
            static_cast<cl_int>(n)
        );

        registry::get().reduce().run(
            "reduce_sum_final",
            ::cl::NDRange(BLOCK_SIZE),
            ::cl::NDRange(BLOCK_SIZE),
            partial, result,
            ::cl::Local(BLOCK_SIZE * sizeof(cl_float)),
            static_cast<cl_int>(num_groups)
        );

        f32 out;
        result.download(&out, 1);
        return out;
    }

    [[nodiscard]] inline
    f32 reduce_max(const sys::buffer& x) {
        const size_t n          = x.count();
        const size_t num_groups = align_to(n, BLOCK_SIZE) / BLOCK_SIZE;

        sys::buffer partial(num_groups);
        sys::buffer result(1);

        registry::get().reduce().run(
            "reduce_max_partial",
            ::cl::NDRange(align_to(n, BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, partial,
            ::cl::Local(BLOCK_SIZE * sizeof(cl_float)),
            static_cast<cl_int>(n)
        );

        registry::get().reduce().run(
            "reduce_max_final",
            ::cl::NDRange(BLOCK_SIZE),
            ::cl::NDRange(BLOCK_SIZE),
            partial, result,
            ::cl::Local(BLOCK_SIZE * sizeof(cl_float)),
            static_cast<cl_int>(num_groups)
        );

        f32 out;
        result.download(&out, 1);
        return out;
    }

    [[nodiscard]] inline
    f32 reduce_min(const sys::buffer& x) {
        const size_t n          = x.count();
        const size_t num_groups = align_to(n, BLOCK_SIZE) / BLOCK_SIZE;

        sys::buffer partial(num_groups);
        sys::buffer result(1);

        registry::get().reduce().run(
            "reduce_min_partial",
            ::cl::NDRange(align_to(n, BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, partial,
            ::cl::Local(BLOCK_SIZE * sizeof(cl_float)),
            static_cast<cl_int>(n)
        );

        registry::get().reduce().run(
            "reduce_min_final",
            ::cl::NDRange(BLOCK_SIZE),
            ::cl::NDRange(BLOCK_SIZE),
            partial, result,
            ::cl::Local(BLOCK_SIZE * sizeof(cl_float)),
            static_cast<cl_int>(num_groups)
        );

        f32 out;
        result.download(&out, 1);
        return out;
    }

    [[nodiscard]] inline
    f32 reduce_mean(const sys::buffer& x) {
        const size_t n          = x.count();
        const size_t num_groups = align_to(n, BLOCK_SIZE) / BLOCK_SIZE;

        sys::buffer partial(num_groups);
        sys::buffer result(1);

        registry::get().reduce().run(
            "reduce_sum_partial",
            ::cl::NDRange(align_to(n, BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, partial,
            ::cl::Local(BLOCK_SIZE * sizeof(cl_float)),
            static_cast<cl_int>(n)
        );

        // Mean final: sum / n
        registry::get().reduce().run(
            "reduce_mean_final",
            ::cl::NDRange(BLOCK_SIZE),
            ::cl::NDRange(BLOCK_SIZE),
            partial, result,
            ::cl::Local(BLOCK_SIZE * sizeof(cl_float)),
            static_cast<cl_int>(num_groups),
            static_cast<cl_int>(n)
        );

        f32 out;
        result.download(&out, 1);
        return out;
    }

    inline void softmax(const sys::buffer& x, sys::buffer& z, const size_t M, const size_t N) {
        CXM_ASSERT(x.count() == M * N, "cortex::_fw::cl2::softmax()", "x size wrong");
        CXM_ASSERT(z.count() == M * N, "cortex::_fw::cl2::softmax()", "z size wrong");

        registry::get().reduce().run(
            "softmax_rowwise",
            ::cl::NDRange(M * BLOCK_SIZE),
            ::cl::NDRange(BLOCK_SIZE),
            x, z,
            ::cl::Local(BLOCK_SIZE * sizeof(cl_float)),
            static_cast<cl_int>(M),
            static_cast<cl_int>(N)
        );
    }

    inline void layer_norm(const sys::buffer& x, const sys::buffer& gamma,
                           const sys::buffer& beta, sys::buffer& z,
                           const size_t M, const size_t N,
                           const f32 eps = 1e-5f) {
        CXM_ASSERT(x.count()     == M * N, "cortex::_fw::cl2::layer_norm()", "x size wrong.");
        CXM_ASSERT(gamma.count() == N,     "cortex::_fw::cl2::layer_norm()", "gamma size wrong.");
        CXM_ASSERT(beta.count()  == N,     "cortex::_fw::cl2::layer_norm()", "beta size wrong.");
        CXM_ASSERT(z.count()     == M * N, "cortex::_fw::cl2::layer_norm()", "z size wrong.");

        registry::get().reduce().run(
            "layer_norm",
            ::cl::NDRange(M * BLOCK_SIZE),
            ::cl::NDRange(BLOCK_SIZE),
            x, gamma, beta, z,
            ::cl::Local(BLOCK_SIZE * sizeof(cl_float)),
            static_cast<cl_int>(M),
            static_cast<cl_int>(N),
            eps
        );
    }

    inline void pow(const sys::buffer& x, const f32 exponent, sys::buffer& z) {
        CXM_ASSERT(x.count() == z.count(), "cortex::_fw::cl2::pow()", "Buffer sizes don't match.");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "pow_scalar",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, exponent, z, n
        );
    }

    inline void eq(const sys::buffer& x, const sys::buffer& y, sys::buffer& z) {
        CXM_ASSERT(x.count() == y.count() && x.count() == z.count(), "cortex::_fw::cl2::eq()", "Buffer sizes don't match");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "eq",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, y, z, n
        );
    }

    inline void ne(const sys::buffer& x, const sys::buffer& y, sys::buffer& z) {
        CXM_ASSERT(x.count() == y.count() && x.count() == z.count(), "cortex::_fw::cl2::ne()", "Buffer sizes don't match");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "ne",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, y, z, n
        );
    }

    inline void gt(const sys::buffer& x, const sys::buffer& y, sys::buffer& z) {
        CXM_ASSERT(x.count() == y.count() && x.count() == z.count(), "cortex::_fw::cl2::gt()", "Buffer sizes don't match");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "gt",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, y, z, n
        );
    }

    inline void lt(const sys::buffer& x, const sys::buffer& y, sys::buffer& z) {
        CXM_ASSERT(x.count() == y.count() && x.count() == z.count(), "cortex::_fw::cl2::lt()", "Buffer sizes don't match");
        const auto n = static_cast<cl_int>(x.count());
        registry::get().elementwise().run(
            "lt",
            ::cl::NDRange(align_to(x.count(), BLOCK_SIZE)),
            ::cl::NDRange(BLOCK_SIZE),
            x, y, z, n
        );
    }

    inline void tanh(const sys::buffer& x, const sys::buffer& y) {
        const int n = static_cast<int>(x.count());
        registry::get().elementwise().run(
            "tanh_kernel",
            cl::NDRange(cl2::align_to(n, cl2::BLOCK_SIZE)),
            cl::NDRange(cl2::BLOCK_SIZE),
            x.handle(), y.handle(),
            static_cast<cl_int>(n)
        );
    }
} // namespace cortex::_fw::cl2

#endif //CORTEXMIND_CORE_ENGINE_CL2_FUNCS_HPP