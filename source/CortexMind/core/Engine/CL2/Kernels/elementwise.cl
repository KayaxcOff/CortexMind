__kernel void add(
    __global const float* x,
    __global const float* y,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = x[i] + y[i];
}

__kernel void sub(
    __global const float* x,
    __global const float* y,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = x[i] - y[i];
}

__kernel void mul(
    __global const float* x,
    __global const float* y,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = x[i] * y[i];
}

__kernel void div(
    __global const float* x,
    __global const float* y,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = x[i] / y[i];
}

__kernel void fma(
    __global const float* x,
    __global const float* y,
    __global const float* z,
    __global       float* m,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) m[i] = fma(x[i], y[i], z[i]);  // OpenCL builtin fma
}

__kernel void add_scalar(
    __global const float* x,
    const          float  s,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = x[i] + s;
}

__kernel void mul_scalar(
    __global const float* x,
    const          float  s,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = x[i] * s;
}

__kernel void neg(
    __global const float* x,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = -x[i];
}

__kernel void abs_val(
    __global const float* x,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = fabs(x[i]);
}

__kernel void sqrt_val(
    __global const float* x,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = sqrt(x[i]);
}

__kernel void rsqrt_val(
    __global const float* x,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = rsqrt(x[i]);
}

__kernel void exp_val(
    __global const float* x,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = exp(x[i]);
}

__kernel void log_val(
    __global const float* x,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = log(x[i]);
}

__kernel void pow_val(
    __global const float* x,
    __global const float* y,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = pow(x[i], y[i]);
}

__kernel void clamp_val(
    __global const float* x,
    const          float  lo,
    const          float  hi,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = clamp(x[i], lo, hi);
}

__kernel void relu(
    __global const float* x,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = fmax(0.0f, x[i]);
}

__kernel void leaky_relu(
    __global const float* x,
    const          float  alpha,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = x[i] > 0.0f ? x[i] : alpha * x[i];
}

__kernel void sigmoid(
    __global const float* x,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = 1.0f / (1.0f + exp(-x[i]));
}

__kernel void gelu(
    __global const float* x,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i >= n) return;

    const float c0  = 0.7978845608028654f;
    const float c1  = 0.044715f;
    const float xi  = x[i];
    const float x3  = xi * xi * xi;
    const float arg = c0 * (xi + c1 * x3);

    z[i] = 0.5f * xi * (1.0f + tanh(arg));
}

__kernel void silu(
    __global const float* x,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = x[i] / (1.0f + exp(-x[i]));
}

__kernel void sub_max_exp(
    __global const float* x,
    const          float  max_val,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = exp(x[i] - max_val);
}

__kernel void div_sum(
    __global const float* x,
    const          float  sum,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = x[i] / sum;
}

__kernel void pow_scalar(
    __global const float* x,
    const          float  exponent,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = pow(x[i], exponent);
}

__kernel void eq(
    __global const float* x,
    __global const float* y,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = (x[i] == y[i]) ? 1.0f : 0.0f;
}

__kernel void ne(
    __global const float* x,
    __global const float* y,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = (x[i] != y[i]) ? 1.0f : 0.0f;
}

__kernel void gt(
    __global const float* x,
    __global const float* y,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = (x[i] > y[i]) ? 1.0f : 0.0f;
}

__kernel void lt(
    __global const float* x,
    __global const float* y,
    __global       float* z,
    const          int    n)
{
    const int i = get_global_id(0);
    if (i < n) z[i] = (x[i] < y[i]) ? 1.0f : 0.0f;
}

__kernel void tanh_kernel(__global const float* x,
                          __global float* y,
                          const int n) {
    const int i = get_global_id(0);
    if (i < n) y[i] = tanh(x[i]);
}