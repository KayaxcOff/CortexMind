// kernels/reduce.cl

#define BLOCK_SIZE 256

// ─── Yardımcı: Local Memory Tree Reduction ─────────────────────────
// Bu macro'yu her kernel içinde kullanacağız
#define LOCAL_REDUCE_SUM(local_buf, local_id, group_size)   \
    barrier(CLK_LOCAL_MEM_FENCE);                           \
    for (int stride = (group_size) / 2; stride > 0; stride >>= 1) { \
        if ((local_id) < stride)                            \
            (local_buf)[(local_id)] += (local_buf)[(local_id) + stride]; \
        barrier(CLK_LOCAL_MEM_FENCE);                       \
    }

#define LOCAL_REDUCE_MAX(local_buf, local_id, group_size)   \
    barrier(CLK_LOCAL_MEM_FENCE);                           \
    for (int stride = (group_size) / 2; stride > 0; stride >>= 1) { \
        if ((local_id) < stride)                            \
            (local_buf)[(local_id)] = fmax((local_buf)[(local_id)], \
                                           (local_buf)[(local_id) + stride]); \
        barrier(CLK_LOCAL_MEM_FENCE);                       \
    }

#define LOCAL_REDUCE_MIN(local_buf, local_id, group_size)   \
    barrier(CLK_LOCAL_MEM_FENCE);                           \
    for (int stride = (group_size) / 2; stride > 0; stride >>= 1) { \
        if ((local_id) < stride)                            \
            (local_buf)[(local_id)] = fmin((local_buf)[(local_id)], \
                                           (local_buf)[(local_id) + stride]); \
        barrier(CLK_LOCAL_MEM_FENCE);                       \
    }

// ─── Sum Reduction ─────────────────────────────────────────────────
// Aşama 1: Her work-group kendi bloğunu toplar
// x: [n], out: [num_groups] — her group bir partial sum yazar

__kernel void reduce_sum_partial(
    __global const float* x,
    __global       float* out,
    __local        float* local_buf,
    const          int    n)
{
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);

    // Sınır dışı elemanlar için 0 yükle
    local_buf[lid] = (gid < n) ? x[gid] : 0.0f;

    LOCAL_REDUCE_SUM(local_buf, lid, BLOCK_SIZE)

    // Sadece ilk work-item group sonucunu yazar
    if (lid == 0)
        out[get_group_id(0)] = local_buf[0];
}

// Aşama 2: Partial sum'ları topla — out küçükse tek geçişte biter
__kernel void reduce_sum_final(
    __global const float* partial,
    __global       float* result,
    __local        float* local_buf,
    const          int    n)
{
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);

    local_buf[lid] = (gid < n) ? partial[gid] : 0.0f;

    LOCAL_REDUCE_SUM(local_buf, lid, BLOCK_SIZE)

    if (lid == 0)
        result[0] = local_buf[0];
}

// ─── Max Reduction ─────────────────────────────────────────────────

__kernel void reduce_max_partial(
    __global const float* x,
    __global       float* out,
    __local        float* local_buf,
    const          int    n)
{
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);

    local_buf[lid] = (gid < n) ? x[gid] : -INFINITY;

    LOCAL_REDUCE_MAX(local_buf, lid, BLOCK_SIZE)

    if (lid == 0)
        out[get_group_id(0)] = local_buf[0];
}

__kernel void reduce_max_final(
    __global const float* partial,
    __global       float* result,
    __local        float* local_buf,
    const          int    n)
{
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);

    local_buf[lid] = (gid < n) ? partial[gid] : -INFINITY;

    LOCAL_REDUCE_MAX(local_buf, lid, BLOCK_SIZE)

    if (lid == 0)
        result[0] = local_buf[0];
}

// ─── Min Reduction ─────────────────────────────────────────────────

__kernel void reduce_min_partial(
    __global const float* x,
    __global       float* out,
    __local        float* local_buf,
    const          int    n)
{
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);

    local_buf[lid] = (gid < n) ? x[gid] : INFINITY;

    LOCAL_REDUCE_MIN(local_buf, lid, BLOCK_SIZE)

    if (lid == 0)
        out[get_group_id(0)] = local_buf[0];
}

__kernel void reduce_min_final(
    __global const float* partial,
    __global       float* result,
    __local        float* local_buf,
    const          int    n)
{
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);

    local_buf[lid] = (gid < n) ? partial[gid] : INFINITY;

    LOCAL_REDUCE_MIN(local_buf, lid, BLOCK_SIZE)

    if (lid == 0)
        result[0] = local_buf[0];
}

// ─── Mean ──────────────────────────────────────────────────────────
// Sum reduction'dan sonra n'e böl

__kernel void reduce_mean_final(
    __global const float* partial,
    __global       float* result,
    __local        float* local_buf,
    const          int    num_groups,
    const          int    n)
{
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);

    local_buf[lid] = (gid < num_groups) ? partial[gid] : 0.0f;

    LOCAL_REDUCE_SUM(local_buf, lid, BLOCK_SIZE)

    if (lid == 0)
        result[0] = local_buf[0] / (float)n;
}

// ─── Softmax — Row-wise ────────────────────────────────────────────
// Her satır bağımsız softmax — sequence boyutu makul (<= BLOCK_SIZE*4) varsayılır
// x: [M x N], z: [M x N]

__kernel void softmax_rowwise(
    __global const float* x,
    __global       float* z,
    __local        float* local_buf,
    const          int    M,
    const          int    N)
{
    const int row = get_group_id(0);   // Her work-group bir satır
    const int lid = get_local_id(0);

    if (row >= M) return;

    const int offset = row * N;

    // ── Adım 1: Max bul (numerically stable için) ──
    local_buf[lid] = -INFINITY;
    for (int i = lid; i < N; i += BLOCK_SIZE)
        local_buf[lid] = fmax(local_buf[lid], x[offset + i]);

    LOCAL_REDUCE_MAX(local_buf, lid, BLOCK_SIZE)

    const float row_max = local_buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // ── Adım 2: exp(x - max) yaz, sum hesapla ──
    local_buf[lid] = 0.0f;
    for (int i = lid; i < N; i += BLOCK_SIZE) {
        const float val = exp(x[offset + i] - row_max);
        z[offset + i]   = val;
        local_buf[lid]  += val;
    }

    LOCAL_REDUCE_SUM(local_buf, lid, BLOCK_SIZE)

    const float row_sum = local_buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // ── Adım 3: Normalize ──
    for (int i = lid; i < N; i += BLOCK_SIZE)
        z[offset + i] /= row_sum;
}

// ─── Layer Norm ────────────────────────────────────────────────────
// Her satır için: (x - mean) / sqrt(var + eps) * gamma + beta
// x: [M x N], gamma/beta: [N], z: [M x N]

__kernel void layer_norm(
    __global const float* x,
    __global const float* gamma,
    __global const float* beta,
    __global       float* z,
    __local        float* local_buf,
    const          int    M,
    const          int    N,
    const          float  eps)
{
    const int row = get_group_id(0);
    const int lid = get_local_id(0);

    if (row >= M) return;

    const int offset = row * N;

    // ── Adım 1: Mean ──
    local_buf[lid] = 0.0f;
    for (int i = lid; i < N; i += BLOCK_SIZE)
        local_buf[lid] += x[offset + i];

    LOCAL_REDUCE_SUM(local_buf, lid, BLOCK_SIZE)

    const float mean = local_buf[0] / (float)N;
    barrier(CLK_LOCAL_MEM_FENCE);

    // ── Adım 2: Variance ──
    local_buf[lid] = 0.0f;
    for (int i = lid; i < N; i += BLOCK_SIZE) {
        const float diff = x[offset + i] - mean;
        local_buf[lid] += diff * diff;
    }

    LOCAL_REDUCE_SUM(local_buf, lid, BLOCK_SIZE)

    const float inv_std = rsqrt(local_buf[0] / (float)N + eps);
    barrier(CLK_LOCAL_MEM_FENCE);

    // ── Adım 3: Normalize + scale + shift ──
    for (int i = lid; i < N; i += BLOCK_SIZE)
        z[offset + i] = (x[offset + i] - mean) * inv_std * gamma[i] + beta[i];
}

// Variance partial — her work-group (x - mean)² toplar
__kernel void reduce_var_partial(
    __global const float* x,
    const          float  mean,
    __global       float* out,
    __local        float* local_buf,
    const          int    n)
{
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);

    const float diff = (gid < n) ? (x[gid] - mean) : 0.0f;
    local_buf[lid] = diff * diff;

    LOCAL_REDUCE_SUM(local_buf, lid, BLOCK_SIZE)

    if (lid == 0)
        out[get_group_id(0)] = local_buf[0];
}