#define TILE_SIZE 16

__kernel void matmul_naive(
    __global const float* x,
    __global const float* y,
    __global       float* z,
    const          int    M,
    const          int    K,
    const          int    N)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += x[row * K + k] * y[k * N + col];

    z[row * N + col] = acc;
}

__kernel void matmul_tiled(
    __global const float* x,
    __global const float* y,
    __global       float* z,
    const          int    M,
    const          int    K,
    const          int    N)
{
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);

    const int global_row = get_group_id(0) * TILE_SIZE + local_row;
    const int global_col = get_group_id(1) * TILE_SIZE + local_col;

    __local float tile_x[TILE_SIZE][TILE_SIZE];
    __local float tile_y[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;

    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        const int x_col = t * TILE_SIZE + local_col;
        if (global_row < M && x_col < K)
            tile_x[local_row][local_col] = x[global_row * K + x_col];
        else
            tile_x[local_row][local_col] = 0.0f;

        const int y_row = t * TILE_SIZE + local_row;
        if (y_row < K && global_col < N)
            tile_y[local_row][local_col] = y[y_row * N + global_col];
        else
            tile_y[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k)
            acc += tile_x[local_row][k] * tile_y[k][local_col];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N)
        z[global_row * N + global_col] = acc;
}

__kernel void matmul_batched(
    __global const float* x,
    __global const float* y,
    __global       float* z,
    const          int    M,
    const          int    K,
    const          int    N)
{
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);

    const int global_row = get_group_id(0) * TILE_SIZE + local_row;
    const int global_col = get_group_id(1) * TILE_SIZE + local_col;
    const int batch      = get_global_id(2);

    const int off_x = batch * M * K;
    const int off_y = batch * K * N;
    const int off_z = batch * M * N;

    __local float tile_x[TILE_SIZE][TILE_SIZE];
    __local float tile_y[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;

    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        const int x_col = t * TILE_SIZE + local_col;
        if (global_row < M && x_col < K)
            tile_x[local_row][local_col] = x[off_x + global_row * K + x_col];
        else
            tile_x[local_row][local_col] = 0.0f;

        const int y_row = t * TILE_SIZE + local_row;
        if (y_row < K && global_col < N)
            tile_y[local_row][local_col] = y[off_y + y_row * N + global_col];
        else
            tile_y[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k)
            acc += tile_x[local_row][k] * tile_y[k][local_col];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N)
        z[off_z + global_row * N + global_col] = acc;
}

__kernel void matmul_xt(
    __global const float* x,
    __global const float* y,
    __global       float* z,
    const          int    M,
    const          int    K,
    const          int    N)
{
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);

    const int global_row = get_group_id(0) * TILE_SIZE + local_row;
    const int global_col = get_group_id(1) * TILE_SIZE + local_col;

    __local float tile_x[TILE_SIZE][TILE_SIZE];
    __local float tile_y[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;

    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        const int k_col = t * TILE_SIZE + local_col;


        if (global_row < M && k_col < K)
            tile_x[local_row][local_col] = x[global_row * K + k_col];
        else
            tile_x[local_row][local_col] = 0.0f;


        const int k_row = t * TILE_SIZE + local_row;
        if (global_col < N && k_row < K)
            tile_y[local_row][local_col] = y[global_col * K + k_row];
        else
            tile_y[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k)
            acc += tile_x[local_row][k] * tile_y[k][local_col];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N)
        z[global_row * N + global_col] = acc;
}