#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512
#define TILE_SIZE 512

// Ceiling funciton for X / Y.
__host__ __device__ static inline int ceil_div(int x, int y) {
    return (x - 1) / y + 1;
}

// Co-rank: find i in A, given k in C
__device__ int co_rank (int k, float* A, int m, float* B, int n)
{
    int low = (k > n ? k - n : 0);
    int high = (k < m ? k : m);

    while (low < high) {
        int i = low + (high - low) / 2;
        int j = k - i;
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            high = i - 1;
        } else if (j > 0 && i < m && A[i] <= B[j - 1]) {
            low = i + 1;
        } else {
            return i;
        }
    }

    return low;
}

/******************************************************************************
 GPU kernels
*******************************************************************************/

/*
 * Sequential merge implementation is given. You can use it in your kernels.
 */
__device__ void merge_sequential(float* A, int A_len, float* B, int B_len, float* C) {
    int i = 0, j = 0, k = 0;

    while ((i < A_len) && (j < B_len)) {
        C[k++] = A[i] <= B[j] ? A[i++] : B[j++];
    }

    if (i == A_len) {
        while (j < B_len) {
            C[k++] = B[j++];
        }
    } else {
        while (i < A_len) {
            C[k++] = A[i++];
        }
    }
}

/*
 * Basic parallel merge kernel using co-rank function
 * A, A_len - input array A and its length
 * B, B_len - input array B and its length
 * C - output array holding the merged elements.
 *      Length of C is A_len + B_len (size pre-allocated for you)
 */
__global__ void gpu_merge_basic_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int elt = ceil_div(A_len + B_len, gridDim.x * blockDim.x);
    int k_curr = tid * elt;
    int k_next = k_curr + elt;
    if (A_len + B_len < k_curr) {k_curr = A_len + B_len;}
    if (A_len + B_len < k_next) {k_next = A_len + B_len;}

    int i_curr = co_rank(k_curr, A, A_len, B, B_len);
    int i_next = co_rank(k_next, A, A_len, B, B_len);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}

/*
 * Arguments are the same as gpu_merge_basic_kernel.
 * In this kernel, use shared memory to increase the reuse.
 */
__global__ void gpu_merge_tiled_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
    __shared__ float tileAB[2 * TILE_SIZE];
    __shared__ int i1, i2;
    float *tileA = &tileAB[0];
    float *tileB = &tileAB[TILE_SIZE];

    int elt = ceil_div(A_len + B_len, gridDim.x);
    int c_curr = min(blockIdx.x * elt, A_len + B_len);
    int c_next = min((blockIdx.x + 1) * elt, A_len + B_len);

    if (threadIdx.x == 0) {
        i1 = co_rank(c_curr, A, A_len, B, B_len);
        i2 = co_rank(c_next, A, A_len, B, B_len);
    }

    __syncthreads();

    int a_curr = i1;
    int a_next = i2;
    int b_curr = c_curr - a_curr;
    int b_next = c_next - a_next;

    int a_length = a_next - a_curr;
    int b_length = b_next - b_curr;
    int c_length = c_next - c_curr;

    int num_tiles = ceil_div(c_length, TILE_SIZE);
    int a_consumed = 0;
    int b_consumed = 0;
    int c_produced = 0;

    int per_thread = ceil_div(TILE_SIZE, blockDim.x);

    for (int count = 0; count < num_tiles; count++) {
        // Load tile
        for (int i = 0; i < TILE_SIZE; i += blockDim.x) {
            if (i + threadIdx.x < min(a_length - a_consumed, TILE_SIZE)) {
                tileA[i + threadIdx.x] = A[a_curr + a_consumed + i + threadIdx.x];
            }
            if (i + threadIdx.x < min(b_length - b_consumed, TILE_SIZE)) {
                tileB[i + threadIdx.x] = B[b_curr + b_consumed + i + threadIdx.x];
            }
        }

        __syncthreads();

        // Process tile
        int c_curr_tile = min(min(per_thread * threadIdx.x, c_length - c_produced), TILE_SIZE);
        int c_next_tile = min(min(per_thread * (threadIdx.x + 1), c_length - c_produced), TILE_SIZE);

        int a_in_tile = min(TILE_SIZE, a_length - a_consumed);
        int b_in_tile = min(TILE_SIZE, b_length - b_consumed);
        int a_curr_tile = co_rank(c_curr_tile, tileA, a_in_tile, tileB, b_in_tile);
        int a_next_tile = co_rank(c_next_tile, tileA, a_in_tile, tileB, b_in_tile);
        int b_curr_tile = c_curr_tile - a_curr_tile;
        int b_next_tile = c_next_tile - a_next_tile;

        merge_sequential(&tileA[a_curr_tile], a_next_tile - a_curr_tile, 
            &tileB[b_curr_tile], b_next_tile - b_curr_tile, 
            &C[c_curr + c_produced + c_curr_tile]
        );

        // Advance variables for next tile
        a_consumed = a_consumed + co_rank(min(TILE_SIZE, c_length - c_produced), tileA, min(TILE_SIZE, a_length - a_consumed), tileB, min(TILE_SIZE, b_length - b_consumed));
        c_produced = c_produced +  min(TILE_SIZE, c_length - c_produced);
        b_consumed = c_produced - a_consumed;
        
        __syncthreads();
    }
}

/*
 * gpu_merge_circular_buffer_kernel is optional.
 * The implementation will be similar to tiled merge kernel.
 * You'll have to modify co-rank function and sequential_merge
 * to accommodate circular buffer.
 */
__global__ void gpu_merge_circular_buffer_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
}

/******************************************************************************
 Functions
*******************************************************************************/

void gpu_basic_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_basic_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_tiled_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_tiled_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_circular_buffer_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_circular_buffer_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}
