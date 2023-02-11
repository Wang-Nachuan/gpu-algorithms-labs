#include <cstdio>
#include <cstdlib>

#include "template.hu"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

  /********************************************************************
  *
  * Compute C = A x B
  *   where A is a (m x k) matrix
  *   where B is a (k x n) matrix
  *   where C is a (m x n) matrix
  *
  * Use register and shared memory tiling and thread coarsening
  *
  * NOTE: A and C are column major, B is row major
  *
  ********************************************************************/

  // Macros for accessing flattened matrices
  #define A(row,col) A[(row) + (col)*m]
  #define B(row,col) B[(row)*n + (col)]
  #define C(row,col) C[(row) + (col)*m]

  // INSERT KERNEL CODE HERE
  __shared__ float B_ds[TILE_SZ_RATIO][TILE_SZ_B];
  int ti = threadIdx.x;     // Thread index
  int row = blockIdx.x * TILE_SZ_A + ti;
  int col = blockIdx.y * TILE_SZ_B;
  float out0 = 0, out1 = 0, out2 = 0, out3 = 0, out4 = 0, out5 = 0, out6 = 0, out7 = 0,
    out8 = 0, out9 = 0, out10 = 0, out11 = 0, out12 = 0, out13 = 0, out14 = 0, out15 = 0;

  for (int tn = 0; tn < ceil((float) k / TILE_SZ_RATIO); tn++) {
    // Load shared memory
    int rowS = ti / TILE_SZ_B;
    int colS = ti % TILE_SZ_B;
    int rowB = tn * TILE_SZ_RATIO + rowS;
    int colB = col + colS;
    if (rowB < k && colB < n) {
        B_ds[rowS][colS] = B(rowB, colB);
    } else {
        B_ds[rowS][colS] = 0;
    }
    __syncthreads();

    // Load register & Accumulate result
    for (int i = 0; i < TILE_SZ_RATIO; i++) {
        if (row < m && tn * TILE_SZ_RATIO < k) {
            float Ai = A(row, tn * TILE_SZ_RATIO + i);
            out0 += Ai * B_ds[i][0];
            out1 += Ai * B_ds[i][1];
            out2 += Ai * B_ds[i][2];
            out3 += Ai * B_ds[i][3];
            out4 += Ai * B_ds[i][4];
            out5 += Ai * B_ds[i][5];
            out6 += Ai * B_ds[i][6];
            out7 += Ai * B_ds[i][7];
            out8 += Ai * B_ds[i][8];
            out9 += Ai * B_ds[i][9];
            out10 += Ai * B_ds[i][10];
            out11 += Ai * B_ds[i][11];
            out12 += Ai * B_ds[i][12];
            out13 += Ai * B_ds[i][13];
            out14 += Ai * B_ds[i][14];
            out15 += Ai * B_ds[i][15];         
        }
    }
    __syncthreads();
  }

  // Write result
  if (row < m) {
    if (col < n) C(row, col) = out0;
    if (col + 1 < n) C(row, col + 1) = out1;
    if (col + 2 < n) C(row, col + 2) = out2;
    if (col + 3 < n) C(row, col + 3) = out3;
    if (col + 4 < n) C(row, col + 4) = out4;
    if (col + 5 < n) C(row, col + 5) = out5;
    if (col + 6 < n) C(row, col + 6) = out6;
    if (col + 7 < n) C(row, col + 7) = out7;
    if (col + 8 < n) C(row, col + 8) = out8;
    if (col + 9 < n) C(row, col + 9) = out9;
    if (col + 10 < n) C(row, col + 10) = out10;
    if (col + 11 < n) C(row, col + 11) = out11;
    if (col + 12 < n) C(row, col + 12) = out12;
    if (col + 13 < n) C(row, col + 13) = out13;
    if (col + 14 < n) C(row, col + 14) = out14;
    if (col + 15 < n) C(row, col + 15) = out15;
  }

  // SSL Hint (9/6/21): try using just one register for the tile of A 
  // rather than several--in other words, load one value (per thread) 
  // from A and compute using that value rather than loading all values 
  // before doing the computation.  This approach seems to be slightly 
  // faster than the alternative.
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'T') && (transb != 't')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    // Your code need only consider the m, n, k, A, B, and C parameters of
    // the function, which provide the matrix sizes (m, n, k) and data
    // (A, B, C).

    //INSERT CODE HERE
    dim3 DimGrid(ceil((float) m / TILE_SZ_A), ceil((float) n / TILE_SZ_B), 1);
    dim3 DimBlock(TILE_SZ_A, 1, 1);

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
    mysgemm<<<DimGrid, DimBlock>>>(m, n, k, A, B, C);

}

