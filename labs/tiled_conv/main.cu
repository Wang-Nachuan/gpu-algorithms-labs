#include "helper.hpp"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)

__constant__ float const_k[32 * 1 * 5 * 5];

// Sequential code for the forward path of the convolution layer
// You should not modify this code
static void conv_forward_valid(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y,
                               const shape &ydims) {
  std::fill(Y, Y + ydims.flattened_length(), 0);

  for (auto i : range(0, ydims.num)) {
    for (auto m : range(0, ydims.depth )) {   // for each output feature map
      for (auto h : range(0, ydims.height)) { // for each output element
        for (auto w : range(0, ydims.width )) {
          const auto yoffset = ((i * ydims.depth + m) * ydims.height + h) * ydims.width + w;
          for (auto c : range(0, xdims.depth )) {     // sum over all input feature maps
            for (auto p : range(0, wdims.height)) {   // filter height
              for (auto q : range(0, wdims.width )) { // filter width
                const auto xoffset = ((((i * xdims.depth) + c) * xdims.height) + (h + p)) * xdims.width + (w + q);
                const auto woffset = ((((m * wdims.depth) + c) * wdims.height) + p) * wdims.width + q;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Baseline GPU kernel code for forward convolution.
// One thread per output index
// You should not modify this kernel as it is used for correctness comparison.
// Instead, define a new one below
__global__ void conv_forward_baseline_kernel(const float *X, const shape xdims, const float *W, const shape wdims, float *Y,
                                    const shape ydims) {


  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = gx; i < ydims.num * ydims.depth * ydims.height * ydims.width; i += blockDim.x * gridDim.x) {
    Y[i] = 0.f;
  }

  for (size_t i = gx; i < ydims.num; i += gridDim.x * blockDim.x) {
    for (auto m : range(0, ydims.depth )) { // for each output feature map
      for (auto h : range(0, ydims.height)) { // for each output element
        for (auto w : range(0, ydims.width )) {
          const size_t yoffset = ((i * ydims.depth + m) * ydims.height + h) * ydims.width + w;
          for (auto c : range(0, xdims.depth )) {     // sum over all input feature maps
            for (auto p : range(0, wdims.height)) {   // filter height
              for (auto q : range(0, wdims.width )) { // filter width
                const size_t xoffset = ((((i * xdims.depth) + c) * xdims.height) + (h + p)) * xdims.width + (w + q);
                const size_t woffset = ((((m * wdims.depth) + c) * wdims.height) + p) * wdims.width + q;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Host code to configure baseline GPU kernel
static void convlayer_gpu_baseline(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y,
  const shape &ydims) {

  dim3 dimGrid(1);
  dim3 dimBlock(32);

  conv_forward_baseline_kernel<<<dimGrid, dimBlock>>>(X, xdims, W, wdims, Y, ydims);
  THROW_IF_ERROR(cudaGetLastError());

}

// Implement your optimized kernel here.
// Make any modifications you wish.
// Don't forget to modify the host code below, if needed!
__global__ void conv_forward_opt_kernel(const float *X, const shape xdims, const float *W, const shape wdims, float *Y,
  const shape ydims) {

  //@@ YOUR CODE HERE!
  
  // Virtualize input/weight as matrix
  const int b = blockIdx.z;
  const int M = ydims.depth;
  const int C1 = xdims.depth;
  const int H = xdims.height;
  const int W1 = xdims.width;
  const int K = wdims.height;
  const int H_out = H - K + 1;
  const int W_out = W1 - K + 1;
  const int K_size = K * K;

  #define C(ri, ci) Y[(b) * (M * H_out * W_out) + (ri) * (H_out * W_out) + ((ci) / W_out) * (W_out) + ((ci) % W_out)]
  #define B(ri, ci) X[(b) * (C1 * H * W1) + ((ri) / K_size) * (H * W1) + (((ri) % K_size) / K + ((ci) / W_out)) * (W1) + (((ri) % K_size) % K + ((ci) % W_out))]
  #define A(ri, ci) const_k[(ri) * (C1 * K * K) + (ci)]

  const int m = M;
  const int k = K_size * C1;
  const int n = H_out * W_out;

  // Do the normal tiled matrix multiplication
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

  #undef A
  #undef B
  #undef C
}

// Host code to configure baseline GPU kernel
static void convlayer_gpu_opt(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y,
  const shape &ydims) {

  // Modify this code to configure your optimized kernel.
  //@@ YOUR CODE HERE!!!
  int H_out = xdims.height - wdims.height + 1;
  int W_out = xdims.width - wdims.width + 1;
  int mtx_numRow = ydims.depth;
  int mtx_numCol = H_out * W_out;
  dim3 dimGrid(ceil((float) mtx_numRow/TILE_SZ_A), ceil((float) mtx_numCol/TILE_SZ_B), ydims.num);
  dim3 dimBlock(TILE_SZ_A, 1, 1);
  conv_forward_opt_kernel<<<dimGrid, dimBlock>>>(X, xdims, W, wdims, Y, ydims);
  THROW_IF_ERROR(cudaGetLastError());

}


static int eval(const shape wDims, const shape xDims, bool doVerify) {

  // Generate model
  const auto conf_info = std::string("conv[wDims:") + std::to_string(wDims.num) + "," +
                                                      std::to_string(wDims.depth) + "," +
                                                      std::to_string(wDims.height) + "," +
                                                      std::to_string(wDims.width) +
                                                      " xDims:" + std::to_string(xDims.num) + "," +
                                                      std::to_string(xDims.depth) + "," +
                                                      std::to_string(xDims.height) + "," +
                                                      std::to_string(xDims.width) + "]";
  INFO("Running "  << conf_info);

  // Generate convolution weights
  float *hostW = allocate<float>(wDims);
  generate_convfilters(hostW, wDims);

  // generate input feature map
  float *hostX = allocate<float>(xDims);
  generate_data(hostX, xDims);

  // generate output feature map for verification
  const shape ydims = {xDims.num, wDims.num, (xDims.height - wDims.height + 1),
      (xDims.width - wDims.width + 1)};
  INFO("Allocating output tensor [" << ydims.num << "," << ydims.depth << "," << ydims.height << "," << ydims.width << "]");
  float *hostY = allocate<float>(ydims);
  float *expected = allocate<float>(ydims);
  generate_data(hostY, ydims);


  const size_t wByteCount = wDims.flattened_length() * sizeof(float);
  const size_t xByteCount = xDims.flattened_length() * sizeof(float);
  const size_t yByteCount = ydims.flattened_length() * sizeof(float);

  float *deviceW = nullptr, *deviceX = nullptr, *deviceY = nullptr;
  timer_start("Allocating GPU memory.");
  THROW_IF_ERROR(cudaMalloc((void **)&deviceW, wByteCount));
  THROW_IF_ERROR(cudaMalloc((void **)&deviceX, xByteCount));
  THROW_IF_ERROR(cudaMalloc((void **)&deviceY, yByteCount));
  timer_stop();


  timer_start("Copying inputs to the GPU.");
  THROW_IF_ERROR(cudaMemcpy(deviceW, hostW, wByteCount, cudaMemcpyDefault));
  THROW_IF_ERROR(cudaMemcpy(deviceX, hostX, xByteCount, cudaMemcpyDefault));
  cudaMemcpyToSymbol(const_k, hostW, wByteCount);
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  convlayer_gpu_opt(deviceX, xDims, deviceW, wDims, deviceY, ydims);
  THROW_IF_ERROR(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  if (doVerify) {
    timer_start("Copying output to the CPU");
    THROW_IF_ERROR(cudaMemcpy(hostY, deviceY, yByteCount, cudaMemcpyDefault));
    timer_stop();

    convlayer_gpu_baseline(deviceX, xDims, deviceW, wDims, deviceY, ydims);
    THROW_IF_ERROR(cudaDeviceSynchronize());
    THROW_IF_ERROR(cudaMemcpy(expected, deviceY, yByteCount, cudaMemcpyDefault));
    // conv_forward_valid(hostX, xDims, hostW, wDims, expected, ydims);
    verify(expected, hostY, ydims);
  }

  THROW_IF_ERROR(cudaFree(deviceW));
  THROW_IF_ERROR(cudaFree(deviceX));
  THROW_IF_ERROR(cudaFree(deviceY));
  free(hostW);
  free(hostX);
  free(hostY);
  free(expected);

  return 0;
}



TEST_CASE("Convlayer", "[convlayer]") {
#if 1
  // test five times in case code errors depend on data
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32,1,5,5}, {20,1,28,28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32,1,5,5}, {20,1,28,28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32,1,5,5}, {20,1,28,28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32,1,5,5}, {20,1,28,28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32,1,5,5}, {20,1,28,28}, true);
  }
#else
  SECTION("[wDims:32,1,5,5 xDims:50000,1,28,28]") {
    eval({32,1,5,5}, {50000,1,28,28}, false);
  }
#endif
}
