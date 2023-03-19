#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

__device__ static uint64_t linear_search(
  const uint32_t *const edgeDst,
  uint32_t uPtr,
  uint32_t uEnd,
  uint32_t vPtr,
  uint32_t vEnd
) {
  uint64_t count = 0;
  uint32_t w1 = edgeDst[uPtr];
  uint32_t w2 = edgeDst[vPtr];

  while (uPtr < uEnd && vPtr < vEnd) {
    if (w1 < w2) {
      w1 = edgeDst[++uPtr];
    } else if (w1 > w2) {
      w2 = edgeDst[++vPtr];
    } else {
      w1 = edgeDst[++uPtr];
      w2 = edgeDst[++vPtr];
      count++;
    }
  }

  return count;
}

__device__ static uint64_t binary_search(
  const uint32_t *const edgeDst,
  uint32_t uPtr,
  uint32_t uEnd,
  uint32_t vPtr,
  uint32_t vEnd
) {
  uint64_t count = 0;
  uint32_t lPtr, lEnd;
  uint32_t bPtr, bEnd;
  
  if (uEnd - uPtr < vEnd - vPtr) {
    lPtr = uPtr;
    lEnd = uEnd;
    bPtr = vPtr;
    bEnd = vEnd;
  } else {
    lPtr = vPtr;
    lEnd = vEnd;
    bPtr = uPtr;
    bEnd = uEnd;
  }

  for (; lPtr < lEnd; lPtr++) {
    uint32_t target = edgeDst[lPtr];
    uint32_t low = bPtr;
    uint32_t high = bEnd - 1;

    while (high - low > 1) {
      uint32_t mid = (low + high) / 2;
      if (edgeDst[mid] < target) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }

    if (edgeDst[low] == target) {
      count++;
    } else if (edgeDst[high] == target) {
      count++;
    }
  }

  return count;
}

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {

  // Determine the source and destination node for the edge
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < numEdges) {
    // Use the row pointer array to determine the start and end of the neighbor list in the column index array
    uint32_t srcNode = edgeSrc[idx];
    uint32_t dstNode = edgeDst[idx];
    uint32_t uPtr = rowPtr[srcNode];
    uint32_t uEnd = rowPtr[srcNode+1];
    uint32_t vPtr = rowPtr[dstNode];
    uint32_t vEnd = rowPtr[dstNode+1];

    // Determine how many elements of those two arrays are common
    triangleCounts[idx] = linear_search(edgeDst, uPtr, uEnd, vPtr, vEnd);
  }
}

__global__ static void kernel_tc_dynamic(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {

  // Determine the source and destination node for the edge
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < numEdges) {
    // Use the row pointer array to determine the start and end of the neighbor list in the column index array
    uint32_t srcNode = edgeSrc[idx];
    uint32_t dstNode = edgeDst[idx];
    uint32_t uPtr = rowPtr[srcNode];
    uint32_t uEnd = rowPtr[srcNode+1];
    uint32_t vPtr = rowPtr[dstNode];
    uint32_t vEnd = rowPtr[dstNode+1];

    // Determine how many elements of those two arrays are common
    uint32_t v = vEnd - vPtr;
    uint32_t u = uEnd - uPtr;
    if (v < u) {
      uint32_t temp = v;
      v = u;
      u = temp;
    }
    if (v >= 64 && v/u >= 6) 
      triangleCounts[idx] = binary_search(edgeDst, uPtr, uEnd, vPtr, vEnd);
    else
      triangleCounts[idx] = linear_search(edgeDst, uPtr, uEnd, vPtr, vEnd);
  }
}

uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) {
  //@@ create a pangolin::Vector (uint64_t) to hold per-edge triangle counts
  // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
  // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.
  pangolin::Vector<uint64_t> tc(view.nnz(), 0);

  uint64_t total = 0;

  dim3 dimBlock(512);
  //@@ calculate the number of blocks needed
  dim3 dimGrid(ceil(view.nnz()/ (float)dimBlock.x));

  if (mode == 1) {

    //@@ launch the linear search kernel here
    kernel_tc<<<dimGrid, dimBlock>>>(tc.data(), view.row_ind(), view.col_ind(), view.row_ptr(), view.nnz());
    cudaDeviceSynchronize();

  } else if (2 == mode) {

    //@@ launch the hybrid search kernel here
    kernel_tc_dynamic<<<dimGrid, dimBlock>>>(tc.data(), view.row_ind(), view.col_ind(), view.row_ptr(), view.nnz());
    cudaDeviceSynchronize();

  } else {
    assert("Unexpected mode");
    return uint64_t(-1);
  }

  //@@ do a global reduction (on CPU or GPU) to produce the final triangle count
  for (int i = 0; i < view.nnz(); i++) {
    total += tc[i];
  }
  return total;
}
