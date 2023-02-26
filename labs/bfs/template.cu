#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 4096

// Number of warp queues per block
#define NUM_WARP_QUEUES 8
// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY (BQ_CAPACITY / NUM_WARP_QUEUES)

/******************************************************************************
 GPU kernels
*******************************************************************************/

__global__ void gpu_global_queueing_kernel(unsigned int *nodePtrs,
                                          unsigned int *nodeNeighbors,
                                          unsigned int *nodeVisited,
                                          unsigned int *currLevelNodes,
                                          unsigned int *nextLevelNodes,
                                          unsigned int *numCurrLevelNodes,
                                          unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  // Loop over all nodes in the current level
  for (; tx < *numCurrLevelNodes; tx += gridDim.x * blockDim.x) {
    int node = currLevelNodes[tx];
    // Loop over all neighbors of the node
    for (int i = nodePtrs[node]; i < nodePtrs[node + 1]; i++) {
      int neighbor = nodeNeighbors[i];
      // If neighbor hasn't been visited yet
      if (!atomicExch(&(nodeVisited[neighbor]), 1)) {
        // Add neighbor to global queue
        int idx = atomicAdd(numNextLevelNodes, 1);
        nextLevelNodes[idx] = neighbor;
      }
    }
  }
}

__global__ void gpu_block_queueing_kernel(unsigned int *nodePtrs,
                                         unsigned int *nodeNeighbors,
                                         unsigned int *nodeVisited,
                                         unsigned int *currLevelNodes,
                                         unsigned int *nextLevelNodes,
                                         unsigned int *numCurrLevelNodes,
                                         unsigned int *numNextLevelNodes) {
  // INSERT KERNEL CODE HERE
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  // Initialize shared memory queue (size should be BQ_CAPACITY)
  __shared__ unsigned int bq[BQ_CAPACITY];
  __shared__ int numBqNodes, gqOffset;

  if (threadIdx.x == 0) {
    numBqNodes = 0;
  }
  __syncthreads();

  // Loop over all nodes in the current level
  for (; tx < *numCurrLevelNodes; tx += gridDim.x * blockDim.x) {
    int node = currLevelNodes[tx];
    // Loop over all neighbors of the node
    for (int i = nodePtrs[node]; i < nodePtrs[node + 1]; i++) {
      int neighbor = nodeNeighbors[i];
      // If neighbor hasn't been visited yet
      if (!atomicExch(&(nodeVisited[neighbor]), 1)) {
        int bqIdx = atomicAdd(&numBqNodes, 1);
        if (bqIdx < BQ_CAPACITY) {
          // Add neighbor to block queue
          bq[bqIdx] = neighbor;
        } else {
          // If full, add neighbor to global queue
          numBqNodes = BQ_CAPACITY;
          nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = neighbor;
        }
      }
    }
  }
  __syncthreads();

  // Allocate space for block queue to go into global queue
  if (threadIdx.x == 0) {
    gqOffset = atomicAdd(numNextLevelNodes, numBqNodes);
  }
  __syncthreads();

  // Store block queue in global queue
  for (int i = threadIdx.x; i < numBqNodes; i += blockDim.x) {
    nextLevelNodes[gqOffset + i] = bq[i];
  }
}

__global__ void gpu_warp_queueing_kernel(unsigned int *nodePtrs,
                                        unsigned int *nodeNeighbors,
                                        unsigned int *nodeVisited,
                                        unsigned int *currLevelNodes,
                                        unsigned int *nextLevelNodes,
                                        unsigned int *numCurrLevelNodes,
                                        unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int wqTx = threadIdx.x % NUM_WARP_QUEUES;     // To different bank
  // This version uses NUM_WARP_QUEUES warp queues of capacity 
  // WQ_CAPACITY.  Be sure to interleave them as discussed in lecture.
  __shared__ unsigned int wq[WQ_CAPACITY][NUM_WARP_QUEUES];
  __shared__ int numWqNodes[NUM_WARP_QUEUES], bqOffset[NUM_WARP_QUEUES];
  // Don't forget that you also need a block queue of capacity BQ_CAPACITY.
  __shared__ unsigned int bq[BQ_CAPACITY];
  __shared__ int numBqNodes, gqOffset;

  // Initialize shared memory queues (warp and block)
  if (threadIdx.x < NUM_WARP_QUEUES) {
    numWqNodes[threadIdx.x] = 0;
    if (threadIdx.x == 0) {
      numBqNodes = 0;
    }
  }
  __syncthreads();

  // Loop over all nodes in the current level
  for (; tx < *numCurrLevelNodes; tx += gridDim.x * blockDim.x) {
    int node = currLevelNodes[tx];
    // Loop over all neighbors of the node
    for (int i = nodePtrs[node]; i < nodePtrs[node + 1]; i++) {
      int neighbor = nodeNeighbors[i];
      // If neighbor hasn't been visited yet
      if (!atomicExch(&(nodeVisited[neighbor]), 1)) {
        int wqIdx = atomicAdd(&(numWqNodes[wqTx]), 1);
        if (wqIdx < WQ_CAPACITY) {
          wq[wqIdx][wqTx] = neighbor;
        } else {
          numWqNodes[wqTx] = WQ_CAPACITY;
          int bqIdx = atomicAdd(&numBqNodes, 1);
          if (bqIdx < BQ_CAPACITY) {
            bq[bqIdx] = neighbor;
          } else {
            numBqNodes = BQ_CAPACITY;
            nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = neighbor;
          }
        }
      }
    }
  }
  __syncthreads();
  
  // Allocate space for warp queue to go into block queue
  if (threadIdx.x == 0) {
    bqOffset[0] = numBqNodes;
    for (int i = 1; i < NUM_WARP_QUEUES; i++) {
      bqOffset[i] = bqOffset[i - 1] + numWqNodes[i - 1];
    }
  }
  __syncthreads();

  // Allocate space for block queue to go into global queue
  if (threadIdx.x == 0) {
    numBqNodes = bqOffset[NUM_WARP_QUEUES - 1] + numWqNodes[NUM_WARP_QUEUES - 1];
    if (numBqNodes > BQ_CAPACITY) {
      numBqNodes = BQ_CAPACITY;
    }
    gqOffset = atomicAdd(numNextLevelNodes, numBqNodes);
  }
  __syncthreads();

  // Store warp queues in block queue (use one warp or one thread per queue)
  // Add any nodes that don't fit (remember, space was allocated above)
  //    to the global queue
  for (int i = 0; i < NUM_WARP_QUEUES; i++) {
    for (int j = threadIdx.x; j < numWqNodes[i]; j += blockDim.x) {
      int w2bIdx = bqOffset[i] + j;
      if (w2bIdx < BQ_CAPACITY) {
        bq[w2bIdx] = wq[j][i];
      } else {
        nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = wq[j][i];
      }
    }
  }
  __syncthreads();

  // Store block queue in global queue
  for (int i = threadIdx.x; i < numBqNodes; i += blockDim.x) {
    nextLevelNodes[gqOffset + i] = bq[i];
  }
}

/******************************************************************************
 Functions
*******************************************************************************/
// DON NOT MODIFY THESE FUNCTIONS!

void gpu_global_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                        unsigned int *nodeVisited, unsigned int *currLevelNodes,
                        unsigned int *nextLevelNodes,
                        unsigned int *numCurrLevelNodes,
                        unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queueing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_block_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                       unsigned int *nodeVisited, unsigned int *currLevelNodes,
                       unsigned int *nextLevelNodes,
                       unsigned int *numCurrLevelNodes,
                       unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queueing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_warp_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                      unsigned int *nodeVisited, unsigned int *currLevelNodes,
                      unsigned int *nextLevelNodes,
                      unsigned int *numCurrLevelNodes,
                      unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queueing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}
