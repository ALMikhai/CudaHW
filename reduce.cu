#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void reduce(int* inData, int* outData) {
    __shared__ int data[BLOCK_SIZE];
    int threadId = threadIsx.x;
    int i = blockIdx.x * blockDim.x + threadId;
    data[threadId] = inData[i];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s){
            data [threadId] += data [threadId + s];
        }
        __syncthreads();
    }
    if (threadId == 0) {
        outData[blockIdx.x] = data[0];
    }
}

int main( int argc, char** argv ) {
    test<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
