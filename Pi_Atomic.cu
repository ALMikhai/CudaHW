#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <cuda.h>
#include <curand.h>
#include <iostream>
#include <numeric>

using namespace std;

const int threadSideSize = 32;
const int blocksSideSize = 255;

__global__ void calculate(int* sum) {
    int xBlock = blockIdx.x / blocksSideSize;
    int yBlock = blockIdx.x % blocksSideSize;
    int xThread = threadIdx.x / threadSideSize;
    int yThread = threadIdx.x % threadSideSize;

    int positionX = xBlock * threadSideSize + xThread;
    int positionY = yBlock * threadSideSize + yThread;

    if (sqrt((float) ((positionX * positionX) + (positionY * positionY))) <= (blocksSideSize * threadSideSize)) {
        atomicAdd(sum, 1);
    }
}

int main() {
    int blockSize = threadSideSize * threadSideSize;
    int numBlocks = blocksSideSize * blocksSideSize;

    int sum = 0;
    int *sumDevice;
	cudaMalloc(&sumDevice, sizeof(int));

	calculate<<<numBlocks, blockSize>>>(sumDevice);
	cudaMemcpy(&sum, sumDevice, sizeof(int), cudaMemcpyDeviceToHost);
    cout << "pi = " << (double)(4 * sum) / (double)((blocksSideSize * threadSideSize) * (blocksSideSize * threadSideSize)) << endl;

    cudaFree(sumDevice);
	return 0;
}
