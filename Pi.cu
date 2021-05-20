#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <cuda.h>
#include <curand.h>
#include <iostream>
#include <numeric>

using namespace std;

const long steps = 1 << 21;

__global__ void belongs_circle(double* x, double* y, double* result) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= steps)
		return;
		
    if (((x[tid] - 0.5) * (x[tid] - 0.5)) + ((y[tid] - 0.5) * (y[tid] - 0.5)) <= (0.5 * 0.5)) {
		result[tid] = 1;
	} else {
		result[tid] = 0;
	}
}

int main() {
	const long size = steps * sizeof(double);
	long blockSize = 256;
	long numBlocks = (steps + blockSize - 1) / blockSize;

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

	double *result, *x, *y;
	cudaMalloc(&result, size);
	cudaMalloc(&x, size);
	cudaMalloc(&y, size);
	
	curandGenerateUniformDouble(gen, x, steps);
	curandGenerateUniformDouble(gen, y, steps);

	belongs_circle <<<numBlocks, blockSize>>>(x, y, result);

	double check[steps];
	cudaMemcpy(check, result, size, cudaMemcpyDeviceToHost);

	double sum = 0;
	for (long i = 0; i < steps; ++i) {
		sum += check[i];
	}

	cout << "Sum is " << sum << endl;
	cout << "Pi is " << 4 * sum / steps << endl;

	cudaFree(result);
	cudaFree(x);
	cudaFree(y);

	return 0;
}
