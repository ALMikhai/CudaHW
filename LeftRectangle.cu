#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <numeric>
using namespace std;

__global__ void init(double* result, double step) {
    const int tid = threadIdx.x;
    result[tid] = result[tid] * result[tid]; // x^2
    result[tid] *= step;
}

__global__ void sum(double* input) { // work only when count is a power of two
	const int tid = threadIdx.x;

	int step_size = 1;
	int number_of_threads = blockDim.x;

	while (number_of_threads > 0) {
		if (tid < number_of_threads) {
			const int fst = tid * step_size * 2;
			const int snd = fst + step_size;
			input[fst] += input[snd];
		}

		step_size <<= 1; 
		number_of_threads >>= 1;
	}
}

int main() {
    const double start = 0;
    const double finish = 8;
    const double step = 0.0625;
    int steps = (finish - start) / step;
	const int size = steps * sizeof(double);

    double input[steps];
    for (int i = 0; i < steps; ++i) {
        input[i] = i * step;
    }

	double* d;
	cudaMalloc(&d, size);
    cudaMemcpy(d, input, size, cudaMemcpyHostToDevice);

	init <<<1, steps>>>(d, step);
	sum <<<1, steps / 2 >>>(d);

	double result;
	cudaMemcpy(&result, d, sizeof(double), cudaMemcpyDeviceToHost);

	cout << "Sum is " << result << endl;

	cudaFree(d);

	return 0;
}
