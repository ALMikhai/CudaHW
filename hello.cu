#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void test(){
    printf("Hi Cuda World");
}

int main( int argc, char** argv )
{
    test<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
