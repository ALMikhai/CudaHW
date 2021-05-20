#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <cmath>

const double niter = 10000;

struct montecarlo :
       public thrust::unary_function<unsigned int, double>
{
    __host__ __device__
    double operator()(unsigned int thread_id)
    {
        unsigned int seed = 123^thread_id;
        double x, y, z, sum = 0.0;

        thrust::default_random_engine rng(seed);
        thrust::uniform_real_distribution<double> u01(0, 1);
        for (int i = 0; i < niter; ++i)
        {
                x = u01(rng);
                y = u01(rng);
                z = (x * x) + (y * y);
                if (z <= 1)
                        sum += 1;
        }
        return sum;
    }
};
 
int main(int argc, char* argv[])
{
    double pi;
    double count = 0.0;
    count = thrust::transform_reduce(thrust::counting_iterator<double>(0),
                                    thrust::counting_iterator<double>(niter),
                                    montecarlo(),
                                    0.0,
                                    thrust::plus<double>());
    pi = (4.0 * count) / (niter * niter);
    std::cout << "Pi: " << pi << std::endl;
}
