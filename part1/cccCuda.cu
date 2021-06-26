//
// Tomás Oliveira e Silva, November 2017
//

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// program configuration
//

#ifndef N_SAMPLES
#define N_SAMPLES (1 << 16)
#endif

static void ccc_samples_cpu_kernel(double *results_data, double *samples_data_x, double *samples_data_y, unsigned int n_samples, unsigned int point);
__global__ static void ccc_samples_cuda_kernel(double *__restrict__ results_data, double *__restrict__ samples_data_x, double *__restrict__ samples_data_y,
                                               unsigned int n_samples);
static double get_delta_time(void);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main program
//

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);
    if (sizeof(unsigned int) != (size_t)4)
        return 1; // fail with prejudice if an integer does not have 4 bytes

    // set up device
    int dev = 0;
    int i;

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // create memory areas in host and device memory where the disk samples data will be stored
    size_t samples_data_size;
    double *host_results_data, *host_samples_data_x, *host_samples_data_y;
    double *device_results_data, *device_samples_data_x, *device_samples_data_y;

    samples_data_size = (size_t)N_SAMPLES * sizeof(double);
    if ((samples_data_size * 3) > (size_t)1.3e9)
    {
        fprintf(stderr, "The GTX 480 cannot handle more than 1.5GiB of memory!\n");
        exit(1);
    }
    printf("Total samples data size: %lu\n", samples_data_size);

    host_results_data = (double *)malloc(samples_data_size);
    host_samples_data_x = (double *)malloc(samples_data_size);
    host_samples_data_y = (double *)malloc(samples_data_size);
    CHECK(cudaMalloc((void **)&device_results_data, samples_data_size));
    CHECK(cudaMalloc((void **)&device_samples_data_x, samples_data_size));
    CHECK(cudaMalloc((void **)&device_samples_data_y, samples_data_size));

    // initialize the host data
    (void)get_delta_time();
    srand(0xCCE2021);
    for (i = 0; i < N_SAMPLES; i++)
    {
        host_results_data[i] = 0;
        host_samples_data_x[i] = ((double)rand() / RAND_MAX) * (0.5 - (-0.5)) + (-0.5);
        host_samples_data_y[i] = ((double)rand() / RAND_MAX) * (0.5 - (-0.5)) + (-0.5);
    }
    printf("The initialization of host data took %.3e seconds\n", get_delta_time());

    // copy the host data to the device memory
    (void)get_delta_time();
    CHECK(cudaMemcpy(device_results_data, host_results_data, samples_data_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_samples_data_x, host_samples_data_x, samples_data_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_samples_data_y, host_samples_data_y, samples_data_size, cudaMemcpyHostToDevice));
    printf("The transfer of %ld bytes from the host to the device took %.3e seconds\n",
           (long)(samples_data_size * 3), get_delta_time());

    // run the computational kernel
    // as an example, N_SECTORS threads are launched where each thread deals with one sector
    unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ;
    int n_samples;

    n_samples = N_SAMPLES;
    blockDimX = 1 << 8; // optimize!
    blockDimY = 1 << 0; // optimize!
    blockDimZ = 1 << 0; // do not change!
    gridDimX = 1 << 8;  // optimize!
    gridDimY = 1 << 0;  // optimize!
    gridDimZ = 1 << 0;  // do not change!

    dim3 grid(gridDimX, gridDimY, gridDimZ);
    dim3 block(blockDimX, blockDimY, blockDimZ);

    if ((gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ) != n_samples)
    {
        printf("Wrong configuration!\n");
        return 1;
    }
    (void)get_delta_time();
    ccc_samples_cuda_kernel<<<grid, block>>>(device_results_data, device_samples_data_x, device_samples_data_y, n_samples);
    CHECK(cudaDeviceSynchronize()); // wait for kernel to finish
    CHECK(cudaGetLastError());      // check for kernel errors
    printf("The CUDA kernel <<<(%d,%d,%d), (%d,%d,%d)>>> took %.3e seconds to run\n",
           gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, get_delta_time());

    // copy kernel result back to host side
    double *modified_device_results_data;

    modified_device_results_data = (double *)malloc(samples_data_size);
    CHECK(cudaMemcpy(modified_device_results_data, device_results_data, samples_data_size, cudaMemcpyDeviceToHost));
    printf("The transfer of %ld bytes from the device to the host took %.3e seconds\n",
           (long)samples_data_size, get_delta_time());

    // compute the modified sector data on the CPU
    (void)get_delta_time();
    for (i = 0; i < N_SAMPLES; i++)
        ccc_samples_cpu_kernel(host_results_data, host_samples_data_x, host_samples_data_y, n_samples, i);
    printf("The cpu kernel took %.3e seconds to run (single core)\n", get_delta_time());

    // compare
    for (i = 0; i < N_SAMPLES; i++)
        if (!((abs(modified_device_results_data[i] - host_results_data[i]) < 1e-6) ||
              ((abs(modified_device_results_data[i]) >= 1e-6) &&
               (abs((modified_device_results_data[i] - host_results_data[i]) / modified_device_results_data[i]) < 1e-6))))
        {
            printf("Expected result not found!\n");
            exit(1);
        }
    printf("All is well!\n");

    // free device global memory
    CHECK(cudaFree(device_results_data));
    CHECK(cudaFree(device_samples_data_x));
    CHECK(cudaFree(device_samples_data_y));

    // free host memory
    free(host_results_data);
    free(host_samples_data_x);
    free(host_samples_data_y);
    free(modified_device_results_data);

    // reset device
    CHECK(cudaDeviceReset());

    return 0;
}

static void ccc_samples_cpu_kernel(double *results_data, double *samples_data_x, double *samples_data_y, unsigned int n_samples, unsigned int point)
{
    for (int k = 0; k < n_samples; k++)
        results_data[point] += (samples_data_x[k] * samples_data_y[(point + k) % n_samples]);
}

__global__ static void ccc_samples_cuda_kernel(double *__restrict__ results_data, double *__restrict__ samples_data_x, double *__restrict__ samples_data_y,
                                               unsigned int n_samples)
{
    unsigned int x, y, idx, point;

    // compute the thread number
    x = (unsigned int)threadIdx.x + (unsigned int)blockDim.x * (unsigned int)blockIdx.x;
    y = (unsigned int)threadIdx.y + (unsigned int)blockDim.y * (unsigned int)blockIdx.y;
    idx = (unsigned int)blockDim.x * (unsigned int)gridDim.x * y + x;
    if (idx >= n_samples)
        return; // safety precaution

    point = idx;

    for (int k = 0; k < n_samples; k++)
        results_data[point] += (samples_data_x[k] * samples_data_y[(point + k) % n_samples]);
}

static double get_delta_time(void)
{
    static struct timespec t0, t1;

    t0 = t1;
    if (clock_gettime(CLOCK_MONOTONIC, &t1) != 0)
    {
        perror("clock_gettime");
        exit(1);
    }
    return (double)(t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
}
