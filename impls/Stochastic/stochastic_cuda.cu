#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void stochastic_kernel(const float* grad, float* var, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        var[idx] = grad[idx] * -lr;
    }
}

void stochastic_cuda(torch::Tensor weights, torch::Tensor grads, float lr) {
    // Checks for correct types
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(grads.is_contiguous(), "grads must be contiguous");
    TORCH_CHECK(weights.sizes() == grads.sizes(), "weights and grads must be same size");

    cudaEvent_t start, stop;
    float elapsed_time_ms1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 
  
    cudaEventRecord(start, 0);  

    int n = weights.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    stochastic_kernel<<<blocks, threads>>>(
        grads.data_ptr<float>(),
        weights.data_ptr<float>(),
        lr,
        n
    );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms1, start, stop);
    printf("Time spent - stochastic kernel: %f\n", elapsed_time_ms1);

    cudaDeviceSynchronize();
}