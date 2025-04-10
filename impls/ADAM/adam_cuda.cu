#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void adam_kernel(const float* grad, float* var, const float* prev_mom, float beta, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float curr_mom = 0f;

    if (idx < size) {
        curr_mom = beta * prev_mom[i] + (1 - beta) * grad[idx];
        var[idx] = curr_mom * lr;
    }
}

void adam_cuda(torch::Tensor weights, torch::Tensor grads, torch::Tensor prev_mom, float beta, float lr) {
    // Checks for correct types
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(grads.is_contiguous(), "grads must be contiguous");
    TORCH_CHECK(weights.sizes() == grads.sizes(), "weights and grads must be same size");s

    int n = weights.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    adam_kernel<<<blocks, threads>>>(
        grads.data_ptr<float>(),
        weights.data_ptr<float>(),
        prev_mom.data_ptr<float>(),
        beta,
        lr,
        n
    );
    
    cudaDeviceSynchronize();
}