#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void adagrad_kernel(float* weights, const float* grads, float* prev_grads, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float true_learn;
        true_learn = lr / (sqrtf(prev_grads[idx]) + 1e-8);  // Update cache (squared gradients)
        weights[idx] -= true_learn * grads[idx];  // Update weights
        prev_grads[idx] += grads[idx] * grads[idx];
    }
}

void adagrad_cuda(torch::Tensor weights, torch::Tensor grads, torch::Tensor prev_grads, float lr) {
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(grads.is_contiguous(), "grads must be contiguous");
    TORCH_CHECK(weights.sizes() == grads.sizes(), "weights and grads must be same size");
    
    int n = weights.size(0);
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    adagrad_kernel<<<blocks, threads>>>(
        weights.data_ptr<float>(),
        grads.data_ptr<float>(),
        prev_grads.data_ptr<float>(),
        lr,
        n
    );

    // Optionally: sync to catch errors
    cudaDeviceSynchronize();
}
