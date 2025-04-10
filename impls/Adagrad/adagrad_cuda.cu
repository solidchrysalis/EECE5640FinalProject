#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void adagrad_kernel(float* weights, const float* grads, float* cache, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        cache[idx] += grads[idx] * grads[idx];  // Update cache (squared gradients)
        weights[idx] -= lr * grads[idx] / (sqrtf(cache[idx]) + 1e-7);  // Update weights
    }
}

void adagrad_cuda(torch::Tensor weights, torch::Tensor grads, float lr) {
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(grads.is_contiguous(), "grads must be contiguous");
    TORCH_CHECK(weights.sizes() == grads.sizes(), "weights and grads must be same size");
    
    int n = weights.size(0);
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Create a tensor for the cache (squared gradients)
    torch::Tensor cache = torch::zeros_like(weights, torch::dtype(torch::kFloat32).device(weights.device()));

    adagrad_kernel<<<blocks, threads>>>(
        weights.data_ptr<float>(),
        grads.data_ptr<float>(),
        cache.data_ptr<float>(),
        lr,
        n
    );

    // Optionally: sync to catch errors
    cudaDeviceSynchronize();
}
