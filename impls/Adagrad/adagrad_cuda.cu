#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void adagrad_kernel(float* var, const float* grads, float* prev_grads, float epsilon, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad = grads[idx];
        prev_grads[idx] += grad * grad;  // Accumulate squared gradients
        float denom = sqrtf(prev_grads[idx] + epsilon);
        var[idx] -= (lr / denom) * grad;  // Update variable
    }
}

void adagrad_cuda(torch::Tensor var, torch::Tensor grads, torch::Tensor prev_grads, float epsilon, float lr) {
    TORCH_CHECK(var.is_cuda(), "var must be a CUDA tensor");
    TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
    TORCH_CHECK(var.is_contiguous(), "var must be contiguous");
    TORCH_CHECK(grads.is_contiguous(), "grads must be contiguous");
    TORCH_CHECK(var.sizes() == grads.sizes(), "var and grads must be same size");
    
    int n = var.size(0);
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    adagrad_kernel<<<blocks, threads>>>(
        var.data_ptr<float>(),
        grads.data_ptr<float>(),
        prev_grads.data_ptr<float>(),
        epsilon,
        lr,
        n
    );

    // Sync to catch errors
    cudaDeviceSynchronize();
}
