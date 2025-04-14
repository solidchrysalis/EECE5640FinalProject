#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void adam_kernel(float* var, const float* grad, float* prev_mean, float* prev_variance, float beta_1, float beta_2, float epoch, float epsilon, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float curr_mean = 0.0;
    float curr_variance = 0.0;
    float corrected_mean = 0.0;
    float corrected_variance = 0.0;

    if (idx < size) {
        curr_mean = (beta_1 * prev_mean[idx]) + ((1 - beta_1) * grad[idx]);
        curr_variance = (beta_2 * prev_variance[idx]) + ((1 - beta_2) * grad[idx] * grad[idx]);
        corrected_mean = curr_mean / (1 - (powf(beta_1, epoch)));
        corrected_variance = curr_variance / (1 - (powf(beta_2, epoch)));
        var[idx] -= ((corrected_mean / (sqrtf(corrected_variance + epsilon))) * lr);
        prev_mean[idx] = curr_mean;
        prev_variance[idx] = curr_variance;
    }
}

void adam_cuda(torch::Tensor weights, torch::Tensor grads, torch::Tensor prev_mean, torch::Tensor prev_variance, float beta_1, float beta_2, float epoch, float epsilon, float lr) {
    // Checks for correct types
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(grads.is_contiguous(), "grads must be contiguous");
    TORCH_CHECK(weights.sizes() == grads.sizes(), "weights and grads must be same size");

    int n = weights.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    adam_kernel<<<blocks, threads>>>(
        weights.data_ptr<float>(),
        grads.data_ptr<float>(),
        prev_mean.data_ptr<float>(),
        prev_variance.data_ptr<float>(),
        beta_1,
        beta_2,
        epoch,
        epsilon,
        lr,
        n
    );
    
    cudaDeviceSynchronize();
}