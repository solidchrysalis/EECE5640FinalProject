#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void stochastic_kernel(const float* grad, float* var, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        var[idx] = grad[idx] * -lr;
    }
}

void stochastic_cuda(torch::Tensor weights, torch::Tensor grads, float lr) {
    int n = weights.size(0);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    stochastic_kernel<<<blocks, threads>>>(
        weights.data_ptr<float>(),
        grads.data_ptr<float>(),
        lr,
        n
    );
}