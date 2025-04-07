#include "sgd_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

__global__ void SGDKernel(const float* grad, float* var, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        var[idx] = grad[idx] * -lr;
    }
}

int main() {
    
}
