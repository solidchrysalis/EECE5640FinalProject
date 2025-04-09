// adagrad.cpp
#include <torch/extension.h>

// Declare the CUDA function
void adagrad_cuda(torch::Tensor weights, torch::Tensor grads, float lr);

// Python-visible function
void adagrad(torch::Tensor weights, torch::Tensor grads, float lr) {
    adagrad_cuda(weights, grads, lr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adagrad", &adagrad, "Custom adagrad update (CUDA)");
}
