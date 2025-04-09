// adam.cpp
#include <torch/extension.h>

// Declare the CUDA function
void adam_cuda(torch::Tensor weights, torch::Tensor grads, float lr);

// Python-visible function
void adam(torch::Tensor weights, torch::Tensor grads, float lr) {
    adam_cuda(weights, grads, lr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adam", &adam, "Custom Adam update (CUDA)");
}
