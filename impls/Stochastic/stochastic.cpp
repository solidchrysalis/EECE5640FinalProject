// stochastic.cpp
#include <torch/extension.h>

// Declare the CUDA function
void stochastic_cuda(torch::Tensor weights, torch::Tensor grads, float lr);

// Python-visible function
void stochastic(torch::Tensor weights, torch::Tensor grads, float lr) {
    stochastic_cuda(weights, grads, lr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stochastic", &stochastic, "Custom stochastic update (CUDA)");
}
