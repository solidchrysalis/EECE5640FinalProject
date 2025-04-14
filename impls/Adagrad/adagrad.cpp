// adagrad.cpp
#include <torch/extension.h>
#include <pybind11/pybind11.h>

// Declare the CUDA function
void adagrad_cuda(torch::Tensor var, torch::Tensor grads, torch::Tensor prev_grads, float epsilon, float lr);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adagrad", &adagrad_cuda, "Custom adagrad update (CUDA)");
}
