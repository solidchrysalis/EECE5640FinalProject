// adam.cpp
#include <torch/extension.h>

// Declare the CUDA function
void adam_cuda(torch::Tensor weights, torch::Tensor grads, torch::Tensor prev_mean, torch::Tensor prev_variance, float beta_1, float beta_2, float epoch, float epsilon, float lr);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adam", &adam_cuda, "Custom Adam update (CUDA)");
}
