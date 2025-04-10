// adam.cpp
#include <torch/extension.h>

// Declare the CUDA function
void adam_cuda(torch::Tensor weights, torch::Tensor grads, float lr, torch::Tensor prev_mom, float beta);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adam", &adam_cuda, "Custom Adam update (CUDA)");
}
