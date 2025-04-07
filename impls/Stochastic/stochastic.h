#ifndef SGD_OP_H_
#define SGD_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

void LaunchSGD(const float* grad, float* var, float lr, int size, const Eigen::GpuDevice& d);

#endif  