#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "sgd_op.h"

using namespace tensorflow;

REGISTER_OP("SGD")
    .Input("var: float32")
    .Input("grad: float32")
    .Attr("lr: float")
    .Output("out: float32");

class MySGDOp : public OpKernel {
public:
    explicit MySGDOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("lr", &lr_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& var = context->input(0);
        const Tensor& grad = context->input(1);

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, var.shape(), &output));

        auto d = context->eigen_device<Eigen::GpuDevice>();
        LaunchMySGD(var.flat<float>().data(), grad.flat<float>().data(), lr_, var.NumElements(), d);

        cudaMemcpy(output->flat<float>().data(), var.flat<float>().data(),
                   var.NumElements() * sizeof(float), cudaMemcpyDeviceToDevice);
    }

private:
    float lr_;
};

REGISTER_KERNEL_BUILDER(Name("SGD").Device(DEVICE_GPU), MySGDOp);
