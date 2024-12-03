import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Function

# add the kernel build path
script_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.join(script_dir, 'kernels/kernels_build')
sys.path.append(build_path)

try:
    import naive_matmul
except ImportError as e:
    print(f"Failed to import naive_matmul module: {e}")
    sys.exit(1)

class NaiveMatmulFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # Save input and weight for backward computation
        ctx.save_for_backward(input, weight, bias)
        output = naive_matmul.naive_matmul(input, weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # Compute gradient with respect to input
        if ctx.needs_input_grad[0]:
            grad_input = naive_matmul.naive_matmul(grad_output, weight)

        # Compute gradient with respect to weight
        if ctx.needs_input_grad[1]:
            grad_weight = naive_matmul.naive_matmul(grad_output.t(), input)

        # Compute gradient with respect to bias
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class NaiveLinear(nn.Module): 
    def __init__(self, in_features, out_features, bias=True):
        super(NaiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias: 
            self.bias = nn.Parameter(torch.randn(out_features))
        else: 
            self.register_parameter('bias', None)

    def forward(self, input): 
        return NaiveMatmulFunction.apply(input, self.weight, self.bias)

if __name__ == "__main__":
    naive_layer = NaiveLinear(20, 10, bias=True)
    input = torch.randn(2, 20, requires_grad=True)
    output = naive_layer(input)
    print(output.shape)

    # Perform a backward pass
    output.sum().backward()
    print("Gradient w.r.t input:", input.grad.shape)
    print("Gradient w.r.t weight:", naive_layer.weight.grad.shape)
    if naive_layer.bias is not None:
        print("Gradient w.r.t bias:", naive_layer.bias.grad.shape)
