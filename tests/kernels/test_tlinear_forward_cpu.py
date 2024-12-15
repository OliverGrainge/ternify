import pytest
import torch
import torch.nn.functional as F 
import ternify.tnn.functional as TF

def test_pack_tlinear_forward_cpu():
    X = torch.randint(-128, 128, (7, 24)).type(torch.int8)
    W = torch.randint(-1, 2, (2, 24)).type(torch.int8)
    W_packed = TF.pack2b(W+1)
    bias = torch.zeros(2, dtype=torch.int8)
    Y = TF.tlinear_forward(X, W_packed, bias=bias)
    assert Y is not None


def test_pack_tlinear_forward_dtype_cpu():
    X = torch.randint(-128, 128, (7, 24)).type(torch.int8)
    W = torch.randint(-1, 2, (2, 24)).type(torch.int8)
    W_packed = TF.pack2b(W+1)
    bias = torch.zeros(2, dtype=torch.int8)
    Y = TF.tlinear_forward(X, W_packed, bias=bias)
    assert Y.dtype == torch.int32



def test_pack_tlinear_forward_shape_cpu():
    X = torch.randint(-128, 128, (7, 12)).type(torch.int8)
    W = torch.randint(-1, 2, (3, 12)).type(torch.int8)
    W_packed = TF.pack2b(W+1)
    bias = torch.zeros(3, dtype=torch.int8)
    Y = TF.tlinear_forward(X, W_packed, bias=bias)
    assert Y.shape == (7, 3)


def test_pack_tlinear_forward_v_torch_without_bias_cpu():
    X = torch.randint(-128, 128, (7, 12)).type(torch.int8)
    W = torch.randint(-1, 2, (3, 12)).type(torch.int8)
    W_packed = TF.pack2b(W+1)
    bias = torch.zeros(3, dtype=torch.int32)
    Y = TF.tlinear_forward(X, W_packed, bias=bias)
    Y_ref = F.linear(X.float(), W.float(), bias=bias.float())
    assert torch.allclose(Y.float(), Y_ref.float())



def test_pack_tlinear_forward_v_torch_with_bias_cpu():
    X = torch.randint(-128, 128, (7, 12)).type(torch.int8)
    W = torch.randint(-1, 2, (3, 12)).type(torch.int8)
    W_packed = TF.pack2b(W+1)
    bias = torch.ones(3, dtype=torch.int32)
    Y = TF.tlinear_forward(X, W_packed, bias=bias)
    Y_ref = F.linear(X.float(), W.float(), bias=bias.float())
    assert torch.allclose(Y.float(), Y_ref.float())


def test_pack_tlinear_forward_v_torch_with_bias_cpu():
    X = torch.randint(-128, 128, (7, 12)).type(torch.int8)
    W = torch.randint(-1, 2, (3, 12)).type(torch.int8)
    W_packed = TF.pack2b(W+1)
    bias = torch.randint(-128, 128, size=(3,), dtype=torch.int32)
    Y = TF.tlinear_forward(X, W_packed, bias=bias)
    Y_ref = F.linear(X.float(), W.float(), bias=bias.float())
    assert torch.allclose(Y.float(), Y_ref.float())


def test_pack_tlinear_forward_v_torch_with_bias_large_cpu():
    X = torch.randint(-128, 128, (3, 28)).type(torch.int8)
    W = torch.randint(-1, 2, (7, 28)).type(torch.int8)
    W_packed = TF.pack2b(W+1)
    bias = torch.randint(-128, 128, size=(7,), dtype=torch.int32)
    Y = TF.tlinear_forward(X, W_packed, bias=bias)
    Y_ref = F.linear(X.float(), W.float(), bias=bias.float())
    assert torch.allclose(Y.float(), Y_ref.float())