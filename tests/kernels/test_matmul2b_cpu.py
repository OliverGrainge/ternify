import torch
import os
import sys
import pytest

try: 
    import ternify.tnn.functional as TF 
except ImportError as e:
    print(f"Failed to import naive_matmul module: {e}")
    print(f"sys.path: {sys.path[-1]}")
    print(f"Files in sys.path[-1]: {os.listdir(sys.path[-1])}")


# Original test cases
def test_matmul2b_cpu_minusone():
    # Test with basic non-batched inputs (all ones)
    batch_size = 1 
    in_channels = 4 
    out_channels = 12 

    X = torch.ones(batch_size, in_channels, dtype=torch.int8) 
    W = torch.zeros(out_channels, in_channels, dtype=torch.int8) # 0 gets mapped to -1

    W_packed = TF.pack2b_cpu(W.t())
    C = TF.matmul2b_cpu(X, W_packed)
    for val in C.flatten(): 
        assert val == -in_channels


def test_matmul2b_cpu_zeros():
    # Test with basic non-batched inputs (all ones)
    batch_size = 1 
    in_channels = 4 
    out_channels = 12 

    X = torch.ones(batch_size, in_channels, dtype=torch.int8) 
    W = torch.ones(out_channels, in_channels, dtype=torch.int8) # 0 gets mapped to -1

    W_packed = TF.pack2b_cpu(W.t())
    C = TF.matmul2b_cpu(X, W_packed)
    for val in C.flatten(): 
        assert val == -0


def test_matmul2b_cpu_ones():
    # Test with basic non-batched inputs (all ones)
    batch_size = 1 
    in_channels = 4 
    out_channels = 12 

    X = torch.ones(batch_size, in_channels, dtype=torch.int8) 
    W = torch.ones(out_channels, in_channels, dtype=torch.int8) + 1 # 0 gets mapped to -1

    W_packed = TF.pack2b_cpu(W.t())
    C = TF.matmul2b_cpu(X, W_packed)
    for val in C.flatten(): 
        assert val == in_channels


def test_matmul2b_cpu_random1():
    # Test with basic non-batched inputs (all ones)
    batch_size = 4 
    in_channels = 4 
    out_channels = 4

    X = torch.randint(-128, 127, (batch_size, in_channels), dtype=torch.int8) 
    W = torch.randint(-1, 2, (out_channels, in_channels), dtype=torch.int8) # 0 gets mapped to -1

    C = torch.matmul(X, W.t())
    W_packed = TF.pack2b_cpu((W+1).t().contiguous())
    C_hat = TF.matmul2b_cpu(X, W_packed)
    
    assert torch.allclose(C, C_hat)


def test_matmul2b_cpu_random2():
    # Test with basic non-batched inputs (all ones)
    batch_size = 4 
    in_channels = 4 
    out_channels = 8

    X = torch.randint(-128, 127, (batch_size, in_channels), dtype=torch.int8) 
    W = torch.randint(-1, 2, (out_channels, in_channels), dtype=torch.int8) # 0 gets mapped to -1

    C = torch.matmul(X, W.t())
    W_packed = TF.pack2b_cpu((W+1).t().contiguous())
    C_hat = TF.matmul2b_cpu(X, W_packed)
    
    assert torch.allclose(C, C_hat)


def test_matmul2b_cpu_random2():
    # Test with basic non-batched inputs (all ones)
    batch_size = 12 
    in_channels = 4 
    out_channels = 8

    X = torch.randint(-128, 127, (batch_size, in_channels), dtype=torch.int8) 
    W = torch.randint(-1, 2, (out_channels, in_channels), dtype=torch.int8) # 0 gets mapped to -1

    C = torch.matmul(X, W.t())
    W_packed = TF.pack2b_cpu((W+1).t().contiguous())
    C_hat = TF.matmul2b_cpu(X, W_packed)
    
    assert torch.allclose(C, C_hat)


def test_matmul2b_cpu_random3():
    # Test with basic non-batched inputs (all ones)
    batch_size = 12 
    in_channels = 4 
    out_channels = 100

    X = torch.randint(-128, 127, (batch_size, in_channels), dtype=torch.int8) 
    W = torch.randint(-1, 2, (out_channels, in_channels), dtype=torch.int8) # 0 gets mapped to -1

    C = torch.matmul(X, W.t())
    W_packed = TF.pack2b_cpu((W+1).t().contiguous())
    C_hat = TF.matmul2b_cpu(X, W_packed)
    
    assert torch.allclose(C, C_hat)


def test_matmul2b_cpu_shape():
    # Test with basic non-batched inputs (all ones)
    batch_size = 12 
    in_channels = 4 
    out_channels = 100

    X = torch.randint(-128, 127, (batch_size, in_channels), dtype=torch.int8) 
    W = torch.randint(-1, 2, (out_channels, in_channels), dtype=torch.int8) # 0 gets mapped to -1

    W_packed = TF.pack2b_cpu((W+1).t().contiguous())
    C_hat = TF.matmul2b_cpu(X, W_packed)
    
    assert C_hat.shape == (batch_size, out_channels)

def test_matmul2b_cpu_feedforward_random1():
    # Test with basic non-batched inputs (all ones)
    batch_size = 12 
    sequence_length = 1024
    in_channels = 4 
    out_channels = 100

    X = torch.randint(-128, 127, (batch_size, sequence_length, in_channels), dtype=torch.int8) 
    W = torch.randint(-1, 2, (out_channels, in_channels), dtype=torch.int8) # 0 gets mapped to -1

    C = torch.matmul(X, W.t())
    W_packed = TF.pack2b_cpu((W+1).t().contiguous())
    C_hat = TF.matmul2b_cpu(X, W_packed)
    assert torch.allclose(C, C_hat)


def test_matmul2b_cpu_feedforward_shape():
    # Test with basic non-batched inputs (all ones)
    batch_size = 12 
    sequence_length = 1024
    in_channels = 4 
    out_channels = 100

    X = torch.randint(-128, 127, (batch_size, sequence_length, in_channels), dtype=torch.int8) 
    W = torch.randint(-1, 2, (out_channels, in_channels), dtype=torch.int8) # 0 gets mapped to -1

    W_packed = TF.pack2b_cpu((W+1).t().contiguous())
    C_hat = TF.matmul2b_cpu(X, W_packed)
    
    assert C_hat.shape == (batch_size, sequence_length, out_channels)



def test_matmul2b_cpu_feedforward_shape():
    # Test with basic non-batched inputs (all ones)
    batch_size = 12 
    sequence_length = 16
    in_channels = 4 
    out_channels = 100

    X = torch.randint(-128, 127, (batch_size, sequence_length, in_channels), dtype=torch.int8) 
    W = torch.randint(-1, 2, (out_channels, in_channels), dtype=torch.int8) # 0 gets mapped to -1

    C = torch.matmul(X, W.t())
    W_packed = TF.pack2b_cpu((W+1).t().contiguous())
    C_hat = TF.matmul2b_cpu(X, W_packed)
    
    assert C_hat.shape == C.shape



