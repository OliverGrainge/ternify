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

    W_packed = TF.pack2b_cpu(W)
    C = TF.matmul2b_cpu(X, W_packed.t())
    for val in C.flatten(): 
        assert val == -in_channels

def test_matmul2b_cpu_zeros():
    # Test with basic non-batched inputs (all ones)
    batch_size = 1 
    in_channels = 4 
    out_channels = 12 

    X = torch.ones(batch_size, in_channels, dtype=torch.int8) 
    W = torch.ones(out_channels, in_channels, dtype=torch.int8) # 1 gets mapped to 0

    W_packed = TF.pack2b_cpu(W)
    C = TF.matmul2b_cpu(X, W_packed.t())
    for val in C.flatten(): 
        assert val == 0

def test_matmul2b_cpu_ones():
    # Test with basic non-batched inputs (all ones)
    batch_size = 1 
    in_channels = 4 
    out_channels = 12 

    X = torch.ones(batch_size, in_channels, dtype=torch.int8) 
    W = torch.ones(out_channels, in_channels, dtype=torch.int8) + 1 # 2 gets mapped to 1

    W_packed = TF.pack2b_cpu(W)
    C = TF.matmul2b_cpu(X, W_packed.t())
    for val in C.flatten(): 
        assert val == in_channels

def test_matmul2b_cpu_minusone2():
    # Test with basic non-batched inputs (all ones)
    batch_size = 1 
    in_channels = 8 
    out_channels = 24

    X = torch.ones(batch_size, in_channels, dtype=torch.int8) 
    W = torch.zeros(out_channels, in_channels, dtype=torch.int8) # 0 gets mapped to -1

    W_packed = TF.pack2b_cpu(W)
    C = TF.matmul2b_cpu(X, W_packed.t())
    for val in C.flatten(): 
        assert val == -in_channels

def test_matmul2b_cpu_zeros2():
    # Test with basic non-batched inputs (all ones)
    batch_size = 1 
    in_channels = 8 
    out_channels = 24 

    X = torch.ones(batch_size, in_channels, dtype=torch.int8) 
    W = torch.ones(out_channels, in_channels, dtype=torch.int8) # 1 gets mapped to 0

    W_packed = TF.pack2b_cpu(W)
    C = TF.matmul2b_cpu(X, W_packed.t())
    for val in C.flatten(): 
        assert val == 0

def test_matmul2b_cpu_ones2():
    # Test with basic non-batched inputs (all ones)
    batch_size = 1 
    in_channels = 8
    out_channels = 24 

    X = torch.ones(batch_size, in_channels, dtype=torch.int8) 
    W = torch.ones(out_channels, in_channels, dtype=torch.int8) + 1 # 2 gets mapped to 1

    W_packed = TF.pack2b_cpu(W)
    C = TF.matmul2b_cpu(X, W_packed.t())
    for val in C.flatten(): 
        assert val == in_channels


# Additional test cases
def test_matmul2b_cpu_random_values():
    # Test with random input values
    batch_size = 1
    in_channels = 8
    out_channels = 16

    X = torch.randint(0, 3, (batch_size, in_channels), dtype=torch.int8)
    W = torch.randint(0, 3, (out_channels, in_channels), dtype=torch.int8)

    W_packed = TF.pack2b_cpu(W)
    C = TF.matmul2b_cpu(X, W_packed.t())
    assert C.shape == (batch_size, out_channels)

def test_matmul2b_cpu_large_inputs():
    # Test with larger inputs
    batch_size = 16
    in_channels = 128
    out_channels = 256

    X = torch.randint(0, 3, (batch_size, in_channels), dtype=torch.int8)
    W = torch.randint(0, 3, (out_channels, in_channels), dtype=torch.int8)

    W_packed = TF.pack2b_cpu(W)
    C = TF.matmul2b_cpu(X, W_packed.t())
    assert C.shape == (batch_size, out_channels)

def test_matmul2b_cpu_batch_processing():
    # Test with batched inputs
    batch_size = 4
    in_channels = 8
    out_channels = 16

    X = torch.randint(0, 3, (batch_size, in_channels), dtype=torch.int8)
    W = torch.randint(0, 3, (out_channels, in_channels), dtype=torch.int8)

    W_packed = TF.pack2b_cpu(W)
    C = TF.matmul2b_cpu(X, W_packed.t())
    assert C.shape == (batch_size, out_channels)

def test_matmul2b_cpu_edge_case_zeros():
    # Test with all zeros
    batch_size = 2
    in_channels = 8
    out_channels = 8

    X = torch.zeros(batch_size, in_channels, dtype=torch.int8)
    W = torch.zeros(out_channels, in_channels, dtype=torch.int8)

    W_packed = TF.pack2b_cpu(W)
    C = TF.matmul2b_cpu(X, W_packed.t())
    for val in C.flatten():
        assert val == 0

def test_matmul2b_cpu_mixed_values():
    # Test with mixed values
    batch_size = 2
    in_channels = 4
    out_channels = 6

    X = torch.tensor([[1, 0, 2, 0], [0, 1, 0, 2]], dtype=torch.int8)
    W = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 1], [1, 2, 0, 1], [0, 1, 2, 1], [2, 2, 1, 0], [1, 0, 1, 2]], dtype=torch.int8)

    W_packed = TF.pack2b_cpu(W)
    C = TF.matmul2b_cpu(X, W_packed.t())
    assert C.shape == (batch_size, out_channels)



def test_matmul2b_cpu_feedforward_shape():
    # Test with all zeros
    batch_size = 2
    sequence_length = 4
    in_channels = 8
    out_channels = 8

    X = torch.zeros(batch_size, sequence_length, in_channels, dtype=torch.int8)
    W = torch.zeros(out_channels, in_channels, dtype=torch.int8)

    W_packed = TF.pack2b_cpu(W)
    C = TF.matmul2b_cpu(X, W_packed.t())
    assert C.shape == (batch_size, sequence_length, out_channels)




def test_matmul2b_cpu_feedforward_minus1():
    # Test with all zeros
    batch_size = 2
    sequence_length = 4
    in_channels = 8
    out_channels = 8

    X = torch.ones(batch_size, sequence_length, in_channels, dtype=torch.int8)
    W = torch.zeros(out_channels, in_channels, dtype=torch.int8)

    W_packed = TF.pack2b_cpu(W)
    C = TF.matmul2b_cpu(X, W_packed.t())
    for val in C.flatten(): 
        assert val == -in_channels


def test_matmul2b_cpu_feedforward_zeros():
    # Test with all zeros
    batch_size = 2
    sequence_length = 4
    in_channels = 8
    out_channels = 8

    X = torch.ones(batch_size, sequence_length, in_channels, dtype=torch.int8)
    W = torch.ones(out_channels, in_channels, dtype=torch.int8)

    W_packed = TF.pack2b_cpu(W)
    C = TF.matmul2b_cpu(X, W_packed.t())
    for val in C.flatten(): 
        assert val == 0



def test_matmul2b_cpu_feedforward_ones():
    # Test with all zeros
    batch_size = 2
    sequence_length = 4
    in_channels = 8
    out_channels = 8

    X = torch.ones(batch_size, sequence_length, in_channels, dtype=torch.int8)
    W = torch.ones(out_channels, in_channels, dtype=torch.int8) + 1

    W_packed = TF.pack2b_cpu(W)
    C = TF.matmul2b_cpu(X, W_packed.t())
    for val in C.flatten(): 
        print(val)
        assert val == in_channels
