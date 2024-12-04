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


    
