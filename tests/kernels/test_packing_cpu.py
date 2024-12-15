import pytest
import torch
import ternify.tnn.functional as TF

def test_pack_unpack_cpu_random():
    # Generate random input tensor
    A = torch.randint(0, 4, (2, 8)).type(torch.int8)

    # Apply the packing function
    B = TF.pack2b(A)

    # Unpack the packed tensor
    A_hat = TF.unpack2b(B)

    # Flatten tensors for easier element-wise comparison
    A_flat = A.flatten()
    A_hat_flat = A_hat.flatten()

    # Verify that each element matches after packing and unpacking
    assert A_flat.shape == A_hat_flat.shape, "Shape mismatch after unpacking"
    for i in range(A_flat.shape[0]):
        assert A_flat[i].item() == A_hat_flat[i].item(), f"Mismatch at index {i}: {A_flat[i].item()} != {A_hat_flat[i].item()}"

def test_pack_unpack_cpu_all_zeros():
    # Generate an input tensor of all zeros
    A = torch.zeros((2, 8), dtype=torch.int8)

    # Apply the packing function
    B = TF.pack2b(A)

    # Unpack the packed tensor
    A_hat = TF.unpack2b(B)

    # Verify that the original and unpacked tensors are equal
    assert torch.equal(A, A_hat), "Unpacking failed for all zeros tensor"

def test_pack_unpack_cpu_all_ones():
    # Generate an input tensor of all ones
    A = torch.ones((2, 8), dtype=torch.int8) * 1

    # Apply the packing function
    B = TF.pack2b(A)

    # Unpack the packed tensor
    A_hat = TF.unpack2b(B)

    # Verify that the original and unpacked tensors are equal
    assert torch.equal(A, A_hat), "Unpacking failed for all ones tensor"

def test_pack_unpack_cpu_large_tensor():
    # Generate a large random input tensor
    A = torch.randint(0, 4, (100, 100)).type(torch.int8)

    # Apply the packing function
    B = TF.pack2b(A)

    # Unpack the packed tensor
    A_hat = TF.unpack2b(B)

    # Verify that the original and unpacked tensors are equal
    assert torch.equal(A, A_hat), "Unpacking failed for large tensor"

def test_pack_unpack_cpu_edge_values():
    # Generate an input tensor with edge values (0 and 3)
    A = torch.tensor([[0, 3, 0, 3], [3, 0, 3, 0]], dtype=torch.int8)

    # Apply the packing function
    B = TF.pack2b(A)

    # Unpack the packed tensor
    A_hat = TF.unpack2b(B)

    # Verify that the original and unpacked tensors are equal
    assert torch.equal(A, A_hat), "Unpacking failed for tensor with edge values"

if __name__ == "__main__":
    pytest.main([__file__])
