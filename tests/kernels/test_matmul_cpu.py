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

def test_naive_matmul_basic():
    # Test with basic non-batched inputs (all ones)
    A = torch.ones(5, 5)
    B = torch.ones(5, 5)

    # Expected result
    expected_output = torch.matmul(A, B)

    # Call naive_matmul function
    output = TF.matmul_cpu(A, B)

    # Assert that the output matches the expected result
    assert torch.allclose(output, expected_output), "The output of naive_matmul is incorrect for basic non-batched inputs."

def test_naive_matmul_basic_batched():
    # Test with basic batched inputs (all ones)
    A = torch.ones(10, 5, 5)  # Batch size of 10
    B = torch.ones(5, 5)      # Shared B matrix across all batches

    # Expected result
    expected_output = torch.matmul(A, B)

    # Call naive_matmul function
    output = TF.matmul_cpu(A, B)

    # Assert that the output matches the expected result
    assert torch.allclose(output, expected_output), "The output of naive_matmul is incorrect for basic batched inputs."

def test_naive_matmul_different_shapes():
    # Test with different input shapes (non-batched)
    A = torch.ones(3, 4)
    B = torch.ones(4, 2)

    # Expected result
    expected_output = torch.matmul(A, B)

    # Call naive_matmul function
    output = TF.matmul_cpu(A, B)
    
    # Assert that the output matches the expected result
    assert torch.allclose(output, expected_output), "The output of naive_matmul is incorrect for different non-batched input shapes."

def test_naive_matmul_different_shapes_batched():
    # Test with different input shapes (batched)
    A = torch.ones(8, 3, 4)  # Batch size of 8
    B = torch.ones(4, 2)     # Shared B matrix across all batches

    # Expected result
    expected_output = torch.matmul(A, B)

    # Call naive_matmul function
    output = TF.matmul_cpu(A, B)

    # Assert that the output matches the expected result
    assert torch.allclose(output, expected_output), "The output of naive_matmul is incorrect for different batched input shapes."

def test_naive_matmul_negative_values():
    # Test with negative values in inputs (batched)
    A = torch.full((5, 5, 5), -1.0)  # Batch size of 5
    B = torch.full((5, 5), -1.0)

    # Expected result
    expected_output = torch.matmul(A, B)

    # Call naive_matmul function
    output = TF.matmul_cpu(A, B)

    # Assert that the output matches the expected result
    assert torch.allclose(output, expected_output), "The output of naive_matmul is incorrect for negative values in batched inputs."

def test_naive_matmul_large_values():
    # Test with large values in inputs (non-batched)
    A = torch.full((5, 5), 1e6)
    B = torch.full((5, 5), 1e6)

    # Expected result
    expected_output = torch.matmul(A, B)

    # Call naive_matmul function
    output = TF.matmul_cpu(A, B)

    # Assert that the output matches the expected result
    assert torch.allclose(output, expected_output), "The output of naive_matmul is incorrect for large values in non-batched inputs."

def test_naive_matmul_small_values():
    # Test with small values in inputs (batched)
    A = torch.full((4, 5, 5), 1e-6)  # Batch size of 4
    B = torch.full((5, 5), 1e-6)

    # Expected result
    expected_output = torch.matmul(A, B)

    # Call naive_matmul function
    output = TF.matmul_cpu(A, B)

    # Assert that the output matches the expected result
    assert torch.allclose(output, expected_output), "The output of naive_matmul is incorrect for small values in batched inputs."

def test_naive_matmul_identity():
    # Test with an identity matrix (non-batched)
    A = torch.eye(5)
    B = torch.ones(5, 5)

    # Expected result
    expected_output = torch.matmul(A, B)

    # Call naive_matmul function
    output = TF.matmul_cpu(A, B)

    # Assert that the output matches the expected result
    assert torch.allclose(output, expected_output), "The output of naive_matmul is incorrect when using an identity matrix."

def test_naive_matmul_identity_batched():
    # Test with an identity matrix (batched)
    A = torch.eye(5)[None, :, :].repeat(6, 1, 1)  # Batch size of 6
    B = torch.ones(5, 5)

    # Expected result
    expected_output = torch.matmul(A, B)

    # Call naive_matmul function
    output = TF.matmul_cpu(A, B)

    # Assert that the output matches the expected result
    assert torch.allclose(output, expected_output), "The output of naive_matmul is incorrect when using a batched identity matrix."

def test_matmul_cpu_v_torch(): 
    A = torch.randn(10, 10)
    B = torch.randn(10, 10)
    C = torch.matmul(A, B)
    C_hat = TF.matmul_cpu(A, B)
    assert torch.allclose(C, C_hat)


def test_matmul_cpu_v_torch_batched(): 
    A = torch.randn(3, 10, 10)
    B = torch.randn(10, 10)
    C = torch.matmul(A, B)
    C_hat = TF.matmul_cpu(A, B)
    assert torch.allclose(C, C_hat)

if __name__ == "__main__":
    pytest.main()
