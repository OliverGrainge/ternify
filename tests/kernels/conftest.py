import os
import sys

def pytest_configure():
    kernel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    sys.path.append(kernel_path)