cmake_minimum_required(VERSION 3.10)
project(TernaryGemmBenchmark)

# On macOS, use Homebrew's LLVM/Clang
if(APPLE)
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
    set(OpenMP_C_FLAGS "-fopenmp")
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/llvm/lib/libomp.dylib")
endif()

# Use C++11.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# Add BLAS package
find_package(BLAS REQUIRED)

# Add OpenMP support
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add OpenBLAS include directory
if(APPLE)
    include_directories(/opt/homebrew/opt/openblas/include)
elseif(UNIX)
    include_directories(/usr/include/openblas)
endif()

# Include the header directory.
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set output directories for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ========================== Common executable ==========================
# Base library without NEON
add_library(ternary_gemm_base
    src/backend/common_tgemm.cpp
    src/tgemm.cpp
    src/tgemm_pack_weights.cpp
)

add_executable(benchmark
    benchmarks/benchmark_flops.cpp
)
target_link_libraries(benchmark
    PRIVATE
    ternary_gemm_base
    BLAS::BLAS
)


