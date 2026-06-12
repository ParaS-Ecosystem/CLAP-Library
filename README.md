# CLAP Library

The mathematical libraries of the ParaS ecosystem form a critical cornerstone of High-Performance Computing (HPC) and Artificial Intelligence (AI) workloads. At the center of this effort is the **CLAP (Cross-platform Linear-algebra And PDE Library)** library, which provides a unified interface for numerical computing across multiple hardware architectures.

CLAP delivers optimized implementations of BLAS, LAPACK, FFT, and PDE kernels while abstracting backend-specific details behind a common API. Unlike traditional approaches that require separate vendor-specific libraries, CLAP enables applications to use a single programming interface while transparently leveraging the most appropriate backend. By consolidating numerical kernels under one framework, CLAP simplifies application development, improves portability, reduces vendor lock-in, and enables efficient execution across CPUs and GPUs.

---

## ⚙️ Prerequisites

Ensure the following dependencies are available before building **CLAP**.

Currently, **CLAP** supports both CPU and GPU architectures, providing a unified interface across multiple BLAS backends. CLAP leverages OpenBLAS for CPU execution, while AMD rocBLAS and NVIDIA cuBLAS enable accelerated execution on GPU platforms.

### Required Software

- **ParaS Compiler** — Version **v0.5** or later
- **CMake** — Version **3.15** or later

### Supported Backends

#### CPU Backend

- **OpenBLAS**
  - Recommended Version: **Latest Stable Release (0.3.30 or newer)**

#### GPU Backends

- **AMD rocBLAS**
  - Recommended Version: **7.1 or newer**

- **NVIDIA cuBLAS**
  - Recommended Version: **CUDA Toolkit 12.2 or newer**

---

# 🏗️ Build, Install, and Run CLAP

Follow the steps below to configure, build, install, and run the CLAP library.

---

## 🧩 1️⃣ Environment Setup

Set the compiler paths before configuring CLAP.

```bash
# Set Compiler Paths
export CC=/path/to/bin/clang
export CXX=/path/to/bin/parascc
```

---

## ⚙️ 2️⃣ Configure CLAP with CMake

Run the following command from the CLAP project root:

```bash
cmake -S . -B build \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX=/path/to/install \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_BUILD_TYPE=Release
```

---

## ⚒️ 3️⃣ Build and Install CLAP

```bash
cd build

make -j$(nproc)

make install
```

---

## 📦 4️⃣ Update Library Paths

After installation, update the CLAP library path:

```bash
export LD_LIBRARY_PATH=/path/to/install/lib64:$LD_LIBRARY_PATH
```

### OpenBLAS Library Path

```bash
export LD_LIBRARY_PATH=/path/to/openblas/lib:$LD_LIBRARY_PATH
```

### AMD ROCm Library Path

```bash
export LD_LIBRARY_PATH=/path/to/rocm/lib64:$LD_LIBRARY_PATH
```

### NVIDIA CUDA Library Path

```bash
export LD_LIBRARY_PATH=/path/to/cuda/lib64:$LD_LIBRARY_PATH
```

---

## 🚀 5️⃣ Compile an Example

Navigate to the examples directory and compile a sample application:

```bash
cd examples

parascc <filename>.cpp \
    -I/path/to/install/include \
    -L/path/to/install/lib64 \
    -lclap
```

This generates the executable:

```bash
./a.out
```

---

## 🚀 6️⃣ Select Backend

CLAP supports backend selection through command-line arguments.

### OpenBLAS Backend (CPU)

```bash
./a.out -cpu
```

### AMD rocBLAS Backend (GPU)

```bash
./a.out -rocm
```

### NVIDIA cuBLAS Backend (GPU)

```bash
./a.out -cuda
```

---

## 📋 Backend Compatibility Summary

| Backend | Recommended Version |
|----------|---------------------|
| OpenBLAS | Latest Stable Release (0.3.30+) |
| AMD rocBLAS | 7.1+ |
| NVIDIA cuBLAS | CUDA Toolkit 12.2+ |

---

## 📝 Notes

- CLAP provides a unified BLAS interface across CPU and GPU architectures.
- Backend selection is performed at runtime using command-line arguments.
- The same executable can be executed with different backends without recompilation.
- Ensure the required backend libraries are installed and available in the system library path.
- OpenBLAS is recommended for CPU execution.
- rocBLAS is recommended for AMD GPU execution.
- cuBLAS is recommended for NVIDIA GPU execution.

## Reporting Issues

For reporting issues, it can be done using either of the following ways: 
+ Raising an issue on the github repository under issues section
+ Contact us at *paras@cdac.in*


