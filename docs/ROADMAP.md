# CLAP Library Roadmap

**CLAP** — **Cross-platform Linear-algebra And PDE Library** — is a numerical library under the ParaS Ecosystem. Its goal is to provide a unified, device-agnostic interface for linear algebra, FFT, and PDE kernels across CPUs, GPUs, and future accelerators.

Repository: [https://github.com/paras-ecosystem/clap-library](https://github.com/paras-ecosystem/clap-library)

---

## Vision

CLAP aims to make scientific, engineering, and AI applications portable across diverse hardware without requiring developers to rewrite numerical code for every vendor-specific library.

The long-term vision is:

> **One numerical API, multiple optimized backends, portable execution across architectures.**

CLAP will evolve as the mathematical library layer of the ParaS Ecosystem, supporting application developers with a common interface while allowing system developers and backend contributors to optimize for different architectures.

---

## Current Status

The current repository provides the foundation for a cross-platform numerical library with:

* Unified BLAS interface for CPU and GPU backends.
* CPU backend support through OpenBLAS.
* NVIDIA GPU backend support through cuBLAS.
* AMD GPU backend support through rocBLAS.
* Runtime backend selection using command-line options such as `-cpu`, `-cuda`, and `-rocm`.
* CMake-based build and installation flow.
* Example programs for validating backend execution, compiled using the ParaS compiler.
* LGPL-3.0 licensing.

---

## Roadmap Objectives

The roadmap is organized around six major goals:

1. **Stabilize the core BLAS abstraction layer.**
2. **Expand numerical coverage from BLAS to LAPACK, FFT, Sparse, and PDE kernels.**
3. **Improve portability across CPU, GPU, and emerging accelerator platforms.**
4. **Provide performance benchmarking and correctness validation.**
5. **Make CLAP easier to build, use, package, and contribute to.**
6. **Enable adoption by HPC, AI, and scientific applications in the ParaS Ecosystem.**

---

## Phase 1: Foundation and Community-Ready Release

**Target:** First open-source baseline release

### 1.1 Repository Readiness

* Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and issue templates.
* Add clear contributor labels such as `good first issue`, `backend`, `blas`, `lapack`, `fft`, `pde`, `documentation`, and `performance`.
* Define coding style, naming conventions, API conventions, and backend contribution guidelines.
* Add Doxygen-ready comments for public APIs.
* Publish API reference documentation using GitHub Pages or similar static documentation flow.

### 1.2 Build and Packaging Improvements

* Add dependency installation notes for new backend libraries like Armpl, Intel MKL, FFTW, cuFFT, rocFFT,     Lapack, cuSolver, rocSolver, PETSc (once integrated).
* Update the CMake configuration files to integrate the newly added backend libraries into CLAP.
* Add documented build modes: Debug, Release, Developer build (with tests enabled).
* Add smoke-test instructions after installation to confirm the library loads and detects backends correctly
* Add installation validation using downstream example projects.
* Provide a container-based build option for reproducibility across machines
* Split CLAP into per-domain plugin libraries — `libclap_blas.so` (existing), `libclap_lapack.so`, `libclap_fft.so`, `libclap_pde.so` — each independently buildable via `CLAP_ENABLE_LAPACK`, `CLAP_ENABLE_FFT`, `CLAP_ENABLE_PDE`.

### 1.3 Continuous Integration

* Add GitHub Actions workflows for:

  * CMake configure/build.
  * CPU-GPU testing.
  * Static checks and formatting.
  * Documentation build validation.

* Add optional self-hosted CI jobs for CUDA and ROCm systems when infrastructure is available.
* Add CI badges in `README.md`.

### 1.4 Baseline Release

* Tag the first public release as `v2026.06`.
* Release scope:

  * Stable selected BLAS Level-1, Level-2, and Level-3 APIs.
  * OpenBLAS, cuBLAS, and rocBLAS backend dispatch.
  * Example programs.
  * Initial documentation.

---

## Phase 2: BLAS Completeness and API Stabilization

**Target:** Robust BLAS layer across supported backends

### 2.1 BLAS Coverage

The following routines are currently supported across OpenBLAS, cuBLAS, and rocBLAS backends. Additional routines will be added incrementally as the library matures.

#### BLAS Level 1

* `axpy`
* `copy`
* `dot`
* `scal`
* `swap`

#### BLAS Level 2

* `gemv`
* `ger`
* `symv`
* `trmv`
* `trsv`
* `sbmv`

#### BLAS Level 3

* `gemm`
* `symm`
* `syrk`
* `trmm`
* `trsm`

### 2.2 Data Type Support

Newly added BLAS routines will support the following data types across all validated backends:

  * `float`
  * `double`
  * `complex<float>`
  * `complex<double>`
* Define future extension path for:

  * `float16`
  * `bfloat16`
  * mixed precision routines for  AI and deep learning workloads where memory bandwidth and throughput are critical.

### 2.3 API Design

* Add explicit error handling and status codes.
* Introduce backend discovery APIs.
* Introduce backend fallback behavior when a requested backend is unavailable.
* Add thread-safety guidelines.

### 2.4 Testing

* Add correctness tests comparing backend output against reference results.
* Add tolerance rules for floating-point and mixed-precision routines.
* Add parameterized tests for matrix sizes, layouts, transposition modes, and strides.

---

## Phase 3: LAPACK and Solver Layer

**Target:** Enable scientific applications requiring dense linear solvers

## 3.1 LAPACK Interface

Provide unified APIs for the following LAPACK routine categories.

### Linear System Solvers

- `getrf`
- `getrs`
- `gesv`

### Cholesky Routines

- `potrf`
- `potrs`

### QR Factorization Routines

- `geqrf`
- `orgqr`

### Eigenvalue Routines

- `syev`
- `syevd`
- `geev`

### Singular Value Decomposition

- `gesvd`

### 3.2 Backend Support

* CPU baseline through LAPACK/OpenBLAS-compatible implementations.
* NVIDIA backend through cuSOLVER where applicable.
* AMD backend through rocSOLVER where applicable.
* Define fallback paths when full GPU solver support is unavailable.

### 3.3 Application Validation

Validate LAPACK APIs with representative workloads:

* Dense linear systems.
* Eigenvalue problems.
* Small-to-medium scientific kernels.
* Application kernels from molecular dynamics, CFD, quantum chemistry, and AI preprocessing.

---

## Phase 4: FFT and Transform Layer

**Target:** Add portable FFT support for HPC and signal-processing workloads

### 4.1 FFT APIs

Add unified APIs for:

* 1D complex-to-complex FFT.
* 2D complex-to-complex FFT.
* 3D complex-to-complex FFT.
* Real-to-complex and complex-to-real transforms.
* Batched FFT execution.

### 4.2 Backend Support

* CPU backend through FFTW or equivalent open backend.
* NVIDIA backend through cuFFT.
* AMD backend through rocFFT.
* Future provision for ParaS-native FFT kernels.

### 4.3 Validation

* Add correctness tests using inverse transform reconstruction.
* Add accuracy and tolerance checks across backends.
* Add benchmark cases for different transform sizes and batch counts.

---

## Phase 5: Sparse Linear Algebra and PDE Kernels

**Target:** Support PDE solvers and large-scale scientific applications

### 5.1 Sparse Matrix Support

Introduce core sparse data structures and APIs:

* Compressed Sparse Row (CSR) matrix representation.
* Coordinate Format (COO) matrix representation.
* Dense vector and dense matrix wrappers.
* Sparse matrix-vector multiplication.
* Sparse triangular solve.
* Sparse matrix-matrix multiplication.

### 5.2 Iterative Solvers

Add portable iterative solvers:

* Conjugate Gradient.
* BiConjugate Gradient Stabilized (BiCGSTAB).
* Generalized Minimal Residual (GMRES).
* Jacobi preconditioner.
* ILU/IC preconditioner interface.

### 5.3 PDE Kernels

Add common PDE-oriented kernels:

* Stencil operators.
* Poisson solver building blocks.
* Heat equation kernels.
* Laplacian operators.
* Structured-grid kernels.
* Finite-difference and Finite-volume primitives.

### 5.4 Backend Support

* CPU sparse backend using open sparse libraries where available.
* NVIDIA sparse backend through cuSPARSE.
* AMD sparse backend through rocSPARSE.
* ParaS-native kernels for selected PDE primitives.

---

## Phase 6: Performance Portability and Auto-Selection

**Target:** Improve usability and runtime performance across platforms

### 6.1 Runtime Capability Query

Add APIs to query:

* Available backends.
* Supported routines per backend.
* Supported data types.
* Device properties.
* Library versions.

### 6.2 Performance Benchmarking

Add benchmark suite for:

* BLAS Level 1/2/3.
* LAPACK routines.
* FFT routines.
* Sparse kernels.
* PDE kernels.

Publish results for:

* x86 CPUs.
* ARM CPUs.
* NVIDIA GPUs.
* AMD GPUs.
* Future RISC-V platforms.

---

## Phase 7: ParaS Ecosystem Integration

**Target:** Make CLAP a core numerical layer for ParaS applications and frameworks

### 7.1 Compiler and Runtime Integration

* Validate CLAP with the ParaS Compiler.
* Add examples showing ParaS-based applications calling CLAP APIs.

### 7.2 Framework Integration

Enable CLAP integration with the following frameworks and libraries:

#### Scientific Python and Array Backends

* ParaS-enabled NumPy backend experiments.
* SciPy linear algebra and sparse solver backends.
* CuPy-compatible path for NVIDIA GPU array operations.
* PDE solver mini-apps using PETSc and PETSc4Py for Python-driven PDE workflows.

#### AI and Deep Learning Math Libraries

* PyTorch ATen operator backend for linear algebra kernels.
* TensorFlow XLA custom calls targeting CLAP BLAS routines.
* ONNX Runtime custom operator backend experiments using CLAP.

#### HPC and Simulation Frameworks

* Scientific application kernels.
* PDE solver mini-apps.

### 7.3 Application Enablement

Prioritize enablement for:

* Molecular dynamics workloads.
* CFD workloads.
* Weather and climate kernels.
* Quantum chemistry kernels.
* AI preprocessing and linear algebra kernels.

---

## Phase 8: Future Hardware and Indigenous Platform Enablement

**Target:** Prepare CLAP for emerging architectures

### 8.1 CPU Architectures

Expand validation across:

* x86 CPUs.
* ARM CPUs.
* RISC-V CPUs.
* Future indigenous CPU platforms.

### 8.2 Accelerator Architectures

Prepare abstraction layers for:

* GPUs.
* NPUs.
* QPUs.
* AI accelerators.
* Future domain-specific accelerators.

### 8.3 Backend Plugin Model

Introduce a backend plugin architecture so new hardware vendors or research teams can add optimized kernels without changing the core CLAP API.

Planned backend structure:

```text
CLAP Public API
    ↓
Backend Dispatch Layer
    ↓
OpenBLAS | cuBLAS | rocBLAS | Future Backends
```

---

## Contribution Areas

CLAP welcomes contributions in the following areas:

* OpenBLAS and LAPACK routine implementation and testing.
* cuSOLVER, rocSOLVER, LAPACK++, and ScaLAPACK solver APIs.
* FFTW, cuFFT, rocFFT, and oneMKL FFT backend integration.
* PETSc, Trilinos, cuSPARSE, rocSPARSE, and SuiteSparse sparse linear algebra kernels.
* PETSc, Firedrake, FEniCS, and deal.II PDE kernels and mini-apps.
* RISC-V toolchains validation and tuning.
* API reference, getting started guides, and porting documentation.
* Benchmark cases and performance comparison reports across backends.
* GitHub Actions workflows, CMake packaging, and release automation.
* Molecular dynamics, CFD, quantum chemistry, and AI application integration examples.

---

## Good First Issues

Suggested beginner-friendly tasks:

* Add examples for BLAS APIs.
* Add unit tests for selected BLAS routines.
* Improve README build instructions.
* Add backend availability checks.
* Add CMake option documentation.
* Add Doxygen comments for public headers.
* Add simple benchmark for `gemm` across available backends.
* Add troubleshooting notes for OpenBLAS, CUDA, and ROCm installations.

---

## Documentation Roadmap

Planned documentation should include:

* Getting started guide.
* Build and installation guide.
* Backend configuration guide.
* API reference.
* Contributor guide.
* Architecture overview.
* Examples guide.
* Benchmarking guide.
* Porting guide for application developers.
* Backend developer guide.

---

## Success Metrics

The progress of CLAP can be measured through:

* Number of supported numerical routines.
* Number of validated backends.
* Number of supported hardware platforms.
* Correctness test coverage.
* Performance benchmark coverage.
* Number of applications using CLAP.
* Community contributions and pull requests.
* Documentation completeness.
* Stable release adoption.

---

## Long-Term Direction

CLAP should become the portable numerical foundation for the ParaS Ecosystem. The library should allow developers to write numerical applications once and execute them efficiently across CPUs, GPUs, and future accelerators.

The long-term goal is not only to wrap existing vendor libraries, but also to create a sustainable open numerical ecosystem where Indian and global contributors can add optimized kernels, validate new hardware, and enable scientific applications without being tied to a single architecture or vendor stack.