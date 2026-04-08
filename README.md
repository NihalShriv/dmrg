# GPU DMRG for the 1D Transverse-Field Ising Model

This repository contains a CUDA/C++ implementation of a two-site Density Matrix Renormalization Group (DMRG) solver for the 1D transverse-field Ising model (TFIM) with open boundary conditions.

The code builds a matrix product state (MPS), represents the Hamiltonian as a matrix product operator (MPO), constructs left/right environments on the GPU, solves the local two-site effective problem with a Davidson eigensolver, and truncates the updated bond with an SVD using cuSOLVER.

## What the program does

- Runs a small exact diagonalization check for `N = 6` to compare the DMRG energy against a reference value.
- Runs larger TFIM ground-state calculations for `N = 80` at `h = 1.0`.
- Sweeps over bond dimensions `D = 32, 48, 64, 80, 96`.
- Prints per-sweep energies and final energies for each bond dimension.

The model used in the code is:

`H = Jzz * sum_i Z_i Z_{i+1} - h * sum_i X_i`

with:

- `Jzz = -1.0`
- open boundary conditions
- physical dimension `d = 2`

## Repository layout

### Entry point

- `main.cu`  
  Runs the exact small-system reference calculation, launches DMRG runs, and prints results.

- `dmrg_prof.cu`  
  Main DMRG driver. Initializes the MPS/MPO, performs left-to-right and right-to-left sweeps, checks convergence, and returns the final energy.

### Tensor network data structures

- `tensor.h`, `tensor.cu`  
  Minimal GPU tensor wrapper around device memory allocation, host/device copies, reshaping, and scalar extraction.

- `mps.h`, `mps.cu`  
  Defines the matrix product state and random initialization with capped bond dimensions.

- `mpo.h`, `mpo.cu`  
  Defines the matrix product operator and builds the TFIM MPO with bond dimension 3.

### Linear algebra

- `linalg.h`, `linalg.cu`  
  Thin wrappers over cuBLAS and cuSOLVER for:
  - GEMM
  - QR
  - symmetric eigendecomposition
  - SVD
  - vector norms and BLAS vector ops

### Canonicalization and environments

- `canonical.h`, `canonical.cu`  
  Left- and right-canonicalization routines using QR decompositions.

- `environment.h`, `environment.cu`  
  Builds and updates left and right contraction environments for the MPS/MPO network.

### Effective Hamiltonian and local optimization

- `heff_builder.h`, `heff_builder.cu`  
  Applies the two-site effective Hamiltonian and includes a small-system validation routine for correctness and Hermiticity checks.

- `two_site_split.h`, `two_site_split.cu`  
  Core two-site DMRG update:
  - forms the two-site wavefunction
  - solves the local ground-state problem with Davidson iteration
  - reshapes and normalizes the result
  - performs SVD truncation
  - updates neighboring MPS tensors

### Diagnostics and observables

- `energy_eval.h`, `energy_eval.cu`  
  Computes total energy using the full left environment contraction.

- `norm_eval.h`, `norm_eval.cu`  
  Computes the MPS norm.

## Algorithm overview

1. Initialize a random MPS with maximum bond dimension `D`.
2. Build the TFIM MPO.
3. Left-canonicalize the initial MPS.
4. Perform two-site DMRG sweeps:
   - build the opposite-side environment
   - solve the local effective Hamiltonian ground state
   - split the optimized two-site tensor with SVD
   - truncate according to the allowed bond dimension
   - update environments incrementally
5. Recompute the energy after each full sweep.
6. Stop when energy change, residual, and discarded weight are all below tolerance.

## Requirements

- NVIDIA GPU
- CUDA Toolkit
- cuBLAS
- cuSOLVER
- C++ compiler supported by `nvcc`

This repo does not currently include a `CMakeLists.txt` or Makefile, so the project is intended to be compiled directly with `nvcc`.

## Build

Example Windows build command:

```powershell
nvcc -O3 -std=c++17 `
  main.cu dmrg_prof.cu canonical.cu environment.cu heff_builder.cu `
  linalg.cu mpo.cu mps.cu energy_eval.cu norm_eval.cu tensor.cu two_site_split.cu `
  -lcublas -lcusolver `
  -o dmrg_tfim.exe
```

Example Linux build command:

```bash
nvcc -O3 -std=c++17 \
  main.cu dmrg_prof.cu canonical.cu environment.cu heff_builder.cu \
  linalg.cu mpo.cu mps.cu energy_eval.cu norm_eval.cu tensor.cu two_site_split.cu \
  -lcublas -lcusolver \
  -o dmrg_tfim
```

If `nvcc` is not found, make sure the CUDA Toolkit is installed and its `bin` directory is available on your system `PATH`.

## Run

On Windows:

```powershell
.\dmrg_tfim.exe
```

On Linux:

```bash
./dmrg_tfim
```

## Expected output

The program prints:

- an exact-vs-DMRG verification line for the small `N = 6` case
- sweep-by-sweep energies for the larger run
- a final energy summary for each tested bond dimension

A typical output shape looks like:

```text
Check: N=6 h=1 Exact=... DMRG=... Delta=...
Model: TFIM  N=80  h=1
N=80 D=32 Sweep=1 Energy=...
...
Final: N=80 D=32 Energy=...
```

## Notes

- The implementation is focused on the transverse-field Ising model only.
- The MPO construction is specialized for open boundary conditions.
- The code uses custom CUDA kernels for tensor-network contractions rather than a general tensor library.
- Several validation and stability checks are included for small systems and environment growth.

## Possible future improvements

- Add `CMakeLists.txt`
- Add command-line arguments for `N`, `D`, `h`, and number of sweeps
- Add timing/profiling output
- Add support for more Hamiltonians
- Add tests and benchmark cases
- Add CPU reference checks for more components

## Summary

This project is a compact GPU DMRG implementation for the 1D TFIM, built from scratch with CUDA, cuBLAS, and cuSOLVER. It is useful as a learning project, a research prototype, or a starting point for extending tensor-network simulations on NVIDIA GPUs.
