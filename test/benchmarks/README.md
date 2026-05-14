<!--
Copyright (c) 2026 LG Electronics, Inc.

Licensed under the MIT License (the "License"); you may not use this file
except in compliance with the License.

You may obtain a copy of the License in the LICENSE file at the project
root or at

https://mit-license.org/

SPDX-License-Identifier: MIT
-->

# FIDESlib GPU Benchmarks

This directory contains GPU vs CPU performance comparison scripts for FIDESlib homomorphic encryption operations, covering all 8 operations from **Table V** of the FIDESlib paper.

## Quick Start

### Run All Benchmarks at Once

```bash
./run_all_table5_benchmarks.sh
```

This will run all 8 operations sequentially and generate a comprehensive comparison report.

### Run Individual Operations

```bash
# Homomorphic operations (ciphertext-ciphertext)
./gpu_vs_cpu_rot.sh        # Rotation
./gpu_vs_cpu_hmult.sh      # Multiplication
./gpu_vs_cpu_hadd.sh       # Addition

# Plaintext operations (ciphertext-plaintext)
./gpu_vs_cpu_ptadd.sh      # Plaintext addition
./gpu_vs_cpu_ptmult.sh     # Plaintext multiplication

# Scalar operations (ciphertext-scalar)
./gpu_vs_cpu_scalaradd.sh  # Scalar addition
./gpu_vs_cpu_scalarmult.sh # Scalar multiplication

# Scale management
./gpu_vs_cpu_rescale.sh    # Rescale operation
```

## Benchmark Parameters

All scripts test with **maximum slots (16,384)** using `custom_11_45` parameters:

- **Multiplicative depth**: 15
- **Scale modulus size**: 45 bits
- **Batch size**: 16,384 complex numbers (100% slot utilization)
- **Ring dimension**: 32,768 (2^15)
- **Level**: 11

This represents 2,048x more data than traditional 8-slot benchmarks with optimal use of ciphertext capacity.

## Available Operations

### Table V Operations Summary

| # | Operation | Script | OpenFHE Equivalent | Expected Speedup |
|---|-----------|--------|-------------------|------------------|
| 1 | **HMult** | `gpu_vs_cpu_hmult.sh` | `EvalMult(ct1, ct2)` | 100-300x |
| 2 | **HRotate** | `gpu_vs_cpu_rot.sh` | `EvalRotate(ct, 1)` | 200-400x |
| 3 | **HAdd** | `gpu_vs_cpu_hadd.sh` | `EvalAdd(ct1, ct2)` | 10-50x |
| 4 | **PtAdd** | `gpu_vs_cpu_ptadd.sh` | `EvalAdd(ct, pt)` | 10-30x |
| 5 | **ScalarAdd** | `gpu_vs_cpu_scalaradd.sh` | `EvalAdd(ct, scalar)` | 5-20x |
| 6 | **PtMult** | `gpu_vs_cpu_ptmult.sh` | `EvalMult(ct, pt)` | 50-150x |
| 7 | **ScalarMult** | `gpu_vs_cpu_scalarmult.sh` | `EvalMult(ct, scalar)` | 10-40x |
| 8 | **Rescale** | `gpu_vs_cpu_rescale.sh` | `ModReduce(ct)` | 50-100x |

### Operation Descriptions

**Homomorphic Operations (Ciphertext-Ciphertext):**

- **HMult**: Homomorphic multiplication with key switching and relinearization
- **HRotate**: Ciphertext rotation by 1 position (requires key switching)
- **HAdd**: Element-wise addition of two ciphertexts

**Plaintext Operations (Ciphertext-Plaintext):**

- **PtAdd**: Add a plaintext to a ciphertext
- **PtMult**: Multiply a ciphertext by a plaintext

**Scalar Operations (Ciphertext-Scalar):**

- **ScalarAdd**: Add a scalar value to all ciphertext slots
- **ScalarMult**: Multiply all ciphertext slots by a scalar

**Scale Management:**

- **Rescale**: Reduce the scale in CKKS (modular reduction)

## Usage

### Quick Start - Run All Benchmarks

To benchmark all 8 operations from Table V at once:

```bash
./run_all_table5_benchmarks.sh
```

This will run all operations sequentially and generate a comprehensive comparison report.

### Individual Operation Benchmarks

Run specific operations from this directory:

```bash
# Homomorphic operations (ciphertext-ciphertext)
./gpu_vs_cpu_rot.sh        # Rotation
./gpu_vs_cpu_hmult.sh      # Multiplication
./gpu_vs_cpu_hadd.sh       # Addition

# Plaintext operations (ciphertext-plaintext)
./gpu_vs_cpu_ptadd.sh      # Plaintext addition
./gpu_vs_cpu_ptmult.sh     # Plaintext multiplication

# Scalar operations (ciphertext-scalar)
./gpu_vs_cpu_scalaradd.sh  # Scalar addition
./gpu_vs_cpu_scalarmult.sh # Scalar multiplication

# Scale management
./gpu_vs_cpu_rescale.sh    # Rescale operation
```

## Output

Each script generates:

- Console output with timing results and speedup analysis
- CSV files with detailed benchmark results in `../../build/` directory:
  - **Rotation**: `gpu_rotation_16k_timing.csv`, `cpu_rotation_16k_timing.csv`
  - **HMult**: `gpu_mult_16k_timing.csv`, `cpu_mult_16k_timing.csv`
  - **HAdd**: `gpu_hadd_16k_timing.csv`, `cpu_hadd_16k_timing.csv`
  - **PtAdd**: `gpu_ptadd_16k_timing.csv`, `cpu_ptadd_16k_timing.csv`
  - **ScalarAdd**: `gpu_scalaradd_16k_timing.csv`, `cpu_scalaradd_16k_timing.csv`
  - **PtMult**: `gpu_ptmult_16k_timing.csv`, `cpu_ptmult_16k_timing.csv`
  - **ScalarMult**: `gpu_scalarmult_16k_timing.csv`, `cpu_scalarmult_16k_timing.csv`
  - **Rescale**: `gpu_rescale_16k_timing.csv`, `cpu_rescale_16k_timing.csv`

## Benchmark Filters

Each script uses specific Google Benchmark filters to isolate individual operations:

```bash
# Homomorphic operations (require level parameter)
HMult:      "^GeneralFixture/CiphertextMultiplication/0/0/16384/11$"
HRotate:    "^GeneralFixture/CiphertextRotation/0/0/16384/11$"
PtMult:     "^GeneralFixture/MultPlaintext/0/0/16384/11$"
ScalarMult: "^GeneralFixture/MultScalar/0/0/16384/11$"
Rescale:    "^GeneralFixture/Rescale/0/0/16384/11$"

# Addition operations (no level parameter needed)
HAdd:       "^GeneralFixture/CiphertextAdd/0/0/16384$"
PtAdd:      "^GeneralFixture/AddPlaintext/0/0/16384$"
ScalarAdd:  "^GeneralFixture/AddScalar/0/0/16384$"
```

## Important Notes

### Data Transfer Overhead

Current benchmarks measure **GPU computation time only**, excluding:

- OpenFHE → FIDESlib conversion overhead
- Host → GPU upload time (~0.56 ms for 7MB ciphertext)
- GPU → Host download time (~0.56 ms)

For **batch operations** on the same data, data transfer costs are amortized across multiple operations.

### Benchmark Configuration

- **Minimum time**: All benchmarks use `--benchmark_min_time=0.1s` for statistical reliability
- **Slot utilization**: 100% capacity (16,384 / 16,384 slots) for optimal performance
- **Iterations**: Multiple iterations automatically run for accurate timing

## Requirements

- FIDESlib must be built (requires `fideslib-bench` executable in `../../build/`)
- CUDA-capable GPU with proper drivers
- Sufficient system memory for 16,384-slot operations

## Troubleshooting

If benchmarks fail, check the following:

1. **Build status**: Ensure `fideslib-bench` is built in `../../build/`

   ```bash
   ls -lh ../../build/fideslib-bench
   ```

2. **GPU drivers**: Verify CUDA and GPU are working

   ```bash
   nvidia-smi
   ```

3. **CUDA installation**: Check CUDA toolkit is properly installed

   ```bash
   nvcc --version
   ```

4. **Memory availability**: 16,384 slots require significant RAM

   ```bash
   free -h
   ```

5. **Build logs**: Review for compilation errors

   ```bash
   cat ../../build/CMakeFiles/CMakeError.log
   ```

## Performance Analysis

### Viewing Results

```bash
# Navigate to results directory
cd ../../build

# List all result files
ls -lh *_16k_timing.csv

# View GPU results for a specific operation
grep "16384" gpu_mult_16k_timing.csv

# View CPU results
grep "16384" cpu_mult_16k_timing.csv
```

### Calculating Speedups

Use Python for quick speedup calculations:

```bash
# Calculate speedup for multiplication
python3 -c "
import csv
gpu = float(list(csv.reader(open('../../build/gpu_mult_16k_timing.csv')))[-1][2])
cpu = float(list(csv.reader(open('../../build/cpu_mult_16k_timing.csv')))[-1][2])
print(f'Speedup: {cpu/gpu:.2f}x')
"
```

### Quick Analysis Script

Create a quick analysis script to compare all operations:

```bash
#!/bin/bash
cd ../../build
for op in mult rotation hadd ptadd scalaradd ptmult scalarmult rescale; do
    echo "=== $op ==="
    python3 -c "
import csv
gpu = float(list(csv.reader(open('gpu_${op}_16k_timing.csv')))[-1][2])
cpu = float(list(csv.reader(open('cpu_${op}_16k_timing.csv')))[-1][2])
print(f'GPU: {gpu:.3f}ms, CPU: {cpu:.3f}ms, Speedup: {cpu/gpu:.2f}x')
    "
done
```

## Maximum Slot Utilization

These benchmarks test with maximum ciphertext capacity:

- **Batch Size**: 16,384 complex numbers (100% slot utilization)
- **Ring Dimension**: 32,768 (2^15)
- **Scaling**: 2,048x more data than traditional 8-slot benchmarks
- **Memory Efficiency**: Optimal use of ciphertext capacity

## Directory Structure

```
test/benchmarks/
├── README.md                       # This file
├── run_all_table5_benchmarks.sh   # Run all 8 operations
├── gpu_vs_cpu_rot.sh              # Rotation benchmarks
├── gpu_vs_cpu_hmult.sh            # Homomorphic multiplication
├── gpu_vs_cpu_hadd.sh             # Homomorphic addition
├── gpu_vs_cpu_ptadd.sh            # Plaintext addition
├── gpu_vs_cpu_scalaradd.sh        # Scalar addition
├── gpu_vs_cpu_ptmult.sh           # Plaintext multiplication
├── gpu_vs_cpu_scalarmult.sh       # Scalar multiplication
└── gpu_vs_cpu_rescale.sh          # Rescale operation
```

Results are saved in the build directory (`../../build/`) alongside the benchmark executable.

## FIDESlib Paper Table V Operations

These scripts cover all 8 operations from **Table V** of the FIDESlib paper:

| Operation | Script | Description |
|-----------|--------|-------------|
| **HMult** | `gpu_vs_cpu_hmult.sh` | Homomorphic multiplication (ct × ct) |
| **HRotate** | `gpu_vs_cpu_rot.sh` | Ciphertext rotation |
| **HAdd** | `gpu_vs_cpu_hadd.sh` | Homomorphic addition (ct + ct) |
| **PtAdd** | `gpu_vs_cpu_ptadd.sh` | Plaintext addition (ct + pt) |
| **ScalarAdd** | `gpu_vs_cpu_scalaradd.sh` | Scalar addition (ct + scalar) |
| **PtMult** | `gpu_vs_cpu_ptmult.sh` | Plaintext multiplication (ct × pt) |
| **ScalarMult** | `gpu_vs_cpu_scalarmult.sh` | Scalar multiplication (ct × scalar) |
| **Rescale** | `gpu_vs_cpu_rescale.sh` | Scale reduction in CKKS |
