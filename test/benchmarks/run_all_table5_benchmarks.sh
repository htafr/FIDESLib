#!/bin/bash

#  Copyright (c) 2026 LG Electronics, Inc.
#
#  Licensed under the MIT License (the "License"); you may not use this file
#  except in compliance with the License.
#
#  You may obtain a copy of the License in the LICENSE file at the project
#  root or at
#
#  https://mit-license.org/
#
#  SPDX-License-Identifier: MIT

echo "═══════════════════════════════════════════════════════════════════════"
echo "  FIDESlib Table V Complete Operations Benchmark Suite"
echo "  Testing all 8 operations with Maximum Slots (16,384)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Parameters: multDepth=15, scaleModSize=45, level=11, batch=16384"
echo "Ring Dimension: 32,768 (2^15)"
echo "Slot Utilization: 100% (16,384 / 16,384 max slots)"
echo ""
echo "This will run benchmarks for all operations from Table V:"
echo "  1. HMult      - Homomorphic Multiplication"
echo "  2. HRotate    - Ciphertext Rotation"
echo "  3. HAdd       - Homomorphic Addition"
echo "  4. PtAdd      - Plaintext Addition"
echo "  5. ScalarAdd  - Scalar Addition"
echo "  6. PtMult     - Plaintext Multiplication"
echo "  7. ScalarMult - Scalar Multiplication"
echo "  8. Rescale    - Scale Reduction"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Store results for final summary
declare -A GPU_TIMES
declare -A CPU_TIMES
declare -A SPEEDUPS

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Operation list
operations=("hmult" "rot" "hadd" "ptadd" "scalaradd" "ptmult" "scalarmult" "rescale")
operation_names=("HMult" "HRotate" "HAdd" "PtAdd" "ScalarAdd" "PtMult" "ScalarMult" "Rescale")

# Run each benchmark
for i in "${!operations[@]}"; do
    op="${operations[$i]}"
    name="${operation_names[$i]}"
    
    echo ""
    echo "───────────────────────────────────────────────────────────────────────"
    echo "  Operation $((i+1))/8: $name"
    echo "───────────────────────────────────────────────────────────────────────"
    
    script="$SCRIPT_DIR/gpu_vs_cpu_${op}.sh"
    
    if [ -f "$script" ]; then
        # Run the benchmark script
        bash "$script"
        
        # Extract results from CSV files in build directory
        cd ../../build
        
        # Determine the CSV filename pattern based on operation
        case "$op" in
            "hmult")
                gpu_csv="gpu_mult_16k_timing.csv"
                cpu_csv="cpu_mult_16k_timing.csv"
                gpu_filter_pattern="CiphertextMultiplication/0/0/16384/11"
                cpu_filter_pattern="GeneralFixture/CiphertextMultiplicationCPU/0/0/16384/11/manual_time"
                ;;
            "rot")
                gpu_csv="gpu_rotation_16k_timing.csv"
                cpu_csv="cpu_rotation_16k_timing.csv"
                gpu_filter_pattern="CiphertextRotation/0/0/16384/11"
                cpu_filter_pattern="GeneralFixture/CiphertextRotationCPU/0/0/16384/11/manual_time"
                ;;
            "hadd")
                gpu_csv="gpu_hadd_16k_timing.csv"
                cpu_csv="cpu_hadd_16k_timing.csv"
                gpu_filter_pattern="CiphertextAdd/0/0/16384"
                cpu_filter_pattern="GeneralFixture/CiphertextAddCPU/0/0/16384/manual_time"
                ;;
            "ptadd")
                gpu_csv="gpu_ptadd_16k_timing.csv"
                cpu_csv="cpu_ptadd_16k_timing.csv"
                gpu_filter_pattern="AddPlaintext/0/0/16384"
                cpu_filter_pattern="GeneralFixture/AddPlaintextCPU/0/0/16384/manual_time"
                ;;
            "scalaradd")
                gpu_csv="gpu_scalaradd_16k_timing.csv"
                cpu_csv="cpu_scalaradd_16k_timing.csv"
                gpu_filter_pattern="AddScalar/0/0/16384"
                cpu_filter_pattern="GeneralFixture/AddScalarCPU/0/0/16384/manual_time"
                ;;
            "ptmult")
                gpu_csv="gpu_ptmult_16k_timing.csv"
                cpu_csv="cpu_ptmult_16k_timing.csv"
                gpu_filter_pattern="MultPlaintext/0/0/16384/11"
                cpu_filter_pattern="GeneralFixture/MultPlaintextCPU/0/0/16384/11/manual_time"
                ;;
            "scalarmult")
                gpu_csv="gpu_scalarmult_16k_timing.csv"
                cpu_csv="cpu_scalarmult_16k_timing.csv"
                gpu_filter_pattern="MultScalar/0/0/16384/11"
                cpu_filter_pattern="GeneralFixture/MultScalarCPU/0/0/16384/11/manual_time"
                ;;
            "rescale")
                gpu_csv="gpu_rescale_16k_timing.csv"
                cpu_csv="cpu_rescale_16k_timing.csv"
                gpu_filter_pattern="Rescale/0/0/16384/11"
                cpu_filter_pattern="GeneralFixture/RescaleCPU/0/0/16384/11/manual_time"
                ;;
        esac
        
        # Extract GPU time
        if [ -f "$gpu_csv" ]; then
            gpu_line=$(grep "$gpu_filter_pattern" "$gpu_csv" | tail -1)
            if [ -n "$gpu_line" ]; then
                GPU_TIMES[$name]=$(echo "$gpu_line" | cut -d',' -f3)
            fi
        fi
        
        # Extract CPU time
        if [ -f "$cpu_csv" ] && [ -n "$cpu_filter_pattern" ]; then
            cpu_line=$(grep "$cpu_filter_pattern" "$cpu_csv" | tail -1)
            if [ -n "$cpu_line" ]; then
                CPU_TIMES[$name]=$(echo "$cpu_line" | cut -d',' -f3)
            fi
        fi
        
        # Calculate speedup
        if [ -n "${GPU_TIMES[$name]}" ] && [ -n "${CPU_TIMES[$name]}" ]; then
            gpu_num=$(printf "%.0f" "${GPU_TIMES[$name]}" 2>/dev/null)
            cpu_num=$(printf "%.0f" "${CPU_TIMES[$name]}" 2>/dev/null)
            if command -v bc >/dev/null 2>&1; then
                SPEEDUPS[$name]=$(echo "scale=2; $cpu_num / $gpu_num" | bc)
            fi
        fi
        
        cd "$SCRIPT_DIR"
    else
        echo "⚠️  Script not found: $script"
    fi
done

# Generate final summary report
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  FINAL SUMMARY: Table V Operations Performance"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "All timings in nanoseconds (ns) for 16,384 slots"
echo ""
printf "%-15s %15s %15s %12s\n" "Operation" "GPU Time (ns)" "CPU Time (ns)" "Speedup"
echo "───────────────────────────────────────────────────────────────────────"

for name in "${operation_names[@]}"; do
    gpu_time="${GPU_TIMES[$name]:-N/A}"
    cpu_time="${CPU_TIMES[$name]:-N/A}"
    speedup="${SPEEDUPS[$name]:-N/A}"
    
    if [ "$speedup" != "N/A" ]; then
        speedup="${speedup}x"
    fi
    
    printf "%-15s %15s %15s %12s\n" "$name" "$gpu_time" "$cpu_time" "$speedup"
done

echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Calculate average speedup
if command -v bc >/dev/null 2>&1; then
    total_speedup=0
    count=0
    for speedup in "${SPEEDUPS[@]}"; do
        if [ -n "$speedup" ]; then
            total_speedup=$(echo "$total_speedup + $speedup" | bc)
            count=$((count + 1))
        fi
    done
    
    if [ $count -gt 0 ]; then
        avg_speedup=$(echo "scale=2; $total_speedup / $count" | bc)
        echo "📊 OVERALL STATISTICS:"
        echo "   Average GPU Speedup: ${avg_speedup}x"
        echo "   Operations Tested: $count/8"
        echo ""
    fi
fi

echo "💾 All detailed results saved in: ../../build/"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Benchmark Suite Complete!"
echo "═══════════════════════════════════════════════════════════════════════"
