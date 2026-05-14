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

echo "=== FIDESlib GPU vs CPU CiphertextMultiplication with Maximum Slots ==="
echo "Parameters: level=11, scaleModSize=45, batch=16384 (maximum slots)"
echo ""

cd ../../build

# Clean up old results
rm -f gpu_mult_16k_timing.csv cpu_mult_16k_timing.csv

echo "🚀 Running GPU multiplication benchmark (16,384 slots)..."
timeout 120 ./fideslib-bench \
    --benchmark_filter="^GeneralFixture/CiphertextMultiplication/0/0/16384/11$" \
    --benchmark_min_time=0.1s \
    --benchmark_out_format=csv \
    --benchmark_out=gpu_mult_16k_timing.csv \
    --benchmark_counters_tabular=true 2>&1 | grep -v "The number of inputs is very large" || true

if [ -f gpu_mult_16k_timing.csv ]; then
    GPU_LINE=$(grep "CiphertextMultiplication/0/0/16384/11" gpu_mult_16k_timing.csv | tail -1)
    if [ -n "$GPU_LINE" ]; then
        GPU_TIME=$(echo "$GPU_LINE" | cut -d',' -f3)
        echo "✅ GPU multiplication (16,384 slots): $GPU_TIME ns"
    fi
fi

echo ""
echo "🔥 Running CPU multiplication benchmark (16,384 slots)..."
timeout 300 ./fideslib-bench \
    --benchmark_filter="^GeneralFixture/CiphertextMultiplicationCPU/0/0/16384/11/manual_time$" \
    --benchmark_min_time=0.1s \
    --benchmark_out_format=csv \
    --benchmark_out=cpu_mult_16k_timing.csv \
    --benchmark_counters_tabular=true 2>&1 | grep -v "The number of inputs is very large" || true

if [ -f cpu_mult_16k_timing.csv ]; then
    CPU_LINE=$(grep "CiphertextMultiplicationCPU/0/0/16384/11" cpu_mult_16k_timing.csv | tail -1)
    if [ -n "$CPU_LINE" ]; then
        CPU_TIME=$(echo "$CPU_LINE" | cut -d',' -f3)
        echo "✅ CPU multiplication (16,384 slots): $CPU_TIME ns"
        
        # Compare with GPU if both results exist
        if [ -n "$GPU_TIME" ] && [ -n "$CPU_TIME" ]; then
            echo ""
            echo "📊 MULTIPLICATION PERFORMANCE COMPARISON (16,384 slots):"
            echo "GPU: $GPU_TIME ns"
            echo "CPU: $CPU_TIME ns"
            
            if command -v bc >/dev/null 2>&1; then
                GPU_TIME_NUMERIC=$(printf "%.0f" "$GPU_TIME" 2>/dev/null || echo "$GPU_TIME")
                CPU_TIME_NUMERIC=$(printf "%.0f" "$CPU_TIME" 2>/dev/null || echo "$CPU_TIME")
                SPEEDUP=$(echo "scale=2; $CPU_TIME_NUMERIC / $GPU_TIME_NUMERIC" | bc)
                echo "Speedup: ${SPEEDUP}x"
                echo ""
                echo "🎯 Maximum Slot Utilization Achievement:"
                echo "  • Processing 16,384 complex numbers simultaneously"
                echo "  • Perfect ciphertext capacity usage (100% slots filled)"
                echo "  • Scaling from 8 to 16,384 slots (2,048x more data)"
            fi
        fi
    fi
fi

echo ""
echo "Results saved to: gpu_mult_16k_timing.csv, cpu_mult_16k_timing.csv"
