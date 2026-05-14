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

echo "=== FIDESlib GPU MultScalar with Maximum Slots ==="
echo "Parameters: level=11, scaleModSize=45, batch=16384 (maximum slots)"
echo "⚠️  Note: CPU benchmark not available for this operation in FIDESlib"
echo ""

cd ../../build

# Clean up old results
rm -f gpu_scalarmult_16k_timing.csv

echo "🚀 Running GPU scalar multiplication benchmark (16,384 slots)..."
timeout 120 ./fideslib-bench \
    --benchmark_filter="^GeneralFixture/MultScalar/0/0/16384/11$" \
    --benchmark_min_time=0.1s \
    --benchmark_out_format=csv \
    --benchmark_out=gpu_scalarmult_16k_timing.csv \
    --benchmark_counters_tabular=true 2>&1 | grep -v "The number of inputs is very large" || true

if [ -f gpu_scalarmult_16k_timing.csv ]; then
    GPU_LINE=$(grep "MultScalar/0/0/16384/11" gpu_scalarmult_16k_timing.csv | tail -1)
    if [ -n "$GPU_LINE" ]; then
        GPU_TIME=$(echo "$GPU_LINE" | cut -d',' -f3)
        GPU_TIME_MS=$(echo "scale=3; $GPU_TIME/1000000" | bc 2>/dev/null || echo "N/A")
        echo "✅ GPU scalar multiplication (16,384 slots): $GPU_TIME ns ($GPU_TIME_MS ms)"
        echo ""
        echo "🎯 Maximum Slot Utilization Achievement:"
        echo "  • Processing 16,384 complex numbers simultaneously"
        echo "  • Perfect ciphertext capacity usage (100% slots filled)"
        echo "  • Scaling from 8 to 16,384 slots (2,048x more data)"
    fi
fi

echo ""
echo "Results saved to: gpu_scalarmult_16k_timing.csv"
echo ""
echo "💡 Note: For CPU comparison, use OpenFHE's EvalMult(ct, scalar) directly"
