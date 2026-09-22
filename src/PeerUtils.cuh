//
// Created by carlosad on 18/12/25.
//

#ifndef FIDESLIB_PEERUTILS_CUH
#define FIDESLIB_PEERUTILS_CUH

#include "CudaUtils.cuh"
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#undef duration

namespace FIDESlib {

/**
 * Check if a HIP stream is currently being captured
 * Returns true if capture is active, false otherwise
 */
bool is_stream_being_captured(hipStream_t stream);

void verify_all_streams_joined(hipStream_t main_stream);

const char* getNodeTypeName(hipGraphNodeType type);

// Print detailed graph with all edges and dependencies
void printGraphDependencies(hipGraph_t graph, const char* name = "Graph");

// Check if a path exists from source to sink
bool pathExists(hipGraphNode_t source, hipGraphNode_t sink, const std::vector<hipGraphNode_t>& all_nodes, std::set<uintptr_t>& visited);

void printGraphDependencies2(hipGraph_t graph, const char* name = "Graph");

// 1D P2P memcpy-style kernel: coalesced, vectorized, no warp specialization
__global__ void p2p_transfer_1d(const float* __restrict__ src, // pointer on source device (with P2P enabled)
  float* __restrict__ dst,									   // pointer on destination device
  size_t n													   // number of float elements
);

/** Initial profiling of peer transfers (PCIe 4.0x16) shows best performance with 4 thread blocks */
void transferKernel(float* src, float* dst, size_t elems, hipStream_t s, int src_dev, int dst_dev, size_t involved_sm = 4);

__global__ void notify_kernel(volatile uint32_t* gpu_complete_flag, uint32_t value);

struct SystemAtomicU64 {
	uint64_t v{ 0 };

	__host__ __device__ uint64_t load(int /*order*/ = __ATOMIC_RELAXED) const volatile {
#ifdef __HIP_DEVICE_COMPILE__
		return __hip_atomic_load(&v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
		return __atomic_load_n(&v, __ATOMIC_RELAXED);
#endif
	}

	__host__ __device__ void store(uint64_t val, int /*order*/ = __ATOMIC_RELAXED) volatile {
#ifdef __HIP_DEVICE_COMPILE__
		__hip_atomic_store(&v, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
		__atomic_store_n(&v, val, __ATOMIC_RELAXED);
#endif
	}

	__host__ __device__ uint64_t fetch_max(uint64_t val, int /*order*/ = __ATOMIC_RELAXED) volatile {
#ifdef __HIP_DEVICE_COMPILE__
		uint64_t old = __hip_atomic_load(&v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
		while (val > old) {
			if (__hip_atomic_compare_exchange_strong(&v, &old, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM))
				break;
		}
		return old;
#else
		uint64_t old = __atomic_load_n(&v, __ATOMIC_RELAXED);
		while (val > old && !__atomic_compare_exchange_n(&v, &old, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
		}
		return old;
#endif
	}
};

struct TimelineSemaphore {
	SystemAtomicU64 value{};
	char pad[120]; // Avoid false sharing
};

__global__ void notify_kernel_hostpin(TimelineSemaphore* gpu_complete_flag, uint64_t value);

void notifyKernel(TimelineSemaphore* gpu1_complete_flag, uint64_t value, hipStream_t s);

__global__ void p2p_polling_kernel(volatile uint32_t* completion_flag, uint32_t value // Flag on destination GPU
);

__global__ void hostpin_polling_kernel(TimelineSemaphore* completion_flag,
  uint64_t value // Flag on destination GPU
);

void pollingKernel(TimelineSemaphore* gpu1_complete_flag, uint64_t value, hipStream_t s);

[[maybe_unused]] __global__ static void dummy_kernel() {
}

} // namespace FIDESlib
#endif // FIDESLIB_PEERUTILS_CUH
