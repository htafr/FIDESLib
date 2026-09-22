//
// Created by carlosad on 18/12/25.
//

#include <errno.h>
#include <nccl.h>

#include <iomanip>

#include <CudaUtils.cuh>
#include <PeerUtils.cuh>
#include <chrono>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include <omp.h>
// Simple CUDA error check macro
#define CUDA_CHECK(stmt)                                                                                                  \
	do {                                                                                                                  \
		hipError_t err = (stmt);                                                                                         \
		if (err != hipSuccess) {                                                                                         \
			std::cerr << "CUDA error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
			FIDESlib::breakpoint();                                                                                       \
		}                                                                                                                 \
	} while (0)

// Helper: initialize host data
static void init_host_data(std::vector<float>& h, float seed) {
	for (size_t i = 0; i < h.size(); ++i) {
		h[i] = seed + static_cast<float>(i) * 0.001f;
	}
}

// Helper: compare host buffers
static void expect_equal(const std::vector<float>& a, const std::vector<float>& b) {
	ASSERT_EQ(a.size(), b.size());
	for (size_t i = 0; i < a.size(); ++i) {
		ASSERT_FLOAT_EQ(a[i], b[i]) << "mismatch at index " << i;
	}
}

/**
 * Enable P2P access safely: handle case where already enabled
 * Returns true if P2P is enabled (either was already or just enabled)
 * Returns false if P2P cannot be enabled
 */
static void safe_enable_p2p(int src_gpu, int dst_gpu) {
	int can_access = 0;
	CUDA_CHECK(hipDeviceCanAccessPeer(&can_access, src_gpu, dst_gpu));

	if (!can_access) {
		std::cout << "  P2P not available: GPU " << src_gpu << " -> GPU " << dst_gpu << std::endl;
		return;
	}

	CUDA_CHECK(hipSetDevice(src_gpu));

	// Try to enable P2P; if already enabled, cudaDeviceEnablePeerAccess
	// returns cudaErrorPeerAccessAlreadyEnabled, which we ignore
	hipError_t err = hipDeviceEnablePeerAccess(dst_gpu, 0);

	if (err == hipErrorPeerAccessAlreadyEnabled) {
		std::cout << "  P2P already enabled: GPU " << src_gpu << " -> GPU " << dst_gpu << std::endl;
		return;
	} else if (err == hipSuccess) {
		std::cout << "  P2P enabled: GPU " << src_gpu << " -> GPU " << dst_gpu << std::endl;
		return;
	} else {
		std::cerr << "  Failed to enable P2P: GPU " << src_gpu << " -> GPU " << dst_gpu << ": " << hipGetErrorString(err) << std::endl;
		return;
	}
}

/**
 * Disable P2P access safely: handle case where not enabled
 */
[[maybe_unused]] static void safe_disable_p2p(int src_gpu, int dst_gpu) {
	CUDA_CHECK(hipSetDevice(src_gpu));
	hipError_t err = hipDeviceDisablePeerAccess(dst_gpu);

	if (err == hipErrorPeerAccessNotEnabled) {
		// Already disabled, that's fine
		return;
	} else if (err == hipSuccess) {
		std::cout << "  P2P disabled: GPU " << src_gpu << " -> GPU " << dst_gpu << std::endl;
	} else {
		std::cerr << "  Warning: Failed to disable P2P: GPU " << src_gpu << " -> GPU " << dst_gpu << ": " << hipGetErrorString(err) << std::endl;
	}
}

// Common test body: assumes src/dst are on devDst and devSrc chosen appropriately
static void run_p2p_test(int devSrc, int devDst, size_t n) {
	CUDA_CHECK(hipSetDevice(devSrc));
	float* d_src = nullptr;
	CUDA_CHECK(hipMalloc(&d_src, n * sizeof(float)));

	CUDA_CHECK(hipSetDevice(devDst));
	float* d_dst = nullptr;
	CUDA_CHECK(hipMalloc(&d_dst, n * sizeof(float)));

	uint32_t* d_flag = nullptr;
	CUDA_CHECK(hipMalloc(&d_flag, sizeof(uint32_t)));
	CUDA_CHECK(hipMemset(d_flag, 0, sizeof(uint32_t)));

	// Initialize host data and copy to src device
	std::vector<float> h_src(n), h_dst(n);
	init_host_data(h_src, 1.0f);

	CUDA_CHECK(hipSetDevice(devSrc));
	CUDA_CHECK(hipMemcpy(d_src, h_src.data(), n * sizeof(float), hipMemcpyHostToDevice));

	// Launch transfer and notify kernels on src device
	CUDA_CHECK(hipSetDevice(devSrc));
	hipStream_t sSrc;
	CUDA_CHECK(hipStreamCreate(&sSrc));

	int threads = 256;
	int blocks	= static_cast<int>((n + threads - 1) / threads);

	FIDESlib::p2p_transfer_1d<<<blocks, threads, 0, sSrc>>>(d_src, d_dst, n);
	CUDA_CHECK(hipGetLastError());

	uint32_t expectedValue = 1;
	FIDESlib::notify_kernel<<<1, 32, 0, sSrc>>>(d_flag, expectedValue);
	CUDA_CHECK(hipGetLastError());

	// On destination device, wait for flag and then read back
	CUDA_CHECK(hipSetDevice(devDst));
	hipStream_t sDst;
	CUDA_CHECK(hipStreamCreate(&sDst));

	FIDESlib::p2p_polling_kernel<<<1, 256, 0, sDst>>>(d_flag, expectedValue);
	CUDA_CHECK(hipGetLastError());

	// Wait for destination stream to finish (i.e., transfer+notify complete)
	CUDA_CHECK(hipStreamSynchronize(sDst));

	// Copy dst back to host from destination device
	CUDA_CHECK(hipMemcpy(h_dst.data(), d_dst, n * sizeof(float), hipMemcpyDeviceToHost));

	// Verify correctness
	expect_equal(h_src, h_dst);

	// Cleanup
	CUDA_CHECK(hipStreamDestroy(sSrc));
	CUDA_CHECK(hipStreamDestroy(sDst));

	CUDA_CHECK(hipSetDevice(devSrc));
	CUDA_CHECK(hipFree(d_src));

	CUDA_CHECK(hipSetDevice(devDst));
	CUDA_CHECK(hipFree(d_dst));
	CUDA_CHECK(hipFree(d_flag));
}

static void run_p2p_test_graph(int devSrc, int devDst, size_t n) {
	CUDA_CHECK(hipSetDevice(devSrc));
	float* d_src = nullptr;
	CUDA_CHECK(hipMalloc(&d_src, n * sizeof(float)));

	CUDA_CHECK(hipSetDevice(devDst));
	float* d_dst = nullptr;
	CUDA_CHECK(hipMalloc(&d_dst, n * sizeof(float)));

	FIDESlib::TimelineSemaphore* d_flag = nullptr;
	CUDA_CHECK(hipMalloc(&d_flag, sizeof(uint32_t)));
	CUDA_CHECK(hipMemset(d_flag, 0, sizeof(uint32_t)));

	// Initialize host data and copy to src device
	std::vector<float> h_src(n), h_dst(n);
	init_host_data(h_src, 1.0f);

	CUDA_CHECK(hipSetDevice(devSrc));
	CUDA_CHECK(hipMemcpy(d_src, h_src.data(), n * sizeof(float), hipMemcpyHostToDevice));

	// Launch transfer and notify kernels on src device
	CUDA_CHECK(hipSetDevice(devSrc));
	hipStream_t sSrc;
	CUDA_CHECK(hipStreamCreate(&sSrc));

	CUDA_CHECK(hipStreamBeginCapture(sSrc, hipStreamCaptureModeGlobal));

	CUDA_CHECK(hipSetDevice(devDst));
	hipStream_t sDst;
	CUDA_CHECK(hipStreamCreate(&sDst));

	CUDA_CHECK(hipSetDevice(devSrc));

	hipEvent_t event0;
	CUDA_CHECK(hipEventCreateWithFlags(&event0, hipEventDisableTiming));
	CUDA_CHECK(hipEventRecord(event0, sSrc));
	CUDA_CHECK(hipStreamWaitEvent(sDst, event0));

	CUDA_CHECK(hipEventDestroy(event0));

	if (0) {
		int threads = 256;
		int blocks	= static_cast<int>((n + threads - 1) / threads);

		FIDESlib::p2p_transfer_1d<<<blocks, threads, 0, sSrc>>>(d_src, d_dst, n);
	} else {
		FIDESlib::transferKernel(d_src, d_dst, n, sSrc, devSrc, devDst);
	}
	CUDA_CHECK(hipGetLastError());

	uint32_t expectedValue = 1;

	if (0) {
		FIDESlib::notify_kernel_hostpin<<<1, 32, 0, sSrc>>>(d_flag, expectedValue);
	} else {
		FIDESlib::notifyKernel(d_flag, expectedValue, sSrc);
	}
	CUDA_CHECK(hipGetLastError());

	// On destination device, wait for flag and then read back

	if (0) {
		FIDESlib::hostpin_polling_kernel<<<1, 32, 0, sDst>>>(d_flag, expectedValue);
	} else {
		FIDESlib::pollingKernel(d_flag, expectedValue, sDst);
	}

	CUDA_CHECK(hipGetLastError());

	hipEvent_t event1;

	CUDA_CHECK(hipSetDevice(devDst));
	CUDA_CHECK(hipEventCreateWithFlags(&event1, hipEventDisableTiming));
	CUDA_CHECK(hipEventRecord(event1, sDst));
	CUDA_CHECK(hipStreamWaitEvent(sSrc, event1));

	CUDA_CHECK(hipEventDestroy(event1));
	{
		hipGraph_t graph;
		CUDA_CHECK(hipStreamEndCapture(sSrc, &graph));

		hipGraphExec_t exec;
		CUDA_CHECK(hipGraphInstantiate(&exec, graph));

		CUDA_CHECK(hipGraphLaunch(exec, sSrc));
		CudaCheckErrorMod;
		CUDA_CHECK(hipGraphDestroy(graph));
		CUDA_CHECK(hipGraphExecDestroy(exec));
		CudaCheckErrorMod;
	}

	// Wait for destination stream to finish (i.e., transfer+notify complete)
	CUDA_CHECK(hipStreamSynchronize(sDst));
	CUDA_CHECK(hipStreamSynchronize(sSrc));
	// Copy dst back to host from destination device
	CUDA_CHECK(hipMemcpy(h_dst.data(), d_dst, n * sizeof(float), hipMemcpyDeviceToHost));

	// Verify correctness
	expect_equal(h_src, h_dst);

	// Cleanup
	CUDA_CHECK(hipStreamDestroy(sSrc));
	CUDA_CHECK(hipStreamDestroy(sDst));

	CUDA_CHECK(hipSetDevice(devSrc));
	CUDA_CHECK(hipFree(d_src));

	CUDA_CHECK(hipSetDevice(devDst));
	CUDA_CHECK(hipFree(d_dst));
	CUDA_CHECK(hipFree(d_flag));
}

static void run_p2p_test_graph_parallel(int devSrc, int devDst, size_t n) {
	CUDA_CHECK(hipSetDevice(devSrc));
	float* d_src = nullptr;
	CUDA_CHECK(hipMalloc(&d_src, n * sizeof(float)));

	CUDA_CHECK(hipSetDevice(devDst));
	float* d_dst = nullptr;
	CUDA_CHECK(hipMalloc(&d_dst, n * sizeof(float)));

	FIDESlib::TimelineSemaphore* d_flag = nullptr;
	CUDA_CHECK(hipMalloc(&d_flag, sizeof(uint32_t)));
	CUDA_CHECK(hipMemset(d_flag, 0, sizeof(uint32_t)));

	// Initialize host data and copy to src device
	std::vector<float> h_src(n), h_dst(n);
	init_host_data(h_src, 1.0f);

	CUDA_CHECK(hipSetDevice(devSrc));
	CUDA_CHECK(hipMemcpy(d_src, h_src.data(), n * sizeof(float), hipMemcpyHostToDevice));

	uint32_t expectedValue = 1;

	CUDA_CHECK(hipSetDevice(devSrc));
	hipStream_t sSrc;
	CUDA_CHECK(hipStreamCreate(&sSrc));
	CUDA_CHECK(hipSetDevice(devDst));
	hipStream_t sDst;
	CUDA_CHECK(hipStreamCreate(&sDst));

#pragma omp parallel num_threads(2)
	{
		int j = omp_get_thread_num();

		if (j == 1) {
			// Launch transfer and notify kernels on src device
			CUDA_CHECK(hipSetDevice(devSrc));

			CUDA_CHECK(hipStreamBeginCapture(sSrc, hipStreamCaptureModeThreadLocal));

			CUDA_CHECK(hipSetDevice(devSrc));

			if (0) {
				int threads = 256;
				int blocks	= static_cast<int>((n + threads - 1) / threads);

				FIDESlib::p2p_transfer_1d<<<blocks, threads, 0, sSrc>>>(d_src, d_dst, n);
			} else {
				FIDESlib::transferKernel(d_src, d_dst, n, sSrc, devSrc, devDst);
			}
			CUDA_CHECK(hipGetLastError());

			if (0) {
				FIDESlib::notify_kernel_hostpin<<<1, 32, 0, sSrc>>>(d_flag, expectedValue);
			} else {
				FIDESlib::notifyKernel(d_flag, expectedValue, sSrc);
			}
			CUDA_CHECK(hipGetLastError());

			{
				hipGraph_t graph;
				CUDA_CHECK(hipStreamEndCapture(sSrc, &graph));

				hipGraphExec_t exec;
				CUDA_CHECK(hipGraphInstantiate(&exec, graph));

				CUDA_CHECK(hipGraphLaunch(exec, sSrc));
				CudaCheckErrorMod;
				CUDA_CHECK(hipGraphDestroy(graph));
				CUDA_CHECK(hipGraphExecDestroy(exec));
				CudaCheckErrorMod;
			}
		} else {
			CUDA_CHECK(hipSetDevice(devDst));
			// On destination device, wait for flag and then read back
			CUDA_CHECK(hipStreamBeginCapture(sDst, hipStreamCaptureModeThreadLocal));

			if (0) {
				FIDESlib::hostpin_polling_kernel<<<1, 32, 0, sDst>>>(d_flag, expectedValue);
			} else {
				FIDESlib::pollingKernel(d_flag, expectedValue, sDst);
			}

			CUDA_CHECK(hipGetLastError());

			{
				hipGraph_t graph;
				CUDA_CHECK(hipStreamEndCapture(sDst, &graph));

				hipGraphExec_t exec;
				CUDA_CHECK(hipGraphInstantiate(&exec, graph));

				CUDA_CHECK(hipGraphLaunch(exec, sDst));
				CudaCheckErrorMod;
				CUDA_CHECK(hipGraphDestroy(graph));
				CUDA_CHECK(hipGraphExecDestroy(exec));
				CudaCheckErrorMod;
			}
		}
	}
	// Wait for destination stream to finish (i.e., transfer+notify complete)
	CUDA_CHECK(hipStreamSynchronize(sDst));
	CUDA_CHECK(hipStreamSynchronize(sSrc));
	// Copy dst back to host from destination device
	CUDA_CHECK(hipMemcpy(h_dst.data(), d_dst, n * sizeof(float), hipMemcpyDeviceToHost));

	// Verify correctness
	expect_equal(h_src, h_dst);

	// Cleanup
	CUDA_CHECK(hipStreamDestroy(sSrc));
	CUDA_CHECK(hipStreamDestroy(sDst));

	CUDA_CHECK(hipSetDevice(devSrc));
	CUDA_CHECK(hipFree(d_src));

	CUDA_CHECK(hipSetDevice(devDst));
	CUDA_CHECK(hipFree(d_dst));
	CUDA_CHECK(hipFree(d_flag));
}

TEST(P2PTransferTest, SingleGPU) {
	int deviceCount = 0;
	CUDA_CHECK(hipGetDeviceCount(&deviceCount));
	ASSERT_GE(deviceCount, 1) << "Need at least 1 GPU";

	int dev	 = 0;
	size_t n = 1 << 20; // 1M floats

	run_p2p_test(dev, dev, n);
}

TEST(P2PTransferTest, SingleGPUGraph) {
	int deviceCount = 0;
	CUDA_CHECK(hipGetDeviceCount(&deviceCount));
	ASSERT_GE(deviceCount, 1) << "Need at least 1 GPU";

	int dev	 = 0;
	size_t n = 1 << 20; // 1M floats

	run_p2p_test_graph(dev, dev, n);
}

// 2. Two-GPU test: devSrc != devDst, with P2P if available
TEST(P2PTransferTest, MultiGPUIfAvailable) {
	int deviceCount = 0;
	CUDA_CHECK(hipGetDeviceCount(&deviceCount));
	if (deviceCount < 2) {
		GTEST_SKIP() << "Less than 2 GPUs; skipping multi-GPU test";
	}

	int devSrc = 0;
	int devDst = 1;

	int canAccessSrcToDst = 0, canAccessDstToSrc = 0;
	CUDA_CHECK(hipDeviceCanAccessPeer(&canAccessSrcToDst, devSrc, devDst));
	CUDA_CHECK(hipDeviceCanAccessPeer(&canAccessDstToSrc, devDst, devSrc));

	if (!canAccessSrcToDst || !canAccessDstToSrc) {
		GTEST_SKIP() << "P2P not available between GPU " << devSrc << " and GPU " << devDst << "; skipping";
	}

	std::cout << "\nEnabling P2P access..." << std::endl;
	safe_enable_p2p(devSrc, devDst);
	safe_enable_p2p(devDst, devSrc);

	size_t n = 1 << 20; // 1M floats
	run_p2p_test(devSrc, devDst, n);
}

TEST(P2PTransferTest, MultiGPUIfAvailableSingleGraph) {
	int deviceCount = 0;
	CUDA_CHECK(hipGetDeviceCount(&deviceCount));
	if (deviceCount < 2) {
		GTEST_SKIP() << "Less than 2 GPUs; skipping multi-GPU test";
	}

	int devSrc = 0;
	int devDst = 1;

	int canAccessSrcToDst = 0, canAccessDstToSrc = 0;
	CUDA_CHECK(hipDeviceCanAccessPeer(&canAccessSrcToDst, devSrc, devDst));
	CUDA_CHECK(hipDeviceCanAccessPeer(&canAccessDstToSrc, devDst, devSrc));

	if (!canAccessSrcToDst || !canAccessDstToSrc) {
		GTEST_SKIP() << "P2P not available between GPU " << devSrc << " and GPU " << devDst << "; skipping";
	}

	std::cout << "\nEnabling P2P access..." << std::endl;
	safe_enable_p2p(devSrc, devDst);
	safe_enable_p2p(devDst, devSrc);

	size_t n = 1 << 20; // 1M floats
	run_p2p_test_graph(devSrc, devDst, n);
}

TEST(P2PTransferTest, MultiGPUIfAvailableParallelGraphs) {
	int deviceCount = 0;
	CUDA_CHECK(hipGetDeviceCount(&deviceCount));
	if (deviceCount < 2) {
		GTEST_SKIP() << "Less than 2 GPUs; skipping multi-GPU test";
	}

	int devSrc = 0;
	int devDst = 1;

	int canAccessSrcToDst = 0, canAccessDstToSrc = 0;
	CUDA_CHECK(hipDeviceCanAccessPeer(&canAccessSrcToDst, devSrc, devDst));
	CUDA_CHECK(hipDeviceCanAccessPeer(&canAccessDstToSrc, devDst, devSrc));

	if (!canAccessSrcToDst || !canAccessDstToSrc) {
		GTEST_SKIP() << "P2P not available between GPU " << devSrc << " and GPU " << devDst << "; skipping";
	}

	std::cout << "\nEnabling P2P access..." << std::endl;
	safe_enable_p2p(devSrc, devDst);
	safe_enable_p2p(devDst, devSrc);

	size_t n = 1 << 20; // 1M floats
	run_p2p_test_graph_parallel(devSrc, devDst, n);
}

/*
struct BenchmarkResult {
	double transfer_size_mb;
	int num_blocks;
	double bandwidth_gb_s;
	double latency_ms;
	double polling_latency_us;
	bool is_p2p;  // true if multi-GPU, false if single-GPU
};
*/
struct BenchmarkResult {
	double transfer_size_mb;
	int num_blocks;
	double bandwidth_gb_s;
	double gpu_time_ms;			 // GPU-side time (via CUDA events)
	double host_time_ms;		 // Host-side time (includes sync overhead)
	double gpu_only_overhead_us; // GPU launch + polling setup overhead
	bool is_p2p;
};

std::vector<BenchmarkResult> benchmark_results;

// Benchmark: measure bandwidth and latency
static void benchmark_p2p_transfer(int devSrc, int devDst, size_t num_elements, int num_blocks, bool is_p2p, BenchmarkResult* result_) {
	BenchmarkResult& result = *result_;
	result.transfer_size_mb = ((double)num_elements * sizeof(float)) / (1024 * 1024);
	result.num_blocks		= num_blocks;
	result.is_p2p			= is_p2p;

	CUDA_CHECK(hipSetDevice(devSrc));
	float* d_src = nullptr;
	CUDA_CHECK(hipMalloc(&d_src, num_elements * sizeof(float)));

	CUDA_CHECK(hipSetDevice(devDst));
	float* d_dst = nullptr;
	CUDA_CHECK(hipMalloc(&d_dst, num_elements * sizeof(float)));

	uint32_t* d_flag = nullptr;
	CUDA_CHECK(hipMalloc(&d_flag, sizeof(uint32_t)));
	CUDA_CHECK(hipMemset(d_flag, 0, sizeof(uint32_t)));

	// Initialize src
	std::vector<float> h_src(num_elements);
	for (size_t i = 0; i < num_elements; ++i) {
		h_src[i] = static_cast<float>(i) * 0.001f;
	}

	CUDA_CHECK(hipSetDevice(devSrc));
	CUDA_CHECK(hipMemcpy(d_src, h_src.data(), num_elements * sizeof(float), hipMemcpyHostToDevice));

	// Create CUDA events for timing
	hipEvent_t e_transfer_start, e_transfer_end, e_notify_end, e_poll_end;
	CUDA_CHECK(hipSetDevice(devSrc));
	CUDA_CHECK(hipEventCreate(&e_transfer_start));
	CUDA_CHECK(hipEventCreate(&e_transfer_end));

	CUDA_CHECK(hipEventCreate(&e_notify_end));
	CUDA_CHECK(hipSetDevice(devDst));
	CUDA_CHECK(hipEventCreate(&e_poll_end));

	// Warmup run (no timing)
	{
		CUDA_CHECK(hipSetDevice(devSrc));
		hipStream_t sSrc;
		CUDA_CHECK(hipStreamCreate(&sSrc));

		int threads = 128;
		FIDESlib::p2p_transfer_1d<<<num_blocks, threads, 0, sSrc>>>(d_src, d_dst, num_elements);
		FIDESlib::notify_kernel<<<1, 32, 0, sSrc>>>(d_flag, 1);

		CUDA_CHECK(hipSetDevice(devDst));
		hipStream_t sDst;
		CUDA_CHECK(hipStreamCreate(&sDst));
		FIDESlib::p2p_polling_kernel<<<1, 32, 0, sDst>>>(d_flag, 1);
		CUDA_CHECK(hipStreamSynchronize(sDst));

		CUDA_CHECK(hipStreamDestroy(sSrc));
		CUDA_CHECK(hipStreamDestroy(sDst));
		CUDA_CHECK(hipSetDevice(devDst));
		CUDA_CHECK(hipMemset(d_flag, 0, sizeof(uint32_t)));
	}

	// Actual benchmark: multiple iterations
	const int num_iterations = 10;
	std::vector<float> gpu_times, host_times;

	CUDA_CHECK(hipSetDevice(devSrc));
	hipStream_t sSrc;
	CUDA_CHECK(hipStreamCreate(&sSrc));
	CUDA_CHECK(hipSetDevice(devDst));
	hipStream_t sDst;
	CUDA_CHECK(hipStreamCreate(&sDst));
	for (int iter = 0; iter < num_iterations; ++iter) {
		CUDA_CHECK(hipSetDevice(devDst));
		CUDA_CHECK(hipMemset(d_flag, 0, sizeof(uint32_t)));

		// Host-side timer start (includes everything)
		auto t_host_start = std::chrono::high_resolution_clock::now();

		CUDA_CHECK(hipSetDevice(devSrc));

		// GPU-side timer: transfer kernel start
		CUDA_CHECK(hipEventRecord(e_transfer_start, sSrc));

		int threads = 128;
		FIDESlib::p2p_transfer_1d<<<num_blocks, threads, 0, sSrc>>>(d_src, d_dst, num_elements);
		CUDA_CHECK(hipGetLastError());

		// GPU-side marker: transfer complete
		CUDA_CHECK(hipEventRecord(e_transfer_end, sSrc));

		uint32_t expected_value = iter + 1;
		FIDESlib::notify_kernel<<<1, 32, 0, sSrc>>>(d_flag, expected_value);
		CUDA_CHECK(hipGetLastError());

		// GPU-side marker: notify complete
		CUDA_CHECK(hipEventRecord(e_notify_end, sSrc));

		// GPU1: wait for completion
		CUDA_CHECK(hipSetDevice(devDst));

		// GPU-side marker: polling start
		// CUDA_CHECK(cudaEventRecord(e_poll_start, sDst));  // Record before poll

		FIDESlib::p2p_polling_kernel<<<1, 32, 0, sDst>>>(d_flag, expected_value);
		CUDA_CHECK(hipGetLastError());

		// GPU-side marker: polling complete
		CUDA_CHECK(hipEventRecord(e_poll_end, sDst));

		CUDA_CHECK(hipStreamSynchronize(sDst));

		// Host-side timer end
		auto t_host_end		   = std::chrono::high_resolution_clock::now();
		double host_elapsed_ms = std::chrono::duration<double, std::milli>(t_host_end - t_host_start).count();

		// GPU-side timing via events
		float transfer_ms = 0.0f, notify_ms = 0.0f;
		// float poll_ms = 0.0f;
		CUDA_CHECK(hipEventSynchronize(e_notify_end));
		CUDA_CHECK(hipEventElapsedTime(&transfer_ms, e_transfer_start, e_transfer_end));
		CUDA_CHECK(hipEventElapsedTime(&notify_ms, e_transfer_end, e_notify_end));

		// Polling time: from before poll to after poll
		// (Note: This includes busy-wait spin time)

		float total_gpu_ms = transfer_ms + notify_ms;

		gpu_times.push_back(total_gpu_ms);
		host_times.push_back(host_elapsed_ms);
	}

	// Cleanup events
	CUDA_CHECK(hipEventDestroy(e_transfer_start));
	CUDA_CHECK(hipEventDestroy(e_transfer_end));
	CUDA_CHECK(hipEventDestroy(e_notify_end));
	CUDA_CHECK(hipEventDestroy(e_poll_end));

	// Calculate averages
	double avg_gpu_time_ms	= 0.0;
	double avg_host_time_ms = 0.0;
	for (size_t i = 0; i < gpu_times.size(); ++i) {
		avg_gpu_time_ms += gpu_times[i];
		avg_host_time_ms += host_times[i];
	}
	avg_gpu_time_ms /= gpu_times.size();
	avg_host_time_ms /= host_times.size();

	// Bandwidth (GPU-side time only)
	double data_gb				= (num_elements * sizeof(float)) / 1e9;
	result.bandwidth_gb_s		= data_gb / (avg_gpu_time_ms / 1e3);
	result.gpu_time_ms			= avg_gpu_time_ms;
	result.host_time_ms			= avg_host_time_ms;
	result.gpu_only_overhead_us = (avg_host_time_ms - avg_gpu_time_ms) * 1000.0; // Overhead in μs

	// Cleanup
	CUDA_CHECK(hipSetDevice(devSrc));
	CUDA_CHECK(hipFree(d_src));

	CUDA_CHECK(hipSetDevice(devDst));
	CUDA_CHECK(hipFree(d_dst));
	CUDA_CHECK(hipFree(d_flag));
}

// Benchmark test: single GPU, vary block count and transfer size
TEST(P2PBenchmark, SingleGPUVariableBlocksAndSizes) {
	int deviceCount = 0;
	CUDA_CHECK(hipGetDeviceCount(&deviceCount));
	ASSERT_GE(deviceCount, 1) << "Need at least 1 GPU";

	int dev = 0;
	CUDA_CHECK(hipSetDevice(dev));

	// Transfer sizes: 1 MB, 10 MB, 50 MB, 100 MB, 256 MB, 512 MB
	std::vector<double> transfer_sizes_mb = { 0.001, 0.25, 0.5, 1, 10, 50, 100, 256, 512 };
	// Block counts: 1, 2, 4, 8, 16, 32
	std::vector<int> block_counts = { 1, 2, 4, 8, 16, 32, 64, 128 };
	std::cout << "\n=== SINGLE GPU BENCHMARK (with CUDA Events) ===" << std::endl;
	std::cout << std::left << std::setw(15) << "Transfer(MB)" << std::setw(12) << "Blocks" << std::setw(18) << "Bandwidth(GB/s)" << std::setw(16)
			  << "GPU Time(ms)" << std::setw(16) << "Host Time(ms)" << std::setw(18) << "Overhead(μs)" << std::endl;
	std::cout << std::string(95, '-') << std::endl;

	for (double size_mb : transfer_sizes_mb) {
		for (int blocks : block_counts) {
			size_t num_elements = (size_mb * 1024 * 1024) / sizeof(float);
			BenchmarkResult result;
			benchmark_p2p_transfer(dev, dev, num_elements, blocks, false, &result);

			std::cout << std::left << std::setw(15) << result.transfer_size_mb << std::setw(12) << result.num_blocks << std::setw(18) << std::fixed
					  << std::setprecision(4) << result.bandwidth_gb_s << std::setw(16) << result.gpu_time_ms << std::setw(16) << result.host_time_ms
					  << std::setw(18) << result.gpu_only_overhead_us << std::endl;

			benchmark_results.push_back(result);
		}
	}
}

// Benchmark test: multi GPU with P2P if available
TEST(P2PBenchmark, MultiGPUVariableBlocksAndSizes) {
	int deviceCount = 0;
	CUDA_CHECK(hipGetDeviceCount(&deviceCount));
	if (deviceCount < 2) {
		GTEST_SKIP() << "Less than 2 GPUs; skipping multi-GPU benchmark";
	}

	int devSrc = 0;
	int devDst = 1;

	int canAccessSrcToDst = 0, canAccessDstToSrc = 0;
	CUDA_CHECK(hipDeviceCanAccessPeer(&canAccessSrcToDst, devSrc, devDst));
	CUDA_CHECK(hipDeviceCanAccessPeer(&canAccessDstToSrc, devDst, devSrc));

	if (!canAccessSrcToDst || !canAccessDstToSrc) {
		GTEST_SKIP() << "P2P not available between GPU " << devSrc << " and GPU " << devDst << "; skipping";
	}

	std::cout << "\nEnabling P2P access..." << std::endl;
	safe_enable_p2p(devSrc, devDst);
	safe_enable_p2p(devDst, devSrc);

	std::vector<double> transfer_sizes_mb = { 0.001, 0.25, 0.5, 1, 10, 50, 100, 256, 512 };
	std::vector<int> block_counts		  = { 1, 2, 4, 8, 16, 32, 64, 128 };

	std::cout << "\n=== MULTI-GPU P2P BENCHMARK (with CUDA Events) ===" << std::endl;
	std::cout << std::left << std::setw(15) << "Transfer(MB)" << std::setw(12) << "Blocks" << std::setw(18) << "Bandwidth(GB/s)" << std::setw(16)
			  << "GPU Time(ms)" << std::setw(16) << "Host Time(ms)" << std::setw(18) << "Overhead(μs)" << std::endl;
	std::cout << std::string(95, '-') << std::endl;

	for (double size_mb : transfer_sizes_mb) {
		for (int blocks : block_counts) {
			size_t num_elements = (size_mb * 1024 * 1024) / sizeof(float);
			BenchmarkResult result;
			benchmark_p2p_transfer(devSrc, devDst, num_elements, blocks, false, &result);

			std::cout << std::left << std::setw(15) << result.transfer_size_mb << std::setw(12) << result.num_blocks << std::setw(18) << std::fixed
					  << std::setprecision(4) << result.bandwidth_gb_s << std::setw(16) << result.gpu_time_ms << std::setw(16) << result.host_time_ms
					  << std::setw(18) << result.gpu_only_overhead_us << std::endl;

			benchmark_results.push_back(result);
		}
	}
}

TEST(APIbench, APIbenchmark) {
	FIDESlib::Stream s1, s2;
	s1.init();
	s2.init();
	/*
	for (int chunk_size = 1; chunk_size <= 1024; chunk_size *= 2) {
		for (int total_size = chunk_size; total_size <= 8 * 1024; total_size *= 2) {
			std::vector<char*> chunks(total_size / chunk_size);
			for (int i = 0; i < total_size / chunk_size; ++i) {
				cudaMallocAsync(&chunks[i], chunk_size * 1024 * 1024, s1.ptr());
				//ncclMemAlloc((void**)&chunks[i], chunk_size * 1024 * 1024);
				std::vector<char> mem(chunk_size * 1024 * 1024, 0);
				//for (auto j : mem)
				//    j = rand() % 256;
				cudaMemcpyAsync(chunks[i], mem.data(), chunk_size * 1024 * 1024, cudaMemcpyDefault, s1.ptr());
			}
			char* aux;
			std::string src = "Hola buenaass\n";
			src.resize(1024);
			cudaDeviceSynchronize();
			auto start = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < 10000; ++i) {
				cudaMallocAsync(&aux, 1 * 1024 * 1024, s1.ptr());
				cudaMemcpyAsync(aux, src.data(), 532, cudaMemcpyDefault, s1.ptr());
				s2.wait(s1);
				FIDESlib::dummy_kernel<<<32, 32, 0, s2.ptr()>>>();
				cudaFreeAsync(aux, s2.ptr());
				s1.wait(s2);
				FIDESlib::dummy_kernel<<<32, 32, 0, s1.ptr()>>>();
			}
			auto end = std::chrono::high_resolution_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
			std::cout << total_size << " " << chunk_size << " " << elapsed.count() << std::endl;

			cudaDeviceSynchronize();
			for (int i = 0; i < total_size / chunk_size; ++i) {
				cudaFreeAsync(chunks[i], s1.ptr());
				//ncclMemFree(chunks[i]);
			}
		}
	}

	for (int total_size = 1; total_size <= 8 * 1024; total_size *= 2) {
		std::vector<char*> chunks;
		int rem = total_size;
		while (rem > 0) {
			int size = 1 + (rand() % (std::min(rem, 128)));
			chunks.push_back(nullptr);
			cudaMallocAsync(&chunks.back(), size * 1024 * 1024, s1.ptr());
			rem -= size;
		}

		cudaDeviceSynchronize();
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 10000; ++i) {
			s2.wait(s1);
			FIDESlib::dummy_kernel<<<32, 32, 0, s2.ptr()>>>();
			s1.wait(s2);
			FIDESlib::dummy_kernel<<<32, 32, 0, s1.ptr()>>>();
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		std::cout << total_size << " " << elapsed.count() << std::endl;

		cudaDeviceSynchronize();
		for (auto i : chunks) {
			cudaFreeAsync(i, s1.ptr());
		}
	}
	*/
	int chunk_size = 1;

	bool SINGLE_STREAM = false;
	bool SINGLE_ALLOC  = false;
	for (int total_size = 1; total_size <= 64 * 1024; total_size *= 2) {
		std::vector<FIDESlib::Stream> chunks_(total_size);
		std::vector<char*> chunks(total_size);

		for (int i = 0; i < total_size; ++i) {
			chunks_[SINGLE_STREAM ? 0 : i].init();
		}

		for (int i = 0; i < total_size; ++i) {
			if (i > 0 && SINGLE_ALLOC)
				continue;
			hipMallocAsync(&chunks[i], chunk_size * 1024, s1.ptr());
			// ncclMemAlloc((void**)&chunks[i], chunk_size * 1024 * 1024);
			std::vector<char> mem(chunk_size, 0);
			// for (auto j : mem)
			//     j = rand() % 256;
			hipMemcpyAsync(chunks[i], mem.data(), chunk_size * 1024, hipMemcpyDefault, s1.ptr());
		}

		hipDeviceSynchronize();
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 10000; ++i) {
			int sel = rand() % total_size;
			chunks_[SINGLE_STREAM ? 0 : sel].wait(s1);
			FIDESlib::dummy_kernel<<<32, 32, 0, chunks_[SINGLE_STREAM ? 0 : sel].ptr()>>>();
			s1.wait(chunks_[SINGLE_STREAM ? 0 : sel]);
			FIDESlib::dummy_kernel<<<32, 32, 0, s1.ptr()>>>();
		}
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		std::cout << total_size << " " << elapsed.count() << std::endl;
		for (int i = 0; i < total_size; ++i) {
			if (i > 0 && SINGLE_ALLOC)
				continue;
			hipFreeAsync(chunks[i], s1.ptr());
			// ncclMemFree(chunks[i]);
		}
		hipDeviceSynchronize();
	}
}

// Export results to CSV
class BenchmarkEnvironment : public ::testing::Environment {
  public:
	void TearDown() override {
		std::ofstream csv("p2p_benchmark_results.csv");
		csv << "transfer_size_mb,num_blocks,bandwidth_gb_s,gpu_time_ms,"
			   "host_time_ms,overhead_us,is_p2p"
			<< std::endl;

		for (const auto& result : benchmark_results) {
			csv << result.transfer_size_mb << "," << result.num_blocks << "," << std::fixed << std::setprecision(6) << result.bandwidth_gb_s << ","
				<< result.gpu_time_ms << "," << result.host_time_ms << "," << result.gpu_only_overhead_us << "," << (result.is_p2p ? 1 : 0) << std::endl;
		}
		csv.close();

		std::cout << "\nResults written to: p2p_benchmark_results.csv" << std::endl;
	}
};
