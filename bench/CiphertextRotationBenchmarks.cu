// Portions copyright(c) 2025 Universidad de Murcia
// Portions copyright(c) 2026 LG Electronics, Inc.
//
//  Licensed under the MIT License (the "License"); you may not use this file
//  except in compliance with the License.
//
//  You may obtain a copy of the License in the LICENSE file at the project
//  root or at
//
//  https://mit-license.org/
//
//  SPDX-License-Identifier: MIT

//
// Created by carlosad on 5/11/24.
//
#include <benchmark/benchmark.h>

#include "Benchmark.cuh"
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

namespace FIDESlib::Benchmarks {
BENCHMARK_DEFINE_F(GeneralFixture, CiphertextRotation)(benchmark::State& state) {
	if (this->generalTestParams.multDepth < static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L < level");
		return;
	}

	int devcount = -1;
	hipGetDeviceCount(&devcount);
	std::vector<int> GPUs = generalTestParams.GPUs;

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	size_t batch_size = static_cast<size_t>(state.range(2));
	std::vector<double> x1;
	for (size_t i = 0; i < batch_size; ++i) {
		x1.push_back(0.25 * (i + 1)); // Generate: 0.25, 0.5, 0.75, 1.0, ...
	}

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	ptxt1->SetLevel(state.range(3));
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	// Ensure ciphertext uses the requested level (drop extra DCRT towers so
	// GetRawCipherText picks the correct number of limbs)
	if (c1) {
		size_t currentTowers = c1->GetElements()[0].GetNumOfElements();
		size_t currentLevel	 = c1->GetLevel();
		size_t totalPrimes	 = currentTowers + currentLevel;
		size_t targetTowers	 = totalPrimes - static_cast<size_t>(state.range(3));
		if (currentTowers > targetTowers) {
			size_t towersToDrop = currentTowers - targetTowers;
			for (auto& elem : c1->GetElements()) {
				elem.DropLastElements(towersToDrop);
			}
		}
		c1->SetLevel(static_cast<size_t>(state.range(3)));
	}

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

	CKKS::GenAndAddRotationKeys(cc, keys, GPUcc, { 1 });
	state.counters["config"] = state.range(0);

	state.counters["p_batch"] = state.range(2);
	state.counters["limbs"]	  = GPUcc->param.L - state.range(3);
	CudaCheckErrorMod;
	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++) {
			GPUct1.rotate(1, true);
		}
		auto end_cpu = std::chrono::high_resolution_clock::now();
		CudaCheckErrorMod;
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count() / 100);
		state.counters["cpu"] = std::chrono::duration_cast<std::chrono::duration<double>>(end_cpu - start).count() / 100;
		state.counters["gpu"] = elapsed.count() / 100;
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, CiphertextRotationCPU)(benchmark::State& state) {
	if (this->generalTestParams.multDepth < static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L < level");
		return;
	}

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);

	// Create vector that respects the actual batch size (CPU version)
	size_t batch_size = static_cast<size_t>(state.range(2));
	std::vector<double> x1;
	for (size_t i = 0; i < batch_size; ++i) {
		x1.push_back(0.25 * (i + 1)); // Generate: 0.25, 0.5, 0.75, 1.0, ...
	}

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	ptxt1->SetLevel(state.range(3));
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

	// Generate rotation key for step 1 (similar to GPU version)
	cc->EvalRotateKeyGen(keys.secretKey, { 1 });

	for (auto _ : state) {
		auto ct1_clone = c1->Clone();

		auto start	= std::chrono::high_resolution_clock::now();
		auto result = cc->EvalRotate(ct1_clone, 1);
		auto end	= std::chrono::high_resolution_clock::now();

		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
}

BENCHMARK_DEFINE_F(GeneralFixture, CiphertextHoistedRotation)(benchmark::State& state) {
	if (this->generalTestParams.multDepth < static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L < level");
		return;
	}

	int devcount = -1;
	hipGetDeviceCount(&devcount);
	std::vector<int> GPUs = generalTestParams.GPUs;

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	size_t batch_size = static_cast<size_t>(state.range(2));
	std::vector<double> x1;
	for (size_t i = 0; i < batch_size; ++i) {
		x1.push_back(0.25 * (i + 1)); // Generate: 0.25, 0.5, 0.75, 1.0, ...
	}

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	ptxt1->SetLevel(state.range(3));
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	// Ensure ciphertext uses the requested level (drop extra DCRT towers so
	// GetRawCipherText picks the correct number of limbs)
	if (c1) {
		size_t currentTowers = c1->GetElements()[0].GetNumOfElements();
		size_t currentLevel	 = c1->GetLevel();
		size_t totalPrimes	 = currentTowers + currentLevel;
		size_t targetTowers	 = totalPrimes - static_cast<size_t>(state.range(3));
		if (currentTowers > targetTowers) {
			size_t towersToDrop = currentTowers - targetTowers;
			for (auto& elem : c1->GetElements()) {
				elem.DropLastElements(towersToDrop);
			}
		}
		c1->SetLevel(static_cast<size_t>(state.range(3)));
	}

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw1);
	FIDESlib::CKKS::Ciphertext GPUct3(GPUcc, raw1);
	FIDESlib::CKKS::Ciphertext GPUct4(GPUcc, raw1);

	CKKS::GenAndAddRotationKeys(cc, keys, GPUcc, { 1, 2, 3, 4 });
	state.counters["config"] = state.range(0);

	state.counters["p_batch"] = state.range(2);
	state.counters["limbs"]	  = GPUcc->param.L - state.range(3);
	CudaCheckErrorMod;
	// hoisted rotation runs multiple rotations per call; normalize timings per rotation
	std::vector<int> rot_indexes					  = { 1, 2, 3, 4 };
	std::vector<FIDESlib::CKKS::Ciphertext*> rot_outs = { &GPUct2, &GPUct3, &GPUct4, &GPUct1 };
	int nrot										  = static_cast<int>(rot_indexes.size());
	state.counters["rotations"]						  = nrot;
	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++) {
			GPUct1.rotate_hoisted(rot_indexes, rot_outs, false);
		}
		auto end_cpu = std::chrono::high_resolution_clock::now();
		CudaCheckErrorMod;
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		// report per-rotation timings (divide by inner loop count and number of rotations)
		state.SetIterationTime(elapsed.count() / (100 * nrot));
		state.counters["cpu"] = std::chrono::duration_cast<std::chrono::duration<double>>(end_cpu - start).count() / (100 * nrot);
		state.counters["gpu"] = elapsed.count() / (100 * nrot);
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, CiphertextHoistedRotationCPU)(benchmark::State& state) {
	if (this->generalTestParams.multDepth < static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L < level");
		return;
	}

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);

	// Create vector that respects the actual batch size (Hoisted CPU version)
	size_t batch_size = static_cast<size_t>(state.range(2));
	std::vector<double> x1;
	for (size_t i = 0; i < batch_size; ++i) {
		x1.push_back(0.25 * (i + 1)); // Generate: 0.25, 0.5, 0.75, 1.0, ...
	}

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	ptxt1->SetLevel(state.range(3));
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

	// Generate rotation keys for multiple steps (similar to GPU hoisted version)
	cc->EvalRotateKeyGen(keys.secretKey, { 1, 2, 3, 4 });

	for (auto _ : state) {
		auto ct1_clone = c1->Clone();
		auto ct2_clone = c1->Clone();
		auto ct3_clone = c1->Clone();
		auto ct4_clone = c1->Clone();

		auto start = std::chrono::high_resolution_clock::now();
		// Simulate hoisted rotation by performing multiple rotations
		auto result1 = cc->EvalRotate(ct1_clone, 1);
		auto result2 = cc->EvalRotate(ct2_clone, 2);
		auto result3 = cc->EvalRotate(ct3_clone, 3);
		auto result4 = cc->EvalRotate(ct4_clone, 4);
		auto end	 = std::chrono::high_resolution_clock::now();

		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
	}
}

BENCHMARK_DEFINE_F(GeneralFixture, CiphertextRotateAndAccumulate)(benchmark::State& state) {
	if (this->generalTestParams.multDepth < static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L < level");
		return;
	}

	int devcount = -1;
	hipGetDeviceCount(&devcount);
	std::vector<int> GPUs = generalTestParams.GPUs;

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	ptxt1->SetLevel(state.range(3));
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

	int bstep				 = state.range(4);
	std::vector<int> indexes = FIDESlib::CKKS::GetAccumulateRotationIndices(bstep, 1, GPUcc->N / 2);
	FIDESlib::CKKS::GenAndAddRotationKeys(cc, keys, GPUcc, indexes);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	state.counters["config"] = state.range(0);

	state.counters["p_batch"] = state.range(2);
	state.counters["limbs"]	  = GPUcc->param.L - state.range(3);
	CudaCheckErrorMod;
	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++) {
			FIDESlib::CKKS::Accumulate(GPUct1, bstep, 1, GPUcc->N / 2);
		}
		auto end_cpu = std::chrono::high_resolution_clock::now();
		CudaCheckErrorMod;
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count() / 100);
		state.counters["cpu"] = std::chrono::duration_cast<std::chrono::duration<double>>(end_cpu - start).count() / 100;
		state.counters["gpu"] = elapsed.count() / 100;
	}
	CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(GeneralFixture, CiphertextRotation)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG });
BENCHMARK_REGISTER_F(GeneralFixture, CiphertextRotationCPU)->ArgsProduct({ { 0, 1, 2, 3 }, { 0 }, BATCH_CONFIG, LEVEL_CONFIG })->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, CiphertextHoistedRotation)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG });
BENCHMARK_REGISTER_F(GeneralFixture, CiphertextHoistedRotationCPU)->ArgsProduct({ { 0, 1, 2, 3 }, { 0 }, { 2, 6, 12 }, LEVEL_CONFIG })->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, CiphertextRotateAndAccumulate)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG, { 2, 4, 8 } });

// TDPS Experiments

BENCHMARK_REGISTER_F(GeneralFixture, CiphertextRotation)->ArgsProduct({ { 30, 31 }, { 0 }, BATCH_CONFIG, { 0, 12 } })->UseManualTime()->Iterations(10);
BENCHMARK_REGISTER_F(GeneralFixture, CiphertextHoistedRotation)->ArgsProduct({ { 30, 31 }, { 0 }, BATCH_CONFIG, { 0, 12 } })->UseManualTime()->Iterations(10);

} // namespace FIDESlib::Benchmarks