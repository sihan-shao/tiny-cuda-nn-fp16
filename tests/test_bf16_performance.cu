/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 * WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   test_bf16_performance.cu
 *  @brief  Compare compiled half and bf16 FullyFusedMLP/GridEncoding training kernel times.
 */

#include "test_common.h"

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/encodings/multi_level_interface.h>
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>

using namespace tcnn;

#if TCNN_HAS_CUDA_BF16 && TCNN_MIN_GPU_ARCH >= 80
namespace {

bool select_matching_bf16_device() {
	int n_devices = 0;
	CUDA_CHECK_THROW(cudaGetDeviceCount(&n_devices));
	for (int i = 0; i < n_devices; ++i) {
		cudaDeviceProp props;
		CUDA_CHECK_THROW(cudaGetDeviceProperties(&props, i));
		if (props.major * 10 + props.minor == TCNN_MIN_GPU_ARCH && props.major >= 8) {
			CUDA_CHECK_THROW(cudaSetDevice(i));
			return true;
		}
	}

	return false;
}

template <typename T>
__global__ void fill_value(uint32_t n, T* data, float value) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		data[i] = (T)value;
	}
}

template <typename T>
__global__ void fill_sequence(uint32_t n, T* data) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		data[i] = (T)(0.01f + 0.001f * (float)(i % 17));
	}
}

template <typename F>
float benchmark_ms(F&& f, uint32_t warmup_iterations, uint32_t timed_iterations) {
	for (uint32_t i = 0; i < warmup_iterations; ++i) {
		f();
	}
	CUDA_CHECK_THROW(cudaDeviceSynchronize());

	cudaEvent_t start, stop;
	CUDA_CHECK_THROW(cudaEventCreate(&start));
	CUDA_CHECK_THROW(cudaEventCreate(&stop));
	CUDA_CHECK_THROW(cudaEventRecord(start));
	for (uint32_t i = 0; i < timed_iterations; ++i) {
		f();
	}
	CUDA_CHECK_THROW(cudaEventRecord(stop));
	CUDA_CHECK_THROW(cudaEventSynchronize(stop));
	float elapsed_ms = 0.0f;
	CUDA_CHECK_THROW(cudaEventElapsedTime(&elapsed_ms, start, stop));
	CUDA_CHECK_THROW(cudaEventDestroy(start));
	CUDA_CHECK_THROW(cudaEventDestroy(stop));
	return elapsed_ms / (float)timed_iterations;
}

template <typename T>
float benchmark_fully_fused_mlp_train_ms() {
	constexpr uint32_t WIDTH = 64;
	constexpr uint32_t BATCH_SIZE = 32 * BATCH_SIZE_GRANULARITY;

	FullyFusedMLP<T, WIDTH> mlp{WIDTH, WIDTH, 2, Activation::ReLU, Activation::None};

	GPUMemory<T> params{mlp.n_params()};
	GPUMemory<T> gradients{mlp.n_params()};
	fill_sequence<<<n_blocks_linear((uint32_t)params.size()), N_THREADS_LINEAR>>>((uint32_t)params.size(), params.data());
	CUDA_CHECK_THROW(cudaMemset(gradients.data(), 0, gradients.get_bytes()));
	CUDA_CHECK_THROW(cudaGetLastError());
	mlp.set_params(params.data(), params.data(), gradients.data());

	GPUMatrix<T> input{WIDTH, BATCH_SIZE};
	GPUMatrix<T> output{WIDTH, BATCH_SIZE};
	GPUMatrix<T> output_gradient{WIDTH, BATCH_SIZE};
	GPUMatrix<T> input_gradient{WIDTH, BATCH_SIZE};
	fill_value<<<n_blocks_linear(input.n_elements()), N_THREADS_LINEAR>>>(input.n_elements(), input.data(), 0.25f);
	fill_value<<<n_blocks_linear(output_gradient.n_elements()), N_THREADS_LINEAR>>>(output_gradient.n_elements(), output_gradient.data(), 1.0f);
	CUDA_CHECK_THROW(cudaGetLastError());

	auto train_step = [&]() {
		auto ctx = mlp.forward(input, &output, false, true);
		mlp.backward_impl(nullptr, *ctx, input, output, output_gradient, &input_gradient, false, GradientMode::Overwrite);
	};

	return benchmark_ms(train_step, 8, 32);
}

template <typename T>
float benchmark_grid_train_ms() {
	constexpr uint32_t BATCH_SIZE = 64 * BATCH_SIZE_GRANULARITY;

	nlohmann::json config = nlohmann::json::parse(R"({
		"otype": "HashGrid",
		"base_resolution": 16,
		"log2_hashmap_size": 18,
		"n_features_per_level": 2,
		"n_levels": 16,
		"per_level_scale": 1.5,
		"interpolation": "Linear"
	})");

	std::unique_ptr<MultiLevelEncoding<T>> grid{create_grid_encoding<T>(3, config)};
	REQUIRE(grid);

	GPUMemory<T> params{grid->n_params()};
	GPUMemory<T> gradients{grid->n_params()};
	fill_sequence<<<n_blocks_linear((uint32_t)params.size()), N_THREADS_LINEAR>>>((uint32_t)params.size(), params.data());
	CUDA_CHECK_THROW(cudaMemset(gradients.data(), 0, gradients.get_bytes()));
	CUDA_CHECK_THROW(cudaGetLastError());
	grid->set_params(params.data(), params.data(), gradients.data());

	pcg32 rng{1337};
	GPUMatrix<float> input{grid->n_pos_dims(), BATCH_SIZE};
	GPUMatrix<T> output{grid->padded_output_width(), BATCH_SIZE};
	GPUMatrix<T> output_gradient{grid->padded_output_width(), BATCH_SIZE};
	GPUMatrix<float> input_gradient{grid->n_pos_dims(), BATCH_SIZE};
	input.initialize_uniform(rng, 0.001f, 0.999f);
	fill_value<<<n_blocks_linear(output_gradient.n_elements()), N_THREADS_LINEAR>>>(output_gradient.n_elements(), output_gradient.data(), 1.0f);
	CUDA_CHECK_THROW(cudaGetLastError());

	auto train_step = [&]() {
		auto ctx = grid->forward(input, &output, false, true);
		grid->backward_impl(nullptr, *ctx, input, output, output_gradient, &input_gradient, false, GradientMode::Overwrite);
	};

	return benchmark_ms(train_step, 4, 16);
}

void require_close_performance(const std::string& name, float half_ms, float bf16_ms, float max_ratio) {
	const float ratio = bf16_ms / std::max(half_ms, 1.0e-6f);
	std::cout << name << " half=" << half_ms << " ms bf16=" << bf16_ms << " ms ratio=" << ratio << std::endl;
	INFO(name << " half=" << half_ms << " ms bf16=" << bf16_ms << " ms ratio=" << ratio);
	REQUIRE(ratio <= max_ratio);
}

}
#endif

TEST_CASE("Compiled bf16 kernels stay close to half performance", "[bf16][performance]") {
#if TCNN_HAS_CUDA_BF16 && TCNN_MIN_GPU_ARCH >= 80
	if (!select_matching_bf16_device()) {
		SUCCEED("No CUDA device matching TCNN_MIN_GPU_ARCH is available for bf16 performance comparison.");
		return;
	}

	tcnn_test_setup();

	const float mlp_half_ms = benchmark_fully_fused_mlp_train_ms<__half>();
	const float mlp_bf16_ms = benchmark_fully_fused_mlp_train_ms<__nv_bfloat16>();
	require_close_performance("FullyFusedMLP train", mlp_half_ms, mlp_bf16_ms, 1.5f);

	const float grid_half_ms = benchmark_grid_train_ms<__half>();
	const float grid_bf16_ms = benchmark_grid_train_ms<__nv_bfloat16>();
	const float max_grid_ratio = TCNN_MIN_GPU_ARCH >= 90 ? 1.5f : 2.5f;
	require_close_performance("GridEncoding train", grid_half_ms, grid_bf16_ms, max_grid_ratio);
#else
	SUCCEED("bf16 performance comparison requires CUDA bf16 support and TCNN_MIN_GPU_ARCH >= 80.");
#endif
}
