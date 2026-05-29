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
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   test_grid.cu
 *  @author Pierre Wilmot and Thomas Müller, NVIDIA
 *  @brief  Test basic aspects of GridEncoding
 */

#include "test_common.h"

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/encodings/multi_level_interface.h>

#include <cmath>
#include <vector>

using namespace tcnn;

#if TCNN_HAS_CUDA_BF16 && TCNN_MIN_GPU_ARCH >= 80
__global__ void fill_bf16_grid_test(uint32_t n, __nv_bfloat16* data, float value) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		data[i] = (__nv_bfloat16)value;
	}
}

__global__ void fill_bf16_grid_test_sequence(uint32_t n, __nv_bfloat16* data) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		data[i] = (__nv_bfloat16)(0.01f + 0.001f * (float)(i % 17));
	}
}

bool has_only_finite_values_and_any_nonzero_grid_test(const std::vector<__nv_bfloat16>& values) {
	bool any_nonzero = false;
	for (const auto value : values) {
		const float v = __bfloat162float(value);
		if (!std::isfinite(v)) {
			return false;
		}
		any_nonzero |= std::abs(v) > 0.0f;
	}

	return any_nonzero;
}
#endif

TEST_CASE("GridEncoding sanity checks", "[encoding]") {
	tcnn_test_setup();

	const char* config = R"({
		"otype": "Grid",
		"base_resolution": 32,
		"log2_hashmap_size": 16,
		"n_features_per_level": 2,
		"n_levels": 20,
		"otype": "HashGrid",
		"per_level_scale": 1.5
	})";
	nlohmann::json config_json = nlohmann::json::parse(config);
	std::unique_ptr<MultiLevelEncoding<float>> g{create_grid_encoding<float>(3, config_json)};

	REQUIRE(g);

	REQUIRE(g->n_pos_dims() == 3);
	REQUIRE(g->n_features_per_level() == 2);
	REQUIRE(g->padded_output_width() == 20 /* levels */ * 2 /* nb_fetures_per_level */);

	// Level 0 is a dense layer, 3 dimensions with resilution 32
	REQUIRE(g->level_n_params(0) == 32 * 32 * 32);
	REQUIRE(g->level_params_offset(0) == 0);
	// Level 1 is a hash layer as 48 * 48 * 48 > 2 ^ log2_hashmap_size (65536)
	REQUIRE(g->level_n_params(1) == 65536);
	REQUIRE(g->level_params_offset(1) == 32 * 32 * 32);
	// Level 2 is a hash layer as 72 * 72 * 72 > 2 ^ log2_hashmap_size (65536)
	REQUIRE(g->level_n_params(2) == 65536);
	REQUIRE(g->level_params_offset(2) == 32 * 32 * 32 + 65536);

	// Parameters are NOT an encapsulated member of GridEncoding.
	// We need to allocate them manually and set them.
	size_t n_params = g->n_params();
	REQUIRE(n_params == 2555904);
	GPUMemory<char> params_buffer;
	params_buffer.resize(sizeof(float) * n_params);
	float* params = (float*)(params_buffer.data());
	float* inference_params = params;
	float* gradients = nullptr;
	// Using the same values for params and inference params in this test, not setting the rest.
	g->set_params(params, inference_params, gradients);

	unsigned int batch_size = BATCH_SIZE_GRANULARITY;
	GPUMatrix<float> input(g->n_pos_dims(), batch_size);
	GPUMatrix<float> output(g->padded_output_width(), batch_size);
	input.memset(0);

	REQUIRE(input.n_elements() == 3 /* dimensions */ * batch_size);
	REQUIRE(output.n_elements() == 2 /* feture per level */ * 20 /* levels*/ * batch_size);

	std::unique_ptr<Context> c = g->forward(input, &output);

	std::vector<float> result_host(output.n_elements());
	CUDA_CHECK_THROW(cudaMemcpy(result_host.data(), output.data(), output.n_bytes(), cudaMemcpyDeviceToHost));
}


TEST_CASE("GridEncoding supports bf16 forward and backward", "[encoding][bf16]") {
#if TCNN_HAS_CUDA_BF16 && TCNN_MIN_GPU_ARCH >= 80
	using T = __nv_bfloat16;

	int n_devices = 0;
	CUDA_CHECK_THROW(cudaGetDeviceCount(&n_devices));
	int bf16_device = -1;
	for (int i = 0; i < n_devices; ++i) {
		cudaDeviceProp props;
		CUDA_CHECK_THROW(cudaGetDeviceProperties(&props, i));
		if (props.major * 10 + props.minor == TCNN_MIN_GPU_ARCH && props.major >= 8) {
			bf16_device = i;
			break;
		}
	}

	if (bf16_device < 0) {
		SUCCEED("No CUDA device matching TCNN_MIN_GPU_ARCH is available for bf16 grid encoding.");
		return;
	}

	CUDA_CHECK_THROW(cudaSetDevice(bf16_device));
	tcnn_test_setup();

	nlohmann::json config_json = nlohmann::json::parse(R"({
		"otype": "HashGrid",
		"base_resolution": 4,
		"log2_hashmap_size": 4,
		"n_features_per_level": 2,
		"n_levels": 2,
		"per_level_scale": 2.0,
		"interpolation": "Linear"
	})");

	std::unique_ptr<MultiLevelEncoding<T>> g{create_grid_encoding<T>(3, config_json)};
	REQUIRE(g);

	GPUMemory<T> params{g->n_params()};
	GPUMemory<T> gradients{g->n_params()};
	fill_bf16_grid_test_sequence<<<n_blocks_linear(params.size()), N_THREADS_LINEAR>>>((uint32_t)params.size(), params.data());
	CUDA_CHECK_THROW(cudaMemset(gradients.data(), 0, gradients.get_bytes()));
	CUDA_CHECK_THROW(cudaGetLastError());
	g->set_params(params.data(), params.data(), gradients.data());

	const uint32_t batch_size = BATCH_SIZE_GRANULARITY;
	GPUMatrix<float> input{g->n_pos_dims(), batch_size};
	GPUMatrix<T> output{g->padded_output_width(), batch_size};
	GPUMatrix<T> output_gradient{g->padded_output_width(), batch_size};
	GPUMatrix<float> input_gradient{g->n_pos_dims(), batch_size};

	input.memset(0);
	fill_bf16_grid_test<<<n_blocks_linear(output_gradient.n_elements()), N_THREADS_LINEAR>>>(
		output_gradient.n_elements(),
		output_gradient.data(),
		1.0f
	);
	CUDA_CHECK_THROW(cudaMemset(output.data(), 0, output.n_bytes()));
	CUDA_CHECK_THROW(cudaMemset(input_gradient.data(), 0, input_gradient.n_bytes()));
	CUDA_CHECK_THROW(cudaGetLastError());

	std::unique_ptr<Context> ctx;
	REQUIRE_NOTHROW(ctx = g->forward(input, &output, false, true));
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
	REQUIRE(has_only_finite_values_and_any_nonzero_grid_test(output.to_cpu_vector()));

#if defined(TCNN_RTC)
	REQUIRE_NOTHROW(g->backward(*ctx, input, output, output_gradient, &input_gradient, false, GradientMode::Overwrite));
#else
	REQUIRE_NOTHROW(g->backward_impl(nullptr, *ctx, input, output, output_gradient, &input_gradient, false, GradientMode::Overwrite));
#endif
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
	std::vector<T> gradient_values(g->n_params());
	gradients.copy_to_host(gradient_values);
	REQUIRE(has_only_finite_values_and_any_nonzero_grid_test(gradient_values));

#else
	SUCCEED("bf16 grid encoding requires CUDA bf16 support and TCNN_MIN_GPU_ARCH >= 80.");
#endif
}
