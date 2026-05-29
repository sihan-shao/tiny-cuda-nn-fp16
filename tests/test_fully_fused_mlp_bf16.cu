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
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   test_fully_fused_mlp_bf16.cu
 *  @brief  Compile and launch coverage for bf16 FullyFusedMLP fused forward/backward paths.
 */

#include "test_common.h"

#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>

#include <cuda_bf16.h>

#include <cmath>
#include <string>
#include <vector>

using namespace tcnn;

#if TCNN_MIN_GPU_ARCH >= 80

__global__ void fill_bf16(uint32_t n, __nv_bfloat16* data, float value) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		data[i] = (__nv_bfloat16)value;
	}
}

bool has_only_finite_values_and_any_nonzero(const std::vector<__nv_bfloat16>& values) {
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

TEST_CASE("bf16 RTC MLP codegen uses tensor-core MMA path", "[network][bf16][jit]") {
#if TCNN_HAS_CUDA_BF16 && TCNN_MIN_GPU_ARCH >= 80
	const std::string fwd = generate_mlp_device_code<__nv_bfloat16>(16, 16, 16, 16, 2, Activation::ReLU, Activation::None);
	REQUIRE(fwd.find("bf16_mma_vec") != std::string::npos);
	REQUIRE(fwd.find("bf16_mma_mat") != std::string::npos);
	REQUIRE(fwd.find("float acc") == std::string::npos);

	const std::string bwd = generate_backward_mlp_device_code<__nv_bfloat16>(128, 16, 16, 16, 16, 2, Activation::ReLU, Activation::None);
	REQUIRE(bwd.find("bf16_mma_vec") != std::string::npos);
	REQUIRE(bwd.find("bf16_mma_mat") != std::string::npos);
	REQUIRE(bwd.find("outer_product") != std::string::npos);
	REQUIRE(bwd.find("float acc") == std::string::npos);
#else
	SUCCEED("bf16 RTC MMA codegen requires CUDA bf16 support and TCNN_MIN_GPU_ARCH >= 80.");
#endif
}

TEST_CASE("FullyFusedMLP supports bf16 fused forward/backward paths", "[network][bf16]") {
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
		SUCCEED("No CUDA device matching TCNN_MIN_GPU_ARCH is available for bf16 tensor cores.");
		return;
	}

	CUDA_CHECK_THROW(cudaSetDevice(bf16_device));
	tcnn_test_setup();

	FullyFusedMLP<T, 16> mlp{16, 16, 2, Activation::ReLU, Activation::None};

	GPUMemory<T> params{mlp.n_params()};
	GPUMemory<T> gradients{mlp.n_params()};
	fill_bf16<<<n_blocks_linear(mlp.n_params()), N_THREADS_LINEAR>>>((uint32_t)mlp.n_params(), params.data(), 0.01f);
	CUDA_CHECK_THROW(cudaMemset(gradients.data(), 0, gradients.get_bytes()));
	CUDA_CHECK_THROW(cudaGetLastError());
	mlp.set_params(params.data(), params.data(), gradients.data());

	GPUMatrix<T> input{16, BATCH_SIZE_GRANULARITY};
	GPUMatrix<T> output{16, BATCH_SIZE_GRANULARITY};
	GPUMatrix<T> output_gradient{16, BATCH_SIZE_GRANULARITY};
	GPUMatrix<T> input_gradient{16, BATCH_SIZE_GRANULARITY};
	fill_bf16<<<n_blocks_linear(input.n_elements()), N_THREADS_LINEAR>>>(input.n_elements(), input.data(), 0.25f);
	fill_bf16<<<n_blocks_linear(output_gradient.n_elements()), N_THREADS_LINEAR>>>(output_gradient.n_elements(), output_gradient.data(), 1.0f);
	CUDA_CHECK_THROW(cudaMemset(output.data(), 0, output.n_bytes()));
	CUDA_CHECK_THROW(cudaMemset(input_gradient.data(), 0, input_gradient.n_bytes()));
	CUDA_CHECK_THROW(cudaGetLastError());

	REQUIRE_NOTHROW(mlp.inference_mixed_precision(input, output));
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
	REQUIRE(has_only_finite_values_and_any_nonzero(output.to_cpu_vector()));

	std::unique_ptr<Context> ctx;
	REQUIRE_NOTHROW(ctx = mlp.forward(input, &output, false, true));
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
	REQUIRE(has_only_finite_values_and_any_nonzero(output.to_cpu_vector()));

	REQUIRE_NOTHROW(mlp.backward(*ctx, input, output, output_gradient, &input_gradient, false, GradientMode::Ignore));
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
	REQUIRE(has_only_finite_values_and_any_nonzero(input_gradient.to_cpu_vector()));

	CUDA_CHECK_THROW(cudaMemset(gradients.data(), 0, gradients.get_bytes()));
	CUDA_CHECK_THROW(cudaMemset(input_gradient.data(), 0, input_gradient.n_bytes()));
#if defined(TCNN_RTC)
	REQUIRE_NOTHROW(mlp.backward(*ctx, input, output, output_gradient, &input_gradient));
#else
	REQUIRE_NOTHROW(mlp.backward_impl(nullptr, *ctx, input, output, output_gradient, &input_gradient, false, GradientMode::Overwrite));
#endif
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
	REQUIRE(has_only_finite_values_and_any_nonzero(input_gradient.to_cpu_vector()));
	std::vector<T> gradient_values(mlp.n_params());
	gradients.copy_to_host(gradient_values);
	REQUIRE(has_only_finite_values_and_any_nonzero(gradient_values));
}
#else
TEST_CASE("FullyFusedMLP bf16 support requires Ampere or newer", "[network][bf16]") {
	SUCCEED("bf16 tensor cores require TCNN_MIN_GPU_ARCH >= 80.");
}
#endif
