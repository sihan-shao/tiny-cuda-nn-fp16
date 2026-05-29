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

/** @file   cpp_api.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  API to be consumed by cpp (non-CUDA) programs.
 */

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/cpp_api.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/rtc_kernel.h>

#if !defined(TCNN_NO_NETWORKS)
#include <tiny-cuda-nn/network_with_input_encoding.h>
#endif

namespace tcnn { namespace cpp {

uint32_t batch_size_granularity() { return tcnn::BATCH_SIZE_GRANULARITY; }

int cuda_device() { return tcnn::cuda_device(); }
void set_cuda_device(int device) { tcnn::set_cuda_device(device); }
void free_temporary_memory() { tcnn::free_all_gpu_memory_arenas(); }

bool has_networks() {
#if defined(TCNN_NO_NETWORKS)
	return false;
#else
	return true;
#endif
}

float default_loss_scale(Precision p) {
	switch (p) {
		case Precision::Fp32: return tcnn::default_loss_scale<float>();
		case Precision::Fp16: return tcnn::default_loss_scale<__half>();
#if TCNN_HAS_CUDA_BF16
		case Precision::Bf16: return tcnn::default_loss_scale<__nv_bfloat16>();
#else
		case Precision::Bf16: throw std::runtime_error{"TCNN was not compiled with bf16 support."};
#endif
	}

	throw std::runtime_error{"Unknown precision."};
}

template <typename T> constexpr Precision precision();
template <> constexpr Precision precision<float>() { return Precision::Fp32; }
template <> constexpr Precision precision<__half>() { return Precision::Fp16; }
#if TCNN_HAS_CUDA_BF16
template <> constexpr Precision precision<__nv_bfloat16>() { return Precision::Bf16; }
#endif
Precision preferred_precision() { return precision<network_precision_t>(); }

bool supports_jit_fusion(int device) { return device < 0 ? tcnn::supports_jit_fusion() : tcnn::supports_jit_fusion(device); }

void rtc_set_cache_dir(const std::string& dir) { tcnn::rtc_set_cache_dir(dir); }
void rtc_set_include_dir(const std::string& dir) { tcnn::rtc_set_include_dir(dir); }

void set_log_callback(const std::function<void(LogSeverity, const std::string&)>& callback) {
	tcnn::set_log_callback([callback](tcnn::LogSeverity severity, const std::string& msg) { callback((LogSeverity)severity, msg); });
}

template <typename T>
class DifferentiableObject : public Module {
public:
	DifferentiableObject(tcnn::DifferentiableObject<float, T, T>* model)
	: Module{precision<T>(), precision<T>()}, m_model{model}
	{}

	void inference(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params) override {
		m_model->set_params((T*)params, (T*)params, nullptr);

		GPUMatrix<float, MatrixLayout::ColumnMajor> input_matrix((float*)input, m_model->input_width(), n_elements);
		GPUMatrix<T, MatrixLayout::ColumnMajor> output_matrix((T*)output, m_model->padded_output_width(), n_elements);

		// Run on our own custom stream to ensure CUDA graph capture is possible.
		// (Significant possible speedup.)
		SyncedMultiStream synced_stream{stream, 2};
		m_model->inference_mixed_precision(synced_stream.get(1), input_matrix, output_matrix);
	}

	Context forward(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params, bool prepare_input_gradients) override {
		m_model->set_params((T*)params, (T*)params, nullptr);

		GPUMatrix<float, MatrixLayout::ColumnMajor> input_matrix((float*)input, m_model->input_width(), n_elements);
		GPUMatrix<T, MatrixLayout::ColumnMajor> output_matrix((T*)output, m_model->padded_output_width(), n_elements);

		// Run on our own custom stream to ensure CUDA graph capture is possible.
		// (Significant possible speedup.)
		SyncedMultiStream synced_stream{stream, 2};
		return { m_model->forward(synced_stream.get(1), input_matrix, &output_matrix, false, prepare_input_gradients) };
	}

	void backward(cudaStream_t stream, const Context& ctx, uint32_t n_elements, float* dL_dinput, const void* dL_doutput, void* dL_dparams, const float* input, const void* output, const void* params) override {
		m_model->set_params((T*)params, (T*)params, (T*)dL_dparams);

		GPUMatrix<float, MatrixLayout::ColumnMajor> input_matrix((float*)input, m_model->input_width(), n_elements);
		GPUMatrix<float, MatrixLayout::ColumnMajor> dL_dinput_matrix(dL_dinput, m_model->input_width(), n_elements);

		GPUMatrix<T, MatrixLayout::ColumnMajor> output_matrix((T*)output, m_model->padded_output_width(), n_elements);
		GPUMatrix<T, MatrixLayout::ColumnMajor> dL_doutput_matrix((T*)dL_doutput, m_model->padded_output_width(), n_elements);

		// Run on our own custom stream to ensure CUDA graph capture is possible.
		// (Significant possible speedup.)
		SyncedMultiStream synced_stream{stream, 2};
		m_model->backward(synced_stream.get(1), *ctx.ctx, input_matrix, output_matrix, dL_doutput_matrix, dL_dinput ? &dL_dinput_matrix : nullptr, false, dL_dparams ? GradientMode::Overwrite : GradientMode::Ignore);
	}

	void backward_backward_input(cudaStream_t stream, const Context& ctx, uint32_t n_elements, const float* dL_ddLdinput, const float* input, const void* dL_doutput, void* dL_dparams, void* dL_ddLdoutput, float* dL_dinput, const void* params) override {
		// from: dL_ddLdinput
		// to:   dL_ddLdoutput, dL_dparams
		m_model->set_params((T*)params, (T*)params, (T*)dL_dparams);

		GPUMatrix<float, MatrixLayout::ColumnMajor> input_matrix((float*)input, m_model->input_width(), n_elements);
		GPUMatrix<float, MatrixLayout::ColumnMajor> dL_ddLdinput_matrix((float*)dL_ddLdinput, m_model->input_width(), n_elements);

		GPUMatrix<T, MatrixLayout::ColumnMajor> dL_doutput_matrix((T*)dL_doutput, m_model->padded_output_width(), n_elements);
		GPUMatrix<T, MatrixLayout::ColumnMajor> dL_ddLdoutput_matrix((T*)dL_ddLdoutput, m_model->padded_output_width(), n_elements);
		GPUMatrix<float, MatrixLayout::ColumnMajor> dL_dinput_matrix((float*)dL_dinput, m_model->input_width(), n_elements);

		// Run on our own custom stream to ensure CUDA graph capture is possible.
		// (Significant possible speedup.)
		SyncedMultiStream synced_stream{stream, 2};
		m_model->backward_backward_input(synced_stream.get(1), *ctx.ctx, input_matrix, dL_ddLdinput_matrix, dL_doutput_matrix, dL_ddLdoutput ? &dL_ddLdoutput_matrix : nullptr, dL_dinput ? &dL_dinput_matrix : nullptr, false, dL_dparams ? GradientMode::Overwrite : GradientMode::Ignore);
	}

	uint32_t n_input_dims() const override { return m_model->input_width(); }
	uint32_t n_output_dims() const override { return m_model->padded_output_width(); }
	size_t n_params() const override { return m_model->n_params(); }

	void initialize_params(size_t seed, float* params_full_precision, float scale) override {
		pcg32 rng{seed};
		m_model->initialize_params(rng, params_full_precision, scale);
	}

	json hyperparams() const override { return m_model->hyperparams(); }
	std::string name() const override { return m_model->name(); }

	bool jit_fusion() const { return m_model->jit_fusion(); }
	void set_jit_fusion(bool val) { m_model->set_jit_fusion(val); }

private:
	std::shared_ptr<tcnn::DifferentiableObject<float, T, T>> m_model;
};

#if !defined(TCNN_NO_NETWORKS)
Module* create_network_with_input_encoding(uint32_t n_input_dims, uint32_t n_output_dims, const json& encoding, const json& network, Precision requested_precision) {
	switch (requested_precision) {
#if TCNN_HALF_PRECISION
		case Precision::Fp32: throw std::runtime_error{"TCNN was compiled with half-precision network support; fp32 networks are not available in this build."};
#else
		case Precision::Fp32: return new DifferentiableObject<float>{new tcnn::NetworkWithInputEncoding<float>{n_input_dims, n_output_dims, encoding, network}};
#endif
#if TCNN_HALF_PRECISION
		case Precision::Fp16: return new DifferentiableObject<__half>{new tcnn::NetworkWithInputEncoding<__half>{n_input_dims, n_output_dims, encoding, network}};
#else
		case Precision::Fp16: throw std::runtime_error{"TCNN was not compiled with half-precision support."};
#endif
#if TCNN_HAS_CUDA_BF16
		case Precision::Bf16: return new DifferentiableObject<__nv_bfloat16>{new tcnn::NetworkWithInputEncoding<__nv_bfloat16>{n_input_dims, n_output_dims, encoding, network}};
#else
		case Precision::Bf16: throw std::runtime_error{"TCNN was not compiled with bf16 support."};
#endif
	}

	throw std::runtime_error{"Unknown precision."};
}

Module* create_network_with_input_encoding(uint32_t n_input_dims, uint32_t n_output_dims, const json& encoding, const json& network) {
	return create_network_with_input_encoding(n_input_dims, n_output_dims, encoding, network, preferred_precision());
}

Module* create_network(uint32_t n_input_dims, uint32_t n_output_dims, const json& network, Precision requested_precision) {
	return create_network_with_input_encoding(n_input_dims, n_output_dims, {{"otype", "Identity"}}, network, requested_precision);
}

Module* create_network(uint32_t n_input_dims, uint32_t n_output_dims, const json& network) {
	return create_network(n_input_dims, n_output_dims, network, preferred_precision());
}
#endif // !defined(TCNN_NO_NETWORKS)

Module* create_encoding(uint32_t n_input_dims, const json& encoding, Precision requested_precision) {
	switch (requested_precision) {
		case Precision::Fp32: return new DifferentiableObject<float>{tcnn::create_encoding<float>(n_input_dims, encoding, 0)};
#if TCNN_HALF_PRECISION
		case Precision::Fp16: return new DifferentiableObject<__half>{tcnn::create_encoding<__half>(n_input_dims, encoding, 0)};
#else
		case Precision::Fp16: throw std::runtime_error{"TCNN was not compiled with half-precision support."};
#endif
#if TCNN_HAS_CUDA_BF16
		case Precision::Bf16: return new DifferentiableObject<__nv_bfloat16>{tcnn::create_encoding<__nv_bfloat16>(n_input_dims, encoding, 0)};
#else
		case Precision::Bf16: throw std::runtime_error{"TCNN was not compiled with bf16 support."};
#endif
	}

	throw std::runtime_error{"Unknown precision."};
}

}}
