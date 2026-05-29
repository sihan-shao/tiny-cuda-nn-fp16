#!/usr/bin/env python3

# Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import tinycudann as tcnn

net = tcnn.Network(
	n_input_dims=3,
	n_output_dims=3,
	network_config={
		"otype": "FullyFusedMLP",
		"activation": "ReLU",
		"output_activation": "None",
		"n_neurons": 16,
		"n_hidden_layers": 2,
	},
).cuda()

x = torch.rand(256, 3, device='cuda')
y = net(x)
y.sum().backward() # OK


x2 = torch.rand(256, 3, device='cuda')
y = net(x)
y2 = net(x2)
(y + y2).sum().backward() # RuntimeError: Must call forward() before calling backward()



if torch.cuda.get_device_capability()[0] >= 8:
	bf16_model = tcnn.NetworkWithInputEncoding(
		n_input_dims=3,
		n_output_dims=4,
		encoding_config={
			"otype": "HashGrid",
			"base_resolution": 4,
			"log2_hashmap_size": 4,
			"n_features_per_level": 2,
			"n_levels": 2,
			"per_level_scale": 2.0,
			"interpolation": "Linear",
		},
		network_config={
			"otype": "FullyFusedMLP",
			"activation": "ReLU",
			"output_activation": "None",
			"n_neurons": 16,
			"n_hidden_layers": 2,
		},
		dtype=torch.bfloat16,
	).cuda()
	bf16_model.jit_fusion = False
	assert bf16_model.dtype == torch.bfloat16

	bf16_x = torch.rand(256, 3, device='cuda', requires_grad=True)
	bf16_y = bf16_model(bf16_x)
	assert bf16_y.dtype == torch.bfloat16
	bf16_y.float().square().mean().backward()
	assert bf16_x.grad is not None
	assert bf16_model.params.grad is not None

	bf16_jit_model = tcnn.NetworkWithInputEncoding(
		n_input_dims=3,
		n_output_dims=4,
		encoding_config={
			"otype": "HashGrid",
			"base_resolution": 4,
			"log2_hashmap_size": 4,
			"n_features_per_level": 2,
			"n_levels": 2,
			"per_level_scale": 2.0,
			"interpolation": "Linear",
		},
		network_config={
			"otype": "FullyFusedMLP",
			"activation": "ReLU",
			"output_activation": "None",
			"n_neurons": 16,
			"n_hidden_layers": 2,
		},
		dtype=torch.bfloat16,
	).cuda()
	bf16_jit_model.jit_fusion = True
	with torch.no_grad():
		bf16_jit_model.params.copy_(bf16_model.params.detach())
	bf16_model.zero_grad(set_to_none=True)
	bf16_jit_model.zero_grad(set_to_none=True)

	bf16_compare_x = torch.rand(256, 3, device='cuda')
	bf16_ref_x = bf16_compare_x.detach().clone().requires_grad_(True)
	bf16_jit_x = bf16_compare_x.detach().clone().requires_grad_(True)
	bf16_ref_y = bf16_model(bf16_ref_x)
	bf16_jit_y = bf16_jit_model(bf16_jit_x)
	assert bf16_jit_y.dtype == torch.bfloat16
	torch.testing.assert_close(bf16_jit_y.float(), bf16_ref_y.float(), rtol=1e-2, atol=1e-2)
	bf16_ref_y.float().square().mean().backward()
	bf16_jit_y.float().square().mean().backward()
	assert bf16_jit_x.grad is not None
	assert bf16_jit_model.params.grad is not None
	torch.testing.assert_close(bf16_jit_x.grad, bf16_ref_x.grad, rtol=1e-2, atol=1e-5)
	torch.testing.assert_close(bf16_jit_model.params.grad.float(), bf16_model.params.grad.float(), rtol=1e-2, atol=1e-5)
	assert bf16_jit_model.jit_fusion

	bf16_zero_hidden_config = {
		"otype": "CutlassMLP",
		"activation": "ReLU",
		"output_activation": "None",
		"n_neurons": 16,
		"n_hidden_layers": 0,
	}
	bf16_zero_ref = tcnn.NetworkWithInputEncoding(
		n_input_dims=3,
		n_output_dims=4,
		encoding_config={
			"otype": "HashGrid",
			"base_resolution": 4,
			"log2_hashmap_size": 4,
			"n_features_per_level": 2,
			"n_levels": 2,
			"per_level_scale": 2.0,
			"interpolation": "Linear",
		},
		network_config=bf16_zero_hidden_config,
		dtype=torch.bfloat16,
	).cuda()
	bf16_zero_ref.jit_fusion = False
	bf16_zero_jit = tcnn.NetworkWithInputEncoding(
		n_input_dims=3,
		n_output_dims=4,
		encoding_config={
			"otype": "HashGrid",
			"base_resolution": 4,
			"log2_hashmap_size": 4,
			"n_features_per_level": 2,
			"n_levels": 2,
			"per_level_scale": 2.0,
			"interpolation": "Linear",
		},
		network_config=bf16_zero_hidden_config,
		dtype=torch.bfloat16,
	).cuda()
	bf16_zero_jit.jit_fusion = True
	with torch.no_grad():
		bf16_zero_jit.params.copy_(bf16_zero_ref.params.detach())

	bf16_zero_x = torch.rand(128, 3, device='cuda')
	bf16_zero_ref_x = bf16_zero_x.detach().clone().requires_grad_(True)
	bf16_zero_jit_x = bf16_zero_x.detach().clone().requires_grad_(True)
	bf16_zero_ref_y = bf16_zero_ref(bf16_zero_ref_x)
	bf16_zero_jit_y = bf16_zero_jit(bf16_zero_jit_x)
	torch.testing.assert_close(bf16_zero_jit_y.float(), bf16_zero_ref_y.float(), rtol=1e-2, atol=1e-2)
	bf16_zero_ref_y.float().square().mean().backward()
	bf16_zero_jit_y.float().square().mean().backward()
	torch.testing.assert_close(bf16_zero_jit_x.grad, bf16_zero_ref_x.grad, rtol=1e-2, atol=1e-5)
	torch.testing.assert_close(bf16_zero_jit.params.grad.float(), bf16_zero_ref.params.grad.float(), rtol=1e-2, atol=1e-5)
	assert bf16_zero_jit.jit_fusion

print("success!")
