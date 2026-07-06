/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/csrc/stable/tensor.h>

#include "libtorch_stable/torch_utils.h"
void cutlass_scaled_fp4_mm(torch::stable::Tensor& D,
                           const torch::stable::Tensor& A,
                           const torch::stable::Tensor& B,
                           const torch::stable::Tensor& A_sf,
                           const torch::stable::Tensor& B_sf,
                           const torch::stable::Tensor& alpha) {
  STD_TORCH_CHECK_NOT_IMPLEMENTED(false,
                                  "No compiled nvfp4 mm kernel, vLLM should "
                                  "be compiled using CUDA 12.8 and target "
                                  "compute capability 100 or above.");
}

bool cutlass_scaled_mm_supports_fp4(int64_t cuda_device_capability) {
  return false;
}