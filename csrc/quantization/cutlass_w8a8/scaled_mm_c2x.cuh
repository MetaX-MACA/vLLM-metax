// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved. 
#pragma once
#include <stddef.h>
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>

// clang-format will break include orders
// clang-format off
#include "mctlass/mctlass_ex.h"
#include "mctlass/half.h"
#include "mctlass/layout/matrix.h"
#include "mctlass/epilogue/thread/scale_type.h"
#include "mctlass/util/command_line.h"

#include "core/math.hpp"
#include "cutlass_extensions/common.hpp"
// clang-format on