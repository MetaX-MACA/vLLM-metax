cmake_minimum_required(VERSION 3.26)

# When building directly using CMake, make sure you run the install step
# (it places the .so files in the correct location).
#
# Example:
# mkdir build && cd build
# cmake -G Ninja -DVLLM_PYTHON_EXECUTABLE=`which python3` -DCMAKE_INSTALL_PREFIX=.. ..
# cmake --build . --target install
#
# If you want to only build one target, make sure to install it manually:
# cmake --build . --target _C
# cmake --install . --component _C
project(vllm_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA by default, can be overridden by using -DVLLM_TARGET_DEVICE=... (used by setup.py)
set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")

# /-------------------- MACA Support --------------------/
option(USE_MACA "Enable MACA Support" ON)
option(USE_PRECOMPILED_KERNEL "Build custom op" OFF)

set(MACA_PATH "$ENV{MACA_PATH}")

if (NOT EXISTS ${MACA_PATH})
  message(FATAL_ERROR "MACA not found or invalid path, please check your MACA_PATH")
endif()

message(STATUS "MACA found at ${MACA_PATH}")
add_compile_definitions(USE_MACA)
# /-----------------------------------------------------/

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

# Prevent installation of dependencies (cutlass) by default.
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.10" "3.11" "3.12")


#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: the CUDA torch version is derived from pyproject.toml and various
# requirements.txt files and should be kept consistent.  The ROCm torch
# versions are derived from docker/Dockerfile.rocm
#

set(TORCH_SUPPORTED_VERSION_CUDA "2.10.0")

#
# Try to find python package with an executable that exactly matches
# `VLLM_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

# Ensure the 'nvcc' command is in the PATH
find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()
#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)

# Supported NVIDIA architectures.
# This check must happen after find_package(Torch) because that's when CMAKE_CUDA_COMPILER_VERSION gets defined
if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0")
else()
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0")
endif()

#
# Set up GPU language and check the torch version and warn if it isn't
# what is expected.
#
if (CUDA_FOUND)
  set(VLLM_GPU_LANG "CUDA")

  if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
      "expected for CUDA build, saw ${Torch_VERSION} instead.")
  endif()

else()
  message(FATAL_ERROR "Can't find CUDA installation")
endif()


if(VLLM_GPU_LANG STREQUAL "CUDA")
  #
  # For cuda we want to be able to control which architectures we compile for on
  # a per-file basis in order to cut down on compile time. So here we extract
  # the set of architectures we want to compile for and remove the from the
  # CMAKE_CUDA_FLAGS so that they are not applied globally.
  #
  clear_cuda_arches(CUDA_ARCH_FLAGS)
  extract_unique_cuda_archs_ascending(CUDA_ARCHS "${CUDA_ARCH_FLAGS}")
  message(STATUS "CUDA target architectures: ${CUDA_ARCHS}")
  # Filter the target architectures by the supported supported archs
  # since for some files we will build for all CUDA_ARCHS.
  cuda_archs_loose_intersection(CUDA_ARCHS
    "${CUDA_SUPPORTED_ARCHS}" "${CUDA_ARCHS}")
  message(STATUS "CUDA supported target architectures: ${CUDA_ARCHS}")
else()
  #
  # For other GPU targets override the GPU architectures detected by cmake/torch
  # and filter them by the supported versions for the current language.
  # The final set of arches is stored in `VLLM_GPU_ARCHES`.
  #
  override_gpu_arches(VLLM_GPU_ARCHES
    ${VLLM_GPU_LANG}
    "${${VLLM_GPU_LANG}_SUPPORTED_ARCHS}")
endif()

#
# Query torch for additional GPU compilation flags for the given
# `VLLM_GPU_LANG`.
# The final set of arches is stored in `VLLM_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG} USE_MACA)

#
# Set nvcc parallelism.
#
if(NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()


#
# Use FetchContent for C++ dependencies that are compiled as part of vLLM's build process.
# setup.py will override FETCHCONTENT_BASE_DIR to play nicely with sccache.
# Each dependency that produces build artifacts should override its BINARY_DIR to avoid
# conflicts between build types. It should instead be set to ${CMAKE_BINARY_DIR}/<dependency>.
#
include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

#
# Define other extension targets
#

#
# cumem_allocator extension
#

set(VLLM_CUMEM_EXT_SRC
  "csrc/cumem_allocator.cpp")

set_gencode_flags_for_srcs(
  SRCS "${VLLM_CUMEM_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  message(STATUS "Enabling cumem allocator extension.")
  # link against cuda driver library
  list(APPEND CUMEM_LIBS CUDA::cuda_driver)
  define_gpu_extension_target(
    cumem_allocator
    DESTINATION vllm_metax
    LANGUAGE CXX
    SOURCES ${VLLM_CUMEM_EXT_SRC}
    LIBRARIES ${CUMEM_LIBS}
    USE_SABI 3.8
    WITH_SOABI)
endif()

#
# _C_stable_libtorch extension (ops registered via STABLE_TORCH_LIBRARY)
#
set(VLLM_STABLE_EXT_SRC
  "csrc/libtorch_stable/torch_bindings.cpp"
  "csrc/libtorch_stable/activation_kernels.cu"
  "csrc/libtorch_stable/quantization/activation_kernels.cu"
  "csrc/libtorch_stable/quantization/w8a8/int8/scaled_quant.cu"
  "csrc/libtorch_stable/quantization/w8a8/fp8/common.cu"
  "csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu"
  "csrc/libtorch_stable/quantization/w8a8/int8/per_token_group_quant.cu"
  "csrc/libtorch_stable/permute_cols.cu"
  "csrc/libtorch_stable/quantization/gptq/q_gemm.cu"
  "csrc/libtorch_stable/pos_encoding_kernels.cu"
  "csrc/libtorch_stable/fused_qknorm_rope_kernel.cu"
  "csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu"
  "csrc/libtorch_stable/layernorm_kernels.cu"
  "csrc/libtorch_stable/layernorm_quant_kernels.cu"
  "csrc/libtorch_stable/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu"
  "csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu"
  "csrc/libtorch_stable/attention/merge_attn_states.cu"
  "csrc/libtorch_stable/sampler.cu"
  "csrc/libtorch_stable/topk.cu"
  "csrc/libtorch_stable/mamba/selective_scan_fwd.cu"
  "csrc/libtorch_stable/attention/paged_attention_v1.cu"
  "csrc/libtorch_stable/attention/paged_attention_v2.cu"
  "csrc/libtorch_stable/cache_kernels.cu"
  "csrc/libtorch_stable/cache_kernels_fused.cu"
  "csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu"
  "csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel_bf16.cu")

# support opt of gptq-marlin
set_source_files_properties(
  "csrc/libtorch_stable/quantization/gptq/q_gemm.cu"
  PROPERTIES
  COMPILE_FLAGS "-mllvm -metaxgpu-igroup=true -mllvm -metaxgpu-igroup-config=B8 -maxmregcount=128"
)

if(VLLM_GPU_LANG STREQUAL "CUDA")
  SET(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

  # Set CUTLASS_REVISION. Used for FetchContent. Also fixes some bogus messages when building.
  set(CUTLASS_REVISION "v3.9.2" CACHE STRING "CUTLASS revision to use")

  # Use the specified CUTLASS source directory for compilation if VLLM_CUTLASS_SRC_DIR is provided
  if (DEFINED ENV{VLLM_CUTLASS_SRC_DIR})
    set(VLLM_CUTLASS_SRC_DIR $ENV{VLLM_CUTLASS_SRC_DIR})
  endif()

  # Substitute CUTLASS with MATLASS 
  message(WARNING "Use MACA, Overwrite VLLM_CUTLASS_SRC_DIR.")
  set(VLLM_CUTLASS_SRC_DIR $ENV{MACA_PATH}/include)

  if(NOT IS_ABSOLUTE VLLM_CUTLASS_SRC_DIR)
    get_filename_component(VLLM_CUTLASS_SRC_DIR "${VLLM_CUTLASS_SRC_DIR}" ABSOLUTE)
  endif()
  message(STATUS "The VLLM_CUTLASS_SRC_DIR is set, using ${VLLM_CUTLASS_SRC_DIR} for compilation")
  FetchContent_Declare(cutlass SOURCE_DIR ${VLLM_CUTLASS_SRC_DIR})

  list(APPEND VLLM_STABLE_EXT_SRC
    # "csrc/libtorch_stable/cuda_view.cu"
    "csrc/libtorch_stable/cuda_utils_kernels.cu"
    "csrc/libtorch_stable/cutlass_extensions/common.cpp"
    "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu"
    # "csrc/libtorch_stable/quantization/fp4/nvfp4_quant_entry.cu"
    "csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_entry.cu"
    "csrc/libtorch_stable/quantization/awq/gemm_kernels.cu"
    "csrc/libtorch_stable/minimax_reduce_rms_kernel.cu")

  set_gencode_flags_for_srcs(
    SRCS "${VLLM_STABLE_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")

  # support opt of gptq-marlin
  set_source_files_properties(
    "csrc/libtorch_stable/quantization/awq/gemm_kernels.cu"
    PROPERTIES
    COMPILE_FLAGS " -mllvm -metaxgpu-igroup=true -mllvm -metaxgpu-igroup-config=B8 -maxmregcount=128"
  )

  # support opt of cutlass w8a8 scale mm
  set_source_files_properties(
    "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu"
    PROPERTIES
    COMPILE_FLAGS " -mllvm -metaxgpu-igroup=true -mllvm -misched-postra=true"
  )  

  # DeepSeek V3 fused A GEMM kernel (requires SM 9.0+, Hopper and later)
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(DSV3_FUSED_A_GEMM_ARCHS "9.0a;10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(DSV3_FUSED_A_GEMM_ARCHS "9.0a;10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND DSV3_FUSED_A_GEMM_ARCHS)
    set(DSV3_FUSED_A_GEMM_SRCS "csrc/libtorch_stable/dsv3_fused_a_gemm.cu")
    set_gencode_flags_for_srcs(
      SRCS "${DSV3_FUSED_A_GEMM_SRCS}"
      CUDA_ARCHS "${DSV3_FUSED_A_GEMM_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${DSV3_FUSED_A_GEMM_SRCS}")
    message(STATUS "Building dsv3_fused_a_gemm for archs: ${DSV3_FUSED_A_GEMM_ARCHS}")
  else()
    message(STATUS "Not building dsv3_fused_a_gemm as no compatible archs found "
                   "in CUDA target architectures.")
  endif()

  # FP32 router GEMM (H=3072, E=256, M<=32). Requires SM90+ and CUDA >= 12.0.
  cuda_archs_sm90plus(FP32_ROUTER_GEMM_ARCHS "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND FP32_ROUTER_GEMM_ARCHS)
    set(FP32_ROUTER_GEMM_SRCS
      "csrc/libtorch_stable/fp32_router_gemm_entry.cu"
      "csrc/libtorch_stable/fp32_router_gemm.cu")
    set_gencode_flags_for_srcs(
      SRCS "${FP32_ROUTER_GEMM_SRCS}"
      CUDA_ARCHS "${FP32_ROUTER_GEMM_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${FP32_ROUTER_GEMM_SRCS}")
    message(STATUS "Building fp32_router_gemm for archs: ${FP32_ROUTER_GEMM_ARCHS}")
  else()
    message(STATUS "Not building fp32_router_gemm as no compatible archs found "
                   "(requires SM90+ and CUDA >= 12.0).")
  endif()
  
  #
  # For the cutlass_scaled_mm kernels we want to build the c2x (CUTLASS 2.x)
  # kernels for the remaining archs that are not already built for 3x.
  # (Build 8.9 for FP8)
  cuda_archs_loose_intersection(SCALED_MM_2X_ARCHS
    "7.5;8.0;8.7;8.9+PTX" "${CUDA_ARCHS}")
  # subtract out the archs that are already built for 3x
  list(REMOVE_ITEM SCALED_MM_2X_ARCHS ${SCALED_MM_3X_ARCHS})
  if (SCALED_MM_2X_ARCHS)
    set(SCALED_MM_C2X_SRCS "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SCALED_MM_C2X_SRCS}"
      CUDA_ARCHS "${SCALED_MM_2X_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${SCALED_MM_C2X_SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_C2X=1")
    
    # support opt of cutlass w8a8 scale mm
    set_source_files_properties(
      "csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x.cu"
      PROPERTIES
      COMPILE_FLAGS " -mllvm -metaxgpu-igroup=true -mllvm -misched-postra=true"
    )
    message(STATUS "Building scaled_mm_c2x for archs: ${SCALED_MM_2X_ARCHS}")
  else()
    if (SCALED_MM_3X_ARCHS)
      message(STATUS "Not building scaled_mm_c2x as all archs are already built"
                     " for and covered by scaled_mm_c3x")
    else()
      message(STATUS "Not building scaled_mm_c2x as no compatible archs found "
                    "in CUDA target architectures")
    endif()
  endif()


  #
  # CUTLASS MoE kernels (moved from _C to _C_stable_libtorch)
  #

  # The MoE kernel cutlass_moe_mm requires CUDA 12.3 or later (and ONLY works
  # on Hopper). get_cutlass_(batched_)moe_mm_data should only be compiled
  # if it's possible to compile MoE kernels that use its output.
  # cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a" "${CUDA_ARCHS}")
  # if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND SCALED_MM_ARCHS)
  #   set(CUTLASS_MOE_SM90_SRCS "csrc/libtorch_stable/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm90.cu")
  #   set_gencode_flags_for_srcs(
  #     SRCS "${CUTLASS_MOE_SM90_SRCS}"
  #     CUDA_ARCHS "${SCALED_MM_ARCHS}")
  #   list(APPEND VLLM_STABLE_EXT_SRC "${CUTLASS_MOE_SM90_SRCS}")
  #   list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM90=1")
  #   message(STATUS "Building grouped_mm_c3x for archs: ${SCALED_MM_ARCHS}")
  # else()
  #   if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND SCALED_MM_ARCHS)
  #     message(STATUS "Not building grouped_mm_c3x kernels as CUDA Compiler version is "
  #                    "not >= 12.3, we recommend upgrading to CUDA 12.3 or later "
  #                    "if you intend on running FP8 quantized MoE models on Hopper.")
  #   else()
  #     message(STATUS "Not building grouped_mm_c3x as no compatible archs found "
  #                    "in CUDA target architectures.")
  #   endif()
  # endif()
  # if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  #   cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f" "${CUDA_ARCHS}")
  # else()
  #   cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.3a" "${CUDA_ARCHS}")
  # endif()
  # if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
  #   set(CUTLASS_MOE_SM100_SRCS "csrc/libtorch_stable/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu")
  #   set_gencode_flags_for_srcs(
  #     SRCS "${CUTLASS_MOE_SM100_SRCS}"
  #     CUDA_ARCHS "${SCALED_MM_ARCHS}")
  #   list(APPEND VLLM_STABLE_EXT_SRC "${CUTLASS_MOE_SM100_SRCS}")
  #   list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
  #   message(STATUS "Building grouped_mm_c3x for archs: ${SCALED_MM_ARCHS}")
  # else()
  #   if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
  #     message(STATUS "Not building grouped_mm_c3x kernels as CUDA Compiler version is "
  #                    "not >= 12.8, we recommend upgrading to CUDA 12.8 or later "
  #                    "if you intend on running FP8 quantized MoE models on Blackwell.")
  #   else()
  #     message(STATUS "Not building grouped_mm_c3x as no compatible archs found "
  #                    "in CUDA target architectures.")
  #   endif()
  # endif()

  # # moe_data.cu is used by all CUTLASS MoE kernels.
  # if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  #   cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
  # else()
  #   cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
  # endif()
  # if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND CUTLASS_MOE_DATA_ARCHS)
  #   set(CUTLASS_MOE_DATA_SRCS "csrc/libtorch_stable/quantization/w8a8/cutlass/moe/moe_data.cu")
  #   set_gencode_flags_for_srcs(
  #     SRCS "${CUTLASS_MOE_DATA_SRCS}"
  #     CUDA_ARCHS "${CUTLASS_MOE_DATA_ARCHS}")
  #   list(APPEND VLLM_STABLE_EXT_SRC "${CUTLASS_MOE_DATA_SRCS}")
  #   message(STATUS "Building moe_data for archs: ${CUTLASS_MOE_DATA_ARCHS}")
  # else()
  #   if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND CUTLASS_MOE_DATA_ARCHS)
  #     message(STATUS "Not building moe_data as CUDA Compiler version is "
  #                    "not >= 12.3, we recommend upgrading to CUDA 12.3 or later "
  #                    "if you intend on running FP8 quantized MoE models on Hopper or Blackwell.")
  #   else()
  #     message(STATUS "Not building moe_data as no compatible archs found "
  #                    "in CUDA target architectures.")
  #   endif()
  # endif()


  # # CUTLASS MLA Archs and flags
  # cuda_archs_loose_intersection(MLA_ARCHS "10.0a" "${CUDA_ARCHS}")
  # if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.8 AND MLA_ARCHS AND NOT USE_MACA)
  #   set(SRCS
  #     "csrc/attention/mla/cutlass_mla_kernels.cu")
  #   set_gencode_flags_for_srcs(
  #     SRCS "${SRCS}"
  #     CUDA_ARCHS "${MLA_ARCHS}")
  #   list(APPEND VLLM_STABLE_EXT_SRC "${SRCS}")
  #   list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MLA=1")
  #   # Add MLA-specific include directories only to MLA source files
  #   set_source_files_properties(${SRCS}
  #     PROPERTIES INCLUDE_DIRECTORIES "${CUTLASS_DIR}/examples/77_blackwell_fmha;${CUTLASS_DIR}/examples/common")
  #   message(STATUS "Building CUTLASS MLA for archs: ${MLA_ARCHS}")
  # else()
  #   message(WARNING "Use MACA, Skip CUTLASS MLA.")
  #   message(STATUS "Not building CUTLASS MLA as no compatible archs were found.")
  #   # clear MLA_ARCHS
  #   set(MLA_ARCHS)
  # endif()


  #
  # Machete kernels

  # The machete kernels only work on hopper and require CUDA 12.0 or later.
  # Only build Machete kernels if we are building for something compatible with sm90a
  cuda_archs_loose_intersection(MACHETE_ARCHS "9.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0 AND MACHETE_ARCHS)
    #
    # For the Machete kernels we automatically generate sources for various
    # preselected input type pairs and schedules.
    # Generate sources:
    set(MACHETE_GEN_SCRIPT
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantization/machete/generate.py)
    file(MD5 ${MACHETE_GEN_SCRIPT} MACHETE_GEN_SCRIPT_HASH)

    message(STATUS "Machete generation script hash: ${MACHETE_GEN_SCRIPT_HASH}")
    message(STATUS "Last run machete generate script hash: $CACHE{MACHETE_GEN_SCRIPT_HASH}")

    if (NOT DEFINED CACHE{MACHETE_GEN_SCRIPT_HASH}
        OR NOT $CACHE{MACHETE_GEN_SCRIPT_HASH} STREQUAL ${MACHETE_GEN_SCRIPT_HASH})
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass_extensions/:${CUTLASS_DIR}/python/:${VLLM_PYTHON_PATH}:$PYTHONPATH
          ${Python_EXECUTABLE} ${MACHETE_GEN_SCRIPT}
        RESULT_VARIABLE machete_generation_result
        OUTPUT_VARIABLE machete_generation_output
        OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
        ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
      )

      if (NOT machete_generation_result EQUAL 0)
        message(FATAL_ERROR "Machete generation failed."
                            " Result: \"${machete_generation_result}\""
                            "\nCheck the log for details: "
                            "${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log")
      else()
        set(MACHETE_GEN_SCRIPT_HASH ${MACHETE_GEN_SCRIPT_HASH}
            CACHE STRING "Last run machete generate script hash" FORCE)
        message(STATUS "Machete generation completed successfully.")
      endif()
    else()
      message(STATUS "Machete generation script has not changed, skipping generation.")
    endif()

    # Add machete generated sources
    file(GLOB MACHETE_GEN_SOURCES "csrc/quantization/machete/generated/*.cu")
    list(APPEND VLLM_STABLE_EXT_SRC ${MACHETE_GEN_SOURCES})

    # forward compatible
    set_gencode_flags_for_srcs(
      SRCS "${MACHETE_GEN_SOURCES}"
      CUDA_ARCHS "${MACHETE_ARCHS}")

    list(APPEND VLLM_STABLE_EXT_SRC
      csrc/quantization/machete/machete_pytorch.cu)

    message(STATUS "Building Machete kernels for archs: ${MACHETE_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0
        AND MACHETE_ARCHS)
      message(STATUS "Not building Machete kernels as CUDA Compiler version is "
                     "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                     "later if you intend on running w4a16 quantized models on "
                     "Hopper.")
    else()
      message(STATUS "Not building Machete kernels as no compatible archs "
                     "found in CUDA target architectures")
    endif()
  endif()
# if CUDA endif
endif()

if(USE_PRECOMPILED_KERNEL)
  message(STATUS "Skipping _C extension as USE_PRECOMPILED_KERNEL is ON.")
else()
message(STATUS "Enabling C extension.")
list(REMOVE_DUPLICATES VLLM_STABLE_EXT_SRC)
define_gpu_extension_target(
  _C_stable_libtorch
  DESTINATION vllm_metax
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_STABLE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR} ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

target_compile_options(_C_stable_libtorch PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:
  --Wno-maca-min-blocks-per-multiprocessor
  --Wno-ignored-attributes
  --Wno-macro-redefined
  --Wno-pass-failed
  --Wno-maca-compat
  --Wno-sometimes-uninitialized
  >)

# If CUTLASS is compiled on NVCC >= 12.5, it by default uses
# cudaGetDriverEntryPointByVersion as a wrapper to avoid directly calling the
# driver API. This causes problems when linking with earlier versions of CUDA.
# Setting this variable sidesteps the issue by calling the driver directly.
target_compile_definitions(_C_stable_libtorch PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
endif()

# Needed to use cuda/hip APIs from C-shim
if(VLLM_GPU_LANG STREQUAL "CUDA")
  # Set TORCH_TARGET_VERSION for stable ABI compatibility.
  # This ensures we only use C-shim APIs available in PyTorch 2.11.
  # _C_stable_libtorch is abi compatible with PyTorch >= TORCH_TARGET_VERSION
  # which is currently set to 2.11.
  target_compile_definitions(_C_stable_libtorch PRIVATE
    TORCH_TARGET_VERSION=0x020B000000000000ULL)
  target_compile_definitions(_C_stable_libtorch PRIVATE USE_CUDA)
  if(COOPERATIVE_TOPK_ARCHS)
    target_compile_definitions(_C_stable_libtorch PRIVATE
      VLLM_ENABLE_COOPERATIVE_TOPK=1)
  endif()
  # Needed by CUTLASS kernels
  target_compile_definitions(_C_stable_libtorch PRIVATE
    CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
elseif(VLLM_GPU_LANG STREQUAL "HIP")
  # Set TORCH_TARGET_VERSION for stable ABI compatibility.
  # This ensures we only use C-shim APIs available in PyTorch 2.10.
  # _C_stable_libtorch is abi compatible with PyTorch >= TORCH_TARGET_VERSION
  # which is currently set to 2.10.
  target_compile_definitions(_C_stable_libtorch PRIVATE
    TORCH_TARGET_VERSION=0x020A000000000000ULL)
  target_compile_definitions(_C_stable_libtorch PRIVATE USE_ROCM)
endif()

#
# _moe_C_stable_libtorch extension
#

set(VLLM_MOE_EXT_SRC
  "csrc/libtorch_stable/moe/torch_bindings.cpp"
  "csrc/libtorch_stable/moe/moe_align_sum_kernels.cu"
  "csrc/libtorch_stable/moe/topk_softmax_kernels.cu"
  "csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_MOE_EXT_SRC
    "csrc/libtorch_stable/moe/moe_wna16.cu"
    "csrc/libtorch_stable/moe/grouped_topk_kernels.cu")
endif()

if(VLLM_GPU_LANG STREQUAL "CUDA")
  set(MOE_PERMUTE_SRC
      "csrc/libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu"
      "csrc/libtorch_stable/moe/moe_permute_unpermute_op.cu"
    )

  list(APPEND VLLM_MOE_EXT_SRC "${MOE_PERMUTE_SRC}")
endif()

set_gencode_flags_for_srcs(
  SRCS "${VLLM_MOE_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  set(VLLM_MOE_WNA16_SRC
    "csrc/libtorch_stable/moe/moe_wna16.cu")

  set_gencode_flags_for_srcs(
    SRCS "${VLLM_MOE_WNA16_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")


  # DeepSeek V3 router GEMM kernel requires SM90+ and CUDA >= 12.0.
  # (fp32_router_gemm has been migrated to _C_stable_libtorch above.)
  cuda_archs_sm90plus(SM90PLUS_ROUTER_GEMM_ARCHS "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND SM90PLUS_ROUTER_GEMM_ARCHS)
    set(DSV3_ROUTER_GEMM_SRC
      "csrc/libtorch_stable/moe/dsv3_router_gemm_entry.cu"
      "csrc/libtorch_stable/moe/dsv3_router_gemm_float_out.cu"
      "csrc/libtorch_stable/moe/dsv3_router_gemm_bf16_out.cu")
    set_gencode_flags_for_srcs(
      SRCS "${DSV3_ROUTER_GEMM_SRC}"
      CUDA_ARCHS "${SM90PLUS_ROUTER_GEMM_ARCHS}")
    list(APPEND VLLM_MOE_EXT_SRC "${DSV3_ROUTER_GEMM_SRC}")

    message(STATUS "Building DSV3 router GEMM kernels for archs: ${SM90PLUS_ROUTER_GEMM_ARCHS}")
  else()
    message(STATUS "Not building DSV3 router GEMM kernels as no compatible archs found"
                   " (requires SM90+ and CUDA >= 12.0)")
  endif()
endif()


if (USE_PRECOMPILED_KERNEL)
  message(STATUS "Skipping _moe_C_stable_libtorch extension as USE_PRECOMPILED_KERNEL is ON.")
else()
define_gpu_extension_target(
  _moe_C_stable_libtorch
  DESTINATION vllm_metax
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_MOE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  ${BLAS_API_ARGS}
  USE_SABI 3
  WITH_SOABI)
endif()

target_compile_options(_moe_C_stable_libtorch PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:
  --Wno-maca-min-blocks-per-multiprocessor
  --Wno-ignored-attributes
  --Wno-macro-redefined
  --Wno-pass-failed
  --Wno-maca-compat
  >)


# Needed to use cuda/hip APIs from C-shim
if(VLLM_GPU_LANG STREQUAL "CUDA")
  # Set TORCH_TARGET_VERSION for stable ABI compatibility.
  # This ensures we only use C-shim APIs available in PyTorch 2.11.
  # _moe_C_stable_libtorch is abi compatible with PyTorch >= TORCH_TARGET_VERSION
  # which is currently set to 2.11.
  target_compile_definitions(_moe_C_stable_libtorch PRIVATE
    TORCH_TARGET_VERSION=0x020B000000000000ULL)
  target_compile_definitions(_moe_C_stable_libtorch PRIVATE USE_CUDA)
  # Needed by CUTLASS kernels
  target_compile_definitions(_moe_C_stable_libtorch PRIVATE
    CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
elseif(VLLM_GPU_LANG STREQUAL "HIP")
  # Set TORCH_TARGET_VERSION for stable ABI compatibility.
  # This ensures we only use C-shim APIs available in PyTorch 2.10.
  # _moe_C_stable_libtorch is abi compatible with PyTorch >= TORCH_TARGET_VERSION
  # which is currently set to 2.10.
  target_compile_definitions(_moe_C_stable_libtorch PRIVATE
    TORCH_TARGET_VERSION=0x020A000000000000ULL)
  target_compile_definitions(_moe_C_stable_libtorch PRIVATE USE_ROCM)
endif()
