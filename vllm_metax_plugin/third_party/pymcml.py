# SPDX-License-Identifier: Apache-2.0

# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# Copyright (c) 2011-2023, NVIDIA Corporation.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the NVIDIA Corporation nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#####

##
# Python bindings for the MXSMLEX library
##
from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string

## C Type mappings ##
## Enums
_nvmlEnableState_t = c_uint
MXSMLEX_FEATURE_DISABLED    = 0
MXSMLEX_FEATURE_ENABLED     = 1

_nvmlBrandType_t = c_uint
MXSMLEX_BRAND_UNKNOWN             = 0
MXSMLEX_BRAND_QUADRO              = 1
MXSMLEX_BRAND_TESLA               = 2
MXSMLEX_BRAND_NVS                 = 3
MXSMLEX_BRAND_GRID                = 4   # Deprecated from API reporting. Keeping definition for backward compatibility.
MXSMLEX_BRAND_GEFORCE             = 5
MXSMLEX_BRAND_TITAN               = 6
MXSMLEX_BRAND_NVIDIA_VAPPS        = 7   # NVIDIA Virtual Applications
MXSMLEX_BRAND_NVIDIA_VPC          = 8   # NVIDIA Virtual PC
MXSMLEX_BRAND_NVIDIA_VCS          = 9   # NVIDIA Virtual Compute Server
MXSMLEX_BRAND_NVIDIA_VWS          = 10  # NVIDIA RTX Virtual Workstation
MXSMLEX_BRAND_NVIDIA_CLOUD_GAMING = 11  # NVIDIA Cloud Gaming
MXSMLEX_BRAND_NVIDIA_VGAMING      = MXSMLEX_BRAND_NVIDIA_CLOUD_GAMING # Deprecated from API reporting. Keeping definition for backward compatibility.
MXSMLEX_BRAND_QUADRO_RTX          = 12
MXSMLEX_BRAND_NVIDIA_RTX          = 13
MXSMLEX_BRAND_NVIDIA              = 14
MXSMLEX_BRAND_GEFORCE_RTX         = 15  # Unused
MXSMLEX_BRAND_TITAN_RTX           = 16  # Unused
MXSMLEX_BRAND_COUNT               = 17

_nvmlTemperatureThresholds_t = c_uint
MXSMLEX_TEMPERATURE_THRESHOLD_SHUTDOWN      = 0
MXSMLEX_TEMPERATURE_THRESHOLD_SLOWDOWN      = 1
MXSMLEX_TEMPERATURE_THRESHOLD_MEM_MAX       = 2
MXSMLEX_TEMPERATURE_THRESHOLD_GPU_MAX       = 3
MXSMLEX_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN  = 4
MXSMLEX_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR = 5
MXSMLEX_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX  = 6
MXSMLEX_TEMPERATURE_THRESHOLD_COUNT         = 7

_nvmlTemperatureSensors_t = c_uint
MXSMLEX_TEMPERATURE_GPU     = 0
MXSMLEX_TEMPERATURE_COUNT   = 1


_nvmlComputeMode_t = c_uint
MXSMLEX_COMPUTEMODE_DEFAULT           = 0
MXSMLEX_COMPUTEMODE_EXCLUSIVE_THREAD  = 1  ## Support Removed
MXSMLEX_COMPUTEMODE_PROHIBITED        = 2
MXSMLEX_COMPUTEMODE_EXCLUSIVE_PROCESS = 3
MXSMLEX_COMPUTEMODE_COUNT             = 4

_nvmlMemoryLocation_t = c_uint
MXSMLEX_MEMORY_LOCATION_L1_CACHE = 0
MXSMLEX_MEMORY_LOCATION_L2_CACHE = 1
MXSMLEX_MEMORY_LOCATION_DEVICE_MEMORY = 2
MXSMLEX_MEMORY_LOCATION_DRAM = 2
MXSMLEX_MEMORY_LOCATION_REGISTER_FILE = 3
MXSMLEX_MEMORY_LOCATION_TEXTURE_MEMORY = 4
MXSMLEX_MEMORY_LOCATION_TEXTURE_SHM = 5
MXSMLEX_MEMORY_LOCATION_CBU = 6
MXSMLEX_MEMORY_LOCATION_SRAM = 7
MXSMLEX_MEMORY_LOCATION_COUNT = 8

MXSMLEX_NVLINK_MAX_LINKS = 18

# For backwards compatibility, maintain the incorrectly-named "LANES" define
MXSMLEX_NVLINK_MAX_LANES = MXSMLEX_NVLINK_MAX_LINKS

_nvmlNvLinkErrorCounter_t = c_uint
MXSMLEX_NVLINK_ERROR_DL_REPLAY = 0
MXSMLEX_NVLINK_ERROR_DL_RECOVERY = 1
MXSMLEX_NVLINK_ERROR_DL_CRC_FLIT = 2
MXSMLEX_NVLINK_ERROR_DL_CRC_DATA = 3
MXSMLEX_NVLINK_ERROR_DL_ECC_DATA = 4
MXSMLEX_NVLINK_ERROR_COUNT = 5

_nvmlNvLinkEccLaneErrorCounter_t = c_uint
MXSMLEX_NVLINK_ERROR_DL_ECC_LANE0 = 0
MXSMLEX_NVLINK_ERROR_DL_ECC_LANE1 = 1
MXSMLEX_NVLINK_ERROR_DL_ECC_LANE2 = 2
MXSMLEX_NVLINK_ERROR_DL_ECC_LANE3 = 3
MXSMLEX_NVLINK_ERROR_DL_ECC_COUNT = 5

_nvmlNvLinkCapability_t = c_uint
MXSMLEX_NVLINK_CAP_P2P_SUPPORTED = 0
MXSMLEX_NVLINK_CAP_SYSMEM_ACCESS = 1
MXSMLEX_NVLINK_CAP_P2P_ATOMICS   = 2
MXSMLEX_NVLINK_CAP_SYSMEM_ATOMICS= 3
MXSMLEX_NVLINK_CAP_SLI_BRIDGE    = 4
MXSMLEX_NVLINK_CAP_VALID         = 5
MXSMLEX_NVLINK_CAP_COUNT         = 6

_nvmlNvLinkUtilizationCountPktTypes_t = c_uint
MXSMLEX_NVLINK_COUNTER_PKTFILTER_NOP        = 0x1
MXSMLEX_NVLINK_COUNTER_PKTFILTER_READ       = 0x2
MXSMLEX_NVLINK_COUNTER_PKTFILTER_WRITE      = 0x4
MXSMLEX_NVLINK_COUNTER_PKTFILTER_RATOM      = 0x8
MXSMLEX_NVLINK_COUNTER_PKTFILTER_NRATOM     = 0x10
MXSMLEX_NVLINK_COUNTER_PKTFILTER_FLUSH      = 0x20
MXSMLEX_NVLINK_COUNTER_PKTFILTER_RESPDATA   = 0x40
MXSMLEX_NVLINK_COUNTER_PKTFILTER_RESPNODATA = 0x80
MXSMLEX_NVLINK_COUNTER_PKTFILTER_ALL        = 0xFF

_nvmlNvLinkUtilizationCountUnits_t = c_uint
MXSMLEX_NVLINK_COUNTER_UNIT_CYCLES   = 0
MXSMLEX_NVLINK_COUNTER_UNIT_PACKETS  = 1
MXSMLEX_NVLINK_COUNTER_UNIT_BYTES    = 2
MXSMLEX_NVLINK_COUNTER_UNIT_RESERVED = 3
MXSMLEX_NVLINK_COUNTER_UNIT_COUNT    = 4

_nvmlNvLinkDeviceType_t = c_uint
MXSMLEX_NVLINK_DEVICE_TYPE_GPU     = 0x00
MXSMLEX_NVLINK_DEVICE_TYPE_IBMNPU  = 0x01
MXSMLEX_NVLINK_DEVICE_TYPE_SWITCH  = 0x02
MXSMLEX_NVLINK_DEVICE_TYPE_UNKNOWN = 0xFF

# These are deprecated, instead use _nvmlMemoryErrorType_t
_nvmlEccBitType_t = c_uint
MXSMLEX_SINGLE_BIT_ECC    = 0
MXSMLEX_DOUBLE_BIT_ECC    = 1
MXSMLEX_ECC_ERROR_TYPE_COUNT = 2

_nvmlEccCounterType_t = c_uint
MXSMLEX_VOLATILE_ECC      = 0
MXSMLEX_AGGREGATE_ECC     = 1
MXSMLEX_ECC_COUNTER_TYPE_COUNT = 2

_nvmlMemoryErrorType_t = c_uint
MXSMLEX_MEMORY_ERROR_TYPE_CORRECTED   = 0
MXSMLEX_MEMORY_ERROR_TYPE_UNCORRECTED = 1
MXSMLEX_MEMORY_ERROR_TYPE_COUNT       = 2

_nvmlClockType_t = c_uint
MXSMLEX_CLOCK_GRAPHICS  = 0
MXSMLEX_CLOCK_SM        = 1
MXSMLEX_CLOCK_MEM       = 2
MXSMLEX_CLOCK_VIDEO     = 3
MXSMLEX_CLOCK_COUNT     = 4

_nvmlClockId_t = c_uint
MXSMLEX_CLOCK_ID_CURRENT            = 0
MXSMLEX_CLOCK_ID_APP_CLOCK_TARGET   = 1
MXSMLEX_CLOCK_ID_APP_CLOCK_DEFAULT  = 2
MXSMLEX_CLOCK_ID_CUSTOMER_BOOST_MAX = 3
MXSMLEX_CLOCK_ID_COUNT              = 4

_nvmlDriverModel_t = c_uint
MXSMLEX_DRIVER_WDDM       = 0
MXSMLEX_DRIVER_WDM        = 1
MXSMLEX_DRIVER_MCDM       = 2

MXSMLEX_MAX_GPU_PERF_PSTATES = 16

_nvmlPstates_t = c_uint
MXSMLEX_PSTATE_0               = 0
MXSMLEX_PSTATE_1               = 1
MXSMLEX_PSTATE_2               = 2
MXSMLEX_PSTATE_3               = 3
MXSMLEX_PSTATE_4               = 4
MXSMLEX_PSTATE_5               = 5
MXSMLEX_PSTATE_6               = 6
MXSMLEX_PSTATE_7               = 7
MXSMLEX_PSTATE_8               = 8
MXSMLEX_PSTATE_9               = 9
MXSMLEX_PSTATE_10              = 10
MXSMLEX_PSTATE_11              = 11
MXSMLEX_PSTATE_12              = 12
MXSMLEX_PSTATE_13              = 13
MXSMLEX_PSTATE_14              = 14
MXSMLEX_PSTATE_15              = 15
MXSMLEX_PSTATE_UNKNOWN         = 32

_nvmlInforomObject_t = c_uint
MXSMLEX_INFOROM_OEM            = 0
MXSMLEX_INFOROM_ECC            = 1
MXSMLEX_INFOROM_POWER          = 2
MXSMLEX_INFOROM_COUNT          = 3

_nvmlReturn_t = c_uint
MXSMLEX_SUCCESS                         = 0
MXSMLEX_ERROR_UNINITIALIZED             = 1
MXSMLEX_ERROR_INVALID_ARGUMENT          = 2
MXSMLEX_ERROR_NOT_SUPPORTED             = 3
MXSMLEX_ERROR_NO_PERMISSION             = 4
MXSMLEX_ERROR_ALREADY_INITIALIZED       = 5
MXSMLEX_ERROR_NOT_FOUND                 = 6
MXSMLEX_ERROR_INSUFFICIENT_SIZE         = 7
MXSMLEX_ERROR_INSUFFICIENT_POWER        = 8
MXSMLEX_ERROR_DRIVER_NOT_LOADED         = 9
MXSMLEX_ERROR_TIMEOUT                   = 10
MXSMLEX_ERROR_IRQ_ISSUE                 = 11
MXSMLEX_ERROR_LIBRARY_NOT_FOUND         = 12
MXSMLEX_ERROR_FUNCTION_NOT_FOUND        = 13
MXSMLEX_ERROR_CORRUPTED_INFOROM         = 14
MXSMLEX_ERROR_GPU_IS_LOST               = 15
MXSMLEX_ERROR_RESET_REQUIRED            = 16
MXSMLEX_ERROR_OPERATING_SYSTEM          = 17
MXSMLEX_ERROR_LIB_RM_VERSION_MISMATCH   = 18
MXSMLEX_ERROR_IN_USE                    = 19
MXSMLEX_ERROR_MEMORY                    = 20
MXSMLEX_ERROR_NO_DATA                   = 21
MXSMLEX_ERROR_VGPU_ECC_NOT_SUPPORTED    = 22
MXSMLEX_ERROR_INSUFFICIENT_RESOURCES    = 23
MXSMLEX_ERROR_FREQ_NOT_SUPPORTED        = 24
MXSMLEX_ERROR_ARGUMENT_VERSION_MISMATCH = 25
MXSMLEX_ERROR_DEPRECATED                = 26
MXSMLEX_ERROR_NOT_READY                 = 27
MXSMLEX_ERROR_GPU_NOT_FOUND             = 28
MXSMLEX_ERROR_INVALID_STATE             = 29
MXSMLEX_ERROR_UNKNOWN                   = 999

_nvmlFanState_t = c_uint
MXSMLEX_FAN_NORMAL             = 0
MXSMLEX_FAN_FAILED             = 1

_nvmlFanControlPolicy_t = c_uint
MXSMLEX_FAN_POLICY_TEMPERATURE_CONTINOUS_SW = 0
MXSMLEX_FAN_POLICY_MANUAL                   = 1

_nvmlLedColor_t = c_uint
MXSMLEX_LED_COLOR_GREEN        = 0
MXSMLEX_LED_COLOR_AMBER        = 1

_nvmlGpuOperationMode_t = c_uint
MXSMLEX_GOM_ALL_ON                 = 0
MXSMLEX_GOM_COMPUTE                = 1
MXSMLEX_GOM_LOW_DP                 = 2

_nvmlPageRetirementCause_t = c_uint
MXSMLEX_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS = 0
MXSMLEX_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR           = 1
MXSMLEX_PAGE_RETIREMENT_CAUSE_COUNT                          = 2

_nvmlRestrictedAPI_t = c_uint
MXSMLEX_RESTRICTED_API_SET_APPLICATION_CLOCKS                = 0
MXSMLEX_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS               = 1
MXSMLEX_RESTRICTED_API_COUNT                                 = 2

_nvmlBridgeChipType_t = c_uint
MXSMLEX_BRIDGE_CHIP_PLX = 0
MXSMLEX_BRIDGE_CHIP_BRO4 = 1
MXSMLEX_MAX_PHYSICAL_BRIDGE = 128

_nvmlValueType_t = c_uint
MXSMLEX_VALUE_TYPE_DOUBLE = 0
MXSMLEX_VALUE_TYPE_UNSIGNED_INT = 1
MXSMLEX_VALUE_TYPE_UNSIGNED_LONG = 2
MXSMLEX_VALUE_TYPE_UNSIGNED_LONG_LONG = 3
MXSMLEX_VALUE_TYPE_SIGNED_LONG_LONG = 4
MXSMLEX_VALUE_TYPE_SIGNED_INT = 5
MXSMLEX_VALUE_TYPE_COUNT = 6

_nvmlPerfPolicyType_t = c_uint
MXSMLEX_PERF_POLICY_POWER = 0
MXSMLEX_PERF_POLICY_THERMAL = 1
MXSMLEX_PERF_POLICY_SYNC_BOOST = 2
MXSMLEX_PERF_POLICY_BOARD_LIMIT = 3
MXSMLEX_PERF_POLICY_LOW_UTILIZATION = 4
MXSMLEX_PERF_POLICY_RELIABILITY = 5
MXSMLEX_PERF_POLICY_TOTAL_APP_CLOCKS = 10
MXSMLEX_PERF_POLICY_TOTAL_BASE_CLOCKS = 11
MXSMLEX_PERF_POLICY_COUNT = 12

_nvmlEncoderQueryType_t = c_uint
MXSMLEX_ENCODER_QUERY_H264 = 0
MXSMLEX_ENCODER_QUERY_HEVC = 1
MXSMLEX_ENCODER_QUERY_AV1 = 2
MXSMLEX_ENCODER_QUERY_UNKNOWN = 255

_nvmlFBCSessionType_t = c_uint
MXSMLEX_FBC_SESSION_TYPE_UNKNOWN = 0
MXSMLEX_FBC_SESSION_TYPE_TOSYS = 1
MXSMLEX_FBC_SESSION_TYPE_CUDA = 2
MXSMLEX_FBC_SESSION_TYPE_VID = 3
MXSMLEX_FBC_SESSION_TYPE_HWENC = 4

_nvmlDetachGpuState_t = c_uint
MXSMLEX_DETACH_GPU_KEEP = 0
MXSMLEX_DETACH_GPU_REMOVE = 1

_nvmlPcieLinkState_t = c_uint
MXSMLEX_PCIE_LINK_KEEP = 0
MXSMLEX_PCIE_LINK_SHUT_DOWN = 1

_nvmlSamplingType_t = c_uint
MXSMLEX_TOTAL_POWER_SAMPLES = 0
MXSMLEX_GPU_UTILIZATION_SAMPLES = 1
MXSMLEX_MEMORY_UTILIZATION_SAMPLES = 2
MXSMLEX_ENC_UTILIZATION_SAMPLES = 3
MXSMLEX_DEC_UTILIZATION_SAMPLES = 4
MXSMLEX_PROCESSOR_CLK_SAMPLES = 5
MXSMLEX_MEMORY_CLK_SAMPLES = 6
MXSMLEX_MODULE_POWER_SAMPLES = 7
MXSMLEX_JPG_UTILIZATION_SAMPLES = 8
MXSMLEX_OFA_UTILIZATION_SAMPLES = 9
MXSMLEX_SAMPLINGTYPE_COUNT = 10

_nvmlPcieUtilCounter_t = c_uint
MXSMLEX_PCIE_UTIL_TX_BYTES = 0
MXSMLEX_PCIE_UTIL_RX_BYTES = 1
MXSMLEX_PCIE_UTIL_COUNT = 2

_nvmlGpuTopologyLevel_t = c_uint
MXSMLEX_TOPOLOGY_INTERNAL = 0
MXSMLEX_TOPOLOGY_SINGLE = 10
MXSMLEX_TOPOLOGY_MULTIPLE = 20
MXSMLEX_TOPOLOGY_HOSTBRIDGE = 30
MXSMLEX_TOPOLOGY_NODE = 40
MXSMLEX_TOPOLOGY_CPU = MXSMLEX_TOPOLOGY_NODE
MXSMLEX_TOPOLOGY_SYSTEM = 50

_nvmlGpuP2PCapsIndex_t = c_uint
MXSMLEX_P2P_CAPS_INDEX_READ = 0,
MXSMLEX_P2P_CAPS_INDEX_WRITE = 1
#MXSMLEX_P2P_CAPS_INDEX_MXLINK =2
NVML_P2P_CAPS_INDEX_NVLINK = 2
MXSMLEX_P2P_CAPS_INDEX_ATOMICS = 3
#
# MXSMLEX_P2P_CAPS_INDEX_PROP is deprecated.
# Use MXSMLEX_P2P_CAPS_INDEX_PCI instead.
#
MXSMLEX_P2P_CAPS_INDEX_PROP = 4
MXSMLEX_P2P_CAPS_INDEX_PCI = 4
MXSMLEX_P2P_CAPS_INDEX_UNKNOWN = 5

_nvmlGpuP2PStatus_t = c_uint
#MXSMLEX_P2P_STATUS_OK     = 0
NVML_P2P_STATUS_OK = 0
MXSMLEX_P2P_STATUS_CHIPSET_NOT_SUPPORED = 1
MXSMLEX_P2P_STATUS_CHIPSET_NOT_SUPPORTED = MXSMLEX_P2P_STATUS_CHIPSET_NOT_SUPPORED
MXSMLEX_P2P_STATUS_GPU_NOT_SUPPORTED = 2
MXSMLEX_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED =3
MXSMLEX_P2P_STATUS_DISABLED_BY_REGKEY =4
MXSMLEX_P2P_STATUS_NOT_SUPPORTED =5
MXSMLEX_P2P_STATUS_UNKNOWN =6

_nvmlDeviceArchitecture_t = c_uint
MXSMLEX_DEVICE_ARCH_KEPLER   = 2
MXSMLEX_DEVICE_ARCH_MAXWELL  = 3
MXSMLEX_DEVICE_ARCH_PASCAL   = 4
MXSMLEX_DEVICE_ARCH_VOLTA    = 5
MXSMLEX_DEVICE_ARCH_TURING   = 6
MXSMLEX_DEVICE_ARCH_AMPERE   = 7
MXSMLEX_DEVICE_ARCH_ADA      = 8
MXSMLEX_DEVICE_ARCH_HOPPER   = 9
MXSMLEX_DEVICE_ARCH_UNKNOWN  = 0xffffffff

# PCI bus Types
_nvmlBusType_t = c_uint
MXSMLEX_BUS_TYPE_UNKNOWN = 0
MXSMLEX_BUS_TYPE_PCI     = 1
MXSMLEX_BUS_TYPE_PCIE    = 2
MXSMLEX_BUS_TYPE_FPCI    = 3
MXSMLEX_BUS_TYPE_AGP     = 4

_nvmlPowerSource_t = c_uint
MXSMLEX_POWER_SOURCE_AC         = 0x00000000
MXSMLEX_POWER_SOURCE_BATTERY    = 0x00000001
MXSMLEX_POWER_SOURCE_UNDERSIZED = 0x00000002

_nvmlAdaptiveClockInfoStatus_t = c_uint
MXSMLEX_ADAPTIVE_CLOCKING_INFO_STATUS_DISABLED = 0x00000000
MXSMLEX_ADAPTIVE_CLOCKING_INFO_STATUS_ENABLED = 0x00000001

_nvmlClockLimitId_t = c_uint
MXSMLEX_CLOCK_LIMIT_ID_RANGE_START = 0xffffff00
MXSMLEX_CLOCK_LIMIT_ID_TDP         = 0xffffff01
MXSMLEX_CLOCK_LIMIT_ID_UNLIMITED   = 0xffffff02

_nvmlPcieLinkMaxSpeed_t = c_uint
MXSMLEX_PCIE_LINK_MAX_SPEED_INVALID   = 0x00000000
MXSMLEX_PCIE_LINK_MAX_SPEED_2500MBPS  = 0x00000001
MXSMLEX_PCIE_LINK_MAX_SPEED_5000MBPS  = 0x00000002
MXSMLEX_PCIE_LINK_MAX_SPEED_8000MBPS  = 0x00000003
MXSMLEX_PCIE_LINK_MAX_SPEED_16000MBPS = 0x00000004
MXSMLEX_PCIE_LINK_MAX_SPEED_32000MBPS = 0x00000005
MXSMLEX_PCIE_LINK_MAX_SPEED_64000MBPS = 0x00000006

_nvmlAffinityScope_t = c_uint
MXSMLEX_AFFINITY_SCOPE_NODE   = 0
MXSMLEX_AFFINITY_SCOPE_SOCKET = 1

# C preprocessor defined values
nvmlFlagDefault             = 0
nvmlFlagForce               = 1
MXSMLEX_INIT_FLAG_NO_GPUS      = 1
MXSMLEX_INIT_FLAG_NO_ATTACH    = 2

MXSMLEX_MAX_GPC_COUNT          = 32

# buffer size
MXSMLEX_DEVICE_INFOROM_VERSION_BUFFER_SIZE      = 16
MXSMLEX_DEVICE_UUID_BUFFER_SIZE                 = 80
MXSMLEX_DEVICE_UUID_V2_BUFFER_SIZE              = 96
MXSMLEX_SYSTEM_DRIVER_VERSION_BUFFER_SIZE       = 80
MXSMLEX_SYSTEM_MXSMLEX_VERSION_BUFFER_SIZE         = 80
MXSMLEX_DEVICE_NAME_BUFFER_SIZE                 = 64
MXSMLEX_DEVICE_NAME_V2_BUFFER_SIZE              = 96
MXSMLEX_DEVICE_SERIAL_BUFFER_SIZE               = 30
MXSMLEX_DEVICE_PART_NUMBER_BUFFER_SIZE          = 80
MXSMLEX_DEVICE_GPU_PART_NUMBER_BUFFER_SIZE      = 80
MXSMLEX_DEVICE_VBIOS_VERSION_BUFFER_SIZE        = 32
MXSMLEX_DEVICE_PCI_BUS_ID_BUFFER_SIZE           = 32
MXSMLEX_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE        = 16
MXSMLEX_GRID_LICENSE_BUFFER_SIZE                = 128
MXSMLEX_VGPU_NAME_BUFFER_SIZE                   = 64
MXSMLEX_GRID_LICENSE_FEATURE_MAX_COUNT          = 3
MXSMLEX_VGPU_METADATA_OPAQUE_DATA_SIZE          = sizeof(c_uint) + 256
MXSMLEX_VGPU_PGPU_METADATA_OPAQUE_DATA_SIZE     = 256
MXSMLEX_DEVICE_GPU_FRU_PART_NUMBER_BUFFER_SIZE  = 0x14 # NV2080_GPU_MAX_PRODUCT_PART_NUMBER_LENGTH

# Format strings
MXSMLEX_DEVICE_PCI_BUS_ID_LEGACY_FMT   = "%04X:%02X:%02X.0"
MXSMLEX_DEVICE_PCI_BUS_ID_FMT          = "%08X:%02X:%02X.0"

MXSMLEX_VALUE_NOT_AVAILABLE_ulonglong = c_ulonglong(-1)
MXSMLEX_VALUE_NOT_AVAILABLE_uint = c_uint(-1)

'''
 Field Identifiers.

 All Identifiers pertain to a device. Each ID is only used once and is guaranteed never to change.
'''
MXSMLEX_FI_DEV_ECC_CURRENT          = 1   # Current ECC mode. 1=Active. 0=Inactive
MXSMLEX_FI_DEV_ECC_PENDING          = 2   # Pending ECC mode. 1=Active. 0=Inactive

#ECC Count Totals
MXSMLEX_FI_DEV_ECC_SBE_VOL_TOTAL    = 3   # Total single bit volatile ECC errors
MXSMLEX_FI_DEV_ECC_DBE_VOL_TOTAL    = 4   # Total double bit volatile ECC errors
MXSMLEX_FI_DEV_ECC_SBE_AGG_TOTAL    = 5   # Total single bit aggregate (persistent) ECC errors
MXSMLEX_FI_DEV_ECC_DBE_AGG_TOTAL    = 6   # Total double bit aggregate (persistent) ECC errors
#Individual ECC locations
MXSMLEX_FI_DEV_ECC_SBE_VOL_L1       = 7   # L1 cache single bit volatile ECC errors
MXSMLEX_FI_DEV_ECC_DBE_VOL_L1       = 8   # L1 cache double bit volatile ECC errors
MXSMLEX_FI_DEV_ECC_SBE_VOL_L2       = 9   # L2 cache single bit volatile ECC errors
MXSMLEX_FI_DEV_ECC_DBE_VOL_L2       = 10  # L2 cache double bit volatile ECC errors
MXSMLEX_FI_DEV_ECC_SBE_VOL_DEV      = 11  # Device memory single bit volatile ECC errors
MXSMLEX_FI_DEV_ECC_DBE_VOL_DEV      = 12  # Device memory double bit volatile ECC errors
MXSMLEX_FI_DEV_ECC_SBE_VOL_REG      = 13  # Register file single bit volatile ECC errors
MXSMLEX_FI_DEV_ECC_DBE_VOL_REG      = 14  # Register file double bit volatile ECC errors
MXSMLEX_FI_DEV_ECC_SBE_VOL_TEX      = 15  # Texture memory single bit volatile ECC errors
MXSMLEX_FI_DEV_ECC_DBE_VOL_TEX      = 16  # Texture memory double bit volatile ECC errors
MXSMLEX_FI_DEV_ECC_DBE_VOL_CBU      = 17  # CBU double bit volatile ECC errors
MXSMLEX_FI_DEV_ECC_SBE_AGG_L1       = 18  # L1 cache single bit aggregate (persistent) ECC errors
MXSMLEX_FI_DEV_ECC_DBE_AGG_L1       = 19  # L1 cache double bit aggregate (persistent) ECC errors
MXSMLEX_FI_DEV_ECC_SBE_AGG_L2       = 20  # L2 cache single bit aggregate (persistent) ECC errors
MXSMLEX_FI_DEV_ECC_DBE_AGG_L2       = 21  # L2 cache double bit aggregate (persistent) ECC errors
MXSMLEX_FI_DEV_ECC_SBE_AGG_DEV      = 22  # Device memory single bit aggregate (persistent) ECC errors
MXSMLEX_FI_DEV_ECC_DBE_AGG_DEV      = 23  # Device memory double bit aggregate (persistent) ECC errors
MXSMLEX_FI_DEV_ECC_SBE_AGG_REG      = 24  # Register File single bit aggregate (persistent) ECC errors
MXSMLEX_FI_DEV_ECC_DBE_AGG_REG      = 25  # Register File double bit aggregate (persistent) ECC errors
MXSMLEX_FI_DEV_ECC_SBE_AGG_TEX      = 26  # Texture memory single bit aggregate (persistent) ECC errors
MXSMLEX_FI_DEV_ECC_DBE_AGG_TEX      = 27  # Texture memory double bit aggregate (persistent) ECC errors
MXSMLEX_FI_DEV_ECC_DBE_AGG_CBU      = 28  # CBU double bit aggregate ECC errors

# Page Retirement
MXSMLEX_FI_DEV_RETIRED_SBE          = 29  # Number of retired pages because of single bit errors
MXSMLEX_FI_DEV_RETIRED_DBE          = 30  # Number of retired pages because of double bit errors
MXSMLEX_FI_DEV_RETIRED_PENDING      = 31  # If any pages are pending retirement. 1=yes. 0=no.

# NvLink Flit Error Counters
MXSMLEX_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L0   = 32 # NVLink flow control CRC  Error Counter for Lane 0
MXSMLEX_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L1   = 33 # NVLink flow control CRC  Error Counter for Lane 1
MXSMLEX_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L2   = 34 # NVLink flow control CRC  Error Counter for Lane 2
MXSMLEX_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L3   = 35 # NVLink flow control CRC  Error Counter for Lane 3
MXSMLEX_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L4   = 36 # NVLink flow control CRC  Error Counter for Lane 4
MXSMLEX_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L5   = 37 # NVLink flow control CRC  Error Counter for Lane 5
MXSMLEX_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL = 38 # NVLink flow control CRC  Error Counter total for all Lanes

# NvLink CRC Data Error Counters
MXSMLEX_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L0   = 39 # NVLink data CRC Error Counter for Lane 0
MXSMLEX_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L1   = 40 # NVLink data CRC Error Counter for Lane 1
MXSMLEX_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L2   = 41 # NVLink data CRC Error Counter for Lane 2
MXSMLEX_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L3   = 42 # NVLink data CRC Error Counter for Lane 3
MXSMLEX_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L4   = 43 # NVLink data CRC Error Counter for Lane 4
MXSMLEX_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L5   = 44 # NVLink data CRC Error Counter for Lane 5
MXSMLEX_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL = 45 # NvLink data CRC Error Counter total for all Lanes

# NvLink Replay Error Counters
MXSMLEX_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0     = 46 # NVLink Replay Error Counter for Lane 0
MXSMLEX_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1     = 47 # NVLink Replay Error Counter for Lane 1
MXSMLEX_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2     = 48 # NVLink Replay Error Counter for Lane 2
MXSMLEX_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3     = 49 # NVLink Replay Error Counter for Lane 3
MXSMLEX_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4     = 50 # NVLink Replay Error Counter for Lane 4
MXSMLEX_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5     = 51 # NVLink Replay Error Counter for Lane 5
MXSMLEX_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL  = 52 # NVLink Replay Error Counter total for all Lanes

# NvLink Recovery Error Counters
MXSMLEX_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0   = 53 # NVLink Recovery Error Counter for Lane 0
MXSMLEX_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1   = 54 # NVLink Recovery Error Counter for Lane 1
MXSMLEX_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2   = 55 # NVLink Recovery Error Counter for Lane 2
MXSMLEX_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3   = 56 # NVLink Recovery Error Counter for Lane 3
MXSMLEX_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4   = 57 # NVLink Recovery Error Counter for Lane 4
MXSMLEX_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5   = 58 # NVLink Recovery Error Counter for Lane 5
MXSMLEX_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL = 59 # NVLink Recovery Error Counter total for all Lanes

# NvLink Bandwidth Counters
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C0_L0    = 60 # NVLink Bandwidth Counter for Counter Set 0, Lane 0
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C0_L1    = 61 # NVLink Bandwidth Counter for Counter Set 0, Lane 1
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C0_L2    = 62 # NVLink Bandwidth Counter for Counter Set 0, Lane 2
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C0_L3    = 63 # NVLink Bandwidth Counter for Counter Set 0, Lane 3
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C0_L4    = 64 # NVLink Bandwidth Counter for Counter Set 0, Lane 4
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C0_L5    = 65 # NVLink Bandwidth Counter for Counter Set 0, Lane 5
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C0_TOTAL = 66 # NVLink Bandwidth Counter Total for Counter Set 0, All Lanes

# NvLink Bandwidth Counters
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C1_L0    = 67 # NVLink Bandwidth Counter for Counter Set 1, Lane 0
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C1_L1    = 68 # NVLink Bandwidth Counter for Counter Set 1, Lane 1
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C1_L2    = 69 # NVLink Bandwidth Counter for Counter Set 1, Lane 2
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C1_L3    = 70 # NVLink Bandwidth Counter for Counter Set 1, Lane 3
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C1_L4    = 71 # NVLink Bandwidth Counter for Counter Set 1, Lane 4
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C1_L5    = 72 # NVLink Bandwidth Counter for Counter Set 1, Lane 5
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C1_TOTAL = 73 # NVLink Bandwidth Counter Total for Counter Set 1, All Lanes

# Perf Policy Counters
MXSMLEX_FI_DEV_PERF_POLICY_POWER             = 74   # Perf Policy Counter for Power Policy
MXSMLEX_FI_DEV_PERF_POLICY_THERMAL           = 75   # Perf Policy Counter for Thermal Policy
MXSMLEX_FI_DEV_PERF_POLICY_SYNC_BOOST        = 76   # Perf Policy Counter for Sync boost Policy
MXSMLEX_FI_DEV_PERF_POLICY_BOARD_LIMIT       = 77   # Perf Policy Counter for Board Limit
MXSMLEX_FI_DEV_PERF_POLICY_LOW_UTILIZATION   = 78   # Perf Policy Counter for Low GPU Utilization Policy
MXSMLEX_FI_DEV_PERF_POLICY_RELIABILITY       = 79   # Perf Policy Counter for Reliability Policy
MXSMLEX_FI_DEV_PERF_POLICY_TOTAL_APP_CLOCKS  = 80   # Perf Policy Counter for Total App Clock Policy
MXSMLEX_FI_DEV_PERF_POLICY_TOTAL_BASE_CLOCKS = 81   # Perf Policy Counter for Total Base Clocks Policy

# Memory temperatures
MXSMLEX_FI_DEV_MEMORY_TEMP  = 82 # Memory temperature for the device

# Energy Counter
MXSMLEX_FI_DEV_TOTAL_ENERGY_CONSUMPTION = 83 # Total energy consumption for the GPU in mJ since the driver was last reloaded

# NVLink Speed
MXSMLEX_FI_DEV_NVLINK_SPEED_MBPS_L0     = 84
MXSMLEX_FI_DEV_NVLINK_SPEED_MBPS_L1     = 85
MXSMLEX_FI_DEV_NVLINK_SPEED_MBPS_L2     = 86
MXSMLEX_FI_DEV_NVLINK_SPEED_MBPS_L3     = 87
MXSMLEX_FI_DEV_NVLINK_SPEED_MBPS_L4     = 88
MXSMLEX_FI_DEV_NVLINK_SPEED_MBPS_L5     = 89
MXSMLEX_FI_DEV_NVLINK_SPEED_MBPS_COMMON = 90

# NVLink Link Count
MXSMLEX_FI_DEV_NVLINK_LINK_COUNT = 91

# Page Retirement pending fields
MXSMLEX_FI_DEV_RETIRED_PENDING_SBE = 92
MXSMLEX_FI_DEV_RETIRED_PENDING_DBE = 93

# PCIe replay and replay rollover counters
MXSMLEX_FI_DEV_PCIE_REPLAY_COUNTER = 94
MXSMLEX_FI_DEV_PCIE_REPLAY_ROLLOVER_COUNTER = 95

# NvLink Flit Error Counters
MXSMLEX_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L6   = 96 # NVLink flow control CRC  Error Counter for Lane 6
MXSMLEX_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L7   = 97 # NVLink flow control CRC  Error Counter for Lane 7
MXSMLEX_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L8   = 98 # NVLink flow control CRC  Error Counter for Lane 8
MXSMLEX_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L9   = 99 # NVLink flow control CRC  Error Counter for Lane 9
MXSMLEX_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L10  = 100 # NVLink flow control CRC  Error Counter for Lane 10
MXSMLEX_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L11  = 101 # NVLink flow control CRC  Error Counter for Lane 11

# NvLink CRC Data Error Counters
MXSMLEX_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L6   = 102 # NVLink data CRC Error Counter for Lane 6
MXSMLEX_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L7   = 103 # NVLink data CRC Error Counter for Lane 7
MXSMLEX_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L8   = 104 # NVLink data CRC Error Counter for Lane 8
MXSMLEX_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L9   = 105 # NVLink data CRC Error Counter for Lane 9
MXSMLEX_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L10  = 106 # NVLink data CRC Error Counter for Lane 10
MXSMLEX_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L11  = 107 # NVLink data CRC Error Counter for Lane 11

# NvLink Replay Error Counters
MXSMLEX_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6     = 108 # NVLink Replay Error Counter for Lane 6
MXSMLEX_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7     = 109 # NVLink Replay Error Counter for Lane 7
MXSMLEX_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8     = 110 # NVLink Replay Error Counter for Lane 8
MXSMLEX_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9     = 111 # NVLink Replay Error Counter for Lane 9
MXSMLEX_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10    = 112 # NVLink Replay Error Counter for Lane 10
MXSMLEX_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11    = 113 # NVLink Replay Error Counter for Lane 11

# NvLink Recovery Error Counters
MXSMLEX_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6   = 114 # NVLink Recovery Error Counter for Lane 6
MXSMLEX_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7   = 115 # NVLink Recovery Error Counter for Lane 7
MXSMLEX_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8   = 116 # NVLink Recovery Error Counter for Lane 8
MXSMLEX_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9   = 117 # NVLink Recovery Error Counter for Lane 9
MXSMLEX_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10  = 118 # NVLink Recovery Error Counter for Lane 10
MXSMLEX_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11  = 119 # NVLink Recovery Error Counter for Lane 11

# NvLink Bandwidth Counters
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C0_L6    = 120 # NVLink Bandwidth Counter for Counter Set 0, Lane 6
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C0_L7    = 121 # NVLink Bandwidth Counter for Counter Set 0, Lane 7
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C0_L8    = 122 # NVLink Bandwidth Counter for Counter Set 0, Lane 8
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C0_L9    = 123 # NVLink Bandwidth Counter for Counter Set 0, Lane 9
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C0_L10   = 124 # NVLink Bandwidth Counter for Counter Set 0, Lane 10
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C0_L11   = 125 # NVLink Bandwidth Counter for Counter Set 0, Lane 11

# NvLink Bandwidth Counters
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C1_L6    = 126 # NVLink Bandwidth Counter for Counter Set 1, Lane 6
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C1_L7    = 127 # NVLink Bandwidth Counter for Counter Set 1, Lane 7
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C1_L8    = 128 # NVLink Bandwidth Counter for Counter Set 1, Lane 8
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C1_L9    = 129 # NVLink Bandwidth Counter for Counter Set 1, Lane 9
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C1_L10   = 130 # NVLink Bandwidth Counter for Counter Set 1, Lane 10
MXSMLEX_FI_DEV_NVLINK_BANDWIDTH_C1_L11   = 131 # NVLink Bandwidth Counter for Counter Set 1, Lane 11

# NVLink Speed
MXSMLEX_FI_DEV_NVLINK_SPEED_MBPS_L6     = 132
MXSMLEX_FI_DEV_NVLINK_SPEED_MBPS_L7     = 133
MXSMLEX_FI_DEV_NVLINK_SPEED_MBPS_L8     = 134
MXSMLEX_FI_DEV_NVLINK_SPEED_MBPS_L9     = 135
MXSMLEX_FI_DEV_NVLINK_SPEED_MBPS_L10    = 136
MXSMLEX_FI_DEV_NVLINK_SPEED_MBPS_L11    = 137

# NVLink Throughput Counters
MXSMLEX_FI_DEV_NVLINK_THROUGHPUT_DATA_TX = 138 # NVLink TX Data throughput in KiB
MXSMLEX_FI_DEV_NVLINK_THROUGHPUT_DATA_RX = 139 # NVLink RX Data throughput in KiB
MXSMLEX_FI_DEV_NVLINK_THROUGHPUT_RAW_TX  = 140 # NVLink TX Data + protocol overhead in KiB
MXSMLEX_FI_DEV_NVLINK_THROUGHPUT_RAW_RX  = 141 # NVLink RX Data + protocol overhead in KiB

# Row Remapper
MXSMLEX_FI_DEV_REMAPPED_COR        = 142
MXSMLEX_FI_DEV_REMAPPED_UNC        = 143
MXSMLEX_FI_DEV_REMAPPED_PENDING    = 144
MXSMLEX_FI_DEV_REMAPPED_FAILURE    = 145

#Remote device NVLink ID
MXSMLEX_FI_DEV_NVLINK_REMOTE_NVLINK_ID = 146

# Number of NVLinks connected to NVSwitch
MXSMLEX_FI_DEV_NVSWITCH_CONNECTED_LINK_COUNT = 147

# NvLink ECC Data Error Counters
MXSMLEX_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L0    = 148 #< NVLink data ECC Error Counter for Link 0
MXSMLEX_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L1    = 149 #< NVLink data ECC Error Counter for Link 1
MXSMLEX_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L2    = 150 #< NVLink data ECC Error Counter for Link 2
MXSMLEX_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L3    = 151 #< NVLink data ECC Error Counter for Link 3
MXSMLEX_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L4    = 152 #< NVLink data ECC Error Counter for Link 4
MXSMLEX_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L5    = 153 #< NVLink data ECC Error Counter for Link 5
MXSMLEX_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L6    = 154 #< NVLink data ECC Error Counter for Link 6
MXSMLEX_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L7    = 155 #< NVLink data ECC Error Counter for Link 7
MXSMLEX_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L8    = 156 #< NVLink data ECC Error Counter for Link 8
MXSMLEX_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L9    = 157 #< NVLink data ECC Error Counter for Link 9
MXSMLEX_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L10   = 158 #< NVLink data ECC Error Counter for Link 10
MXSMLEX_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_L11   = 159 #< NVLink data ECC Error Counter for Link 11
MXSMLEX_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_TOTAL = 160 #< NvLink data ECC Error Counter total for all Links

MXSMLEX_FI_DEV_NVLINK_ERROR_DL_REPLAY            = 161
MXSMLEX_FI_DEV_NVLINK_ERROR_DL_RECOVERY          = 162
MXSMLEX_FI_DEV_NVLINK_ERROR_DL_CRC               = 163
MXSMLEX_FI_DEV_NVLINK_GET_SPEED                  = 164
MXSMLEX_FI_DEV_NVLINK_GET_STATE                  = 165
MXSMLEX_FI_DEV_NVLINK_GET_VERSION                = 166

MXSMLEX_FI_DEV_NVLINK_GET_POWER_STATE            = 167
MXSMLEX_FI_DEV_NVLINK_GET_POWER_THRESHOLD        = 168

MXSMLEX_FI_DEV_PCIE_L0_TO_RECOVERY_COUNTER       = 169

MXSMLEX_FI_DEV_C2C_LINK_COUNT                    = 170
MXSMLEX_FI_DEV_C2C_LINK_GET_STATUS               = 171
MXSMLEX_FI_DEV_C2C_LINK_GET_MAX_BW               = 172

MXSMLEX_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS     = 173
MXSMLEX_FI_DEV_PCIE_COUNT_NAKS_RECEIVED          = 174
MXSMLEX_FI_DEV_PCIE_COUNT_RECEIVER_ERROR         = 175
MXSMLEX_FI_DEV_PCIE_COUNT_BAD_TLP                = 176
MXSMLEX_FI_DEV_PCIE_COUNT_NAKS_SENT              = 177
MXSMLEX_FI_DEV_PCIE_COUNT_BAD_DLLP               = 178
MXSMLEX_FI_DEV_PCIE_COUNT_NON_FATAL_ERROR        = 179
MXSMLEX_FI_DEV_PCIE_COUNT_FATAL_ERROR            = 180
MXSMLEX_FI_DEV_PCIE_COUNT_UNSUPPORTED_REQ        = 181
MXSMLEX_FI_DEV_PCIE_COUNT_LCRC_ERROR             = 182
MXSMLEX_FI_DEV_PCIE_COUNT_LANE_ERROR             = 183

MXSMLEX_FI_DEV_IS_RESETLESS_MIG_SUPPORTED        = 184

MXSMLEX_FI_DEV_POWER_AVERAGE                     = 185
MXSMLEX_FI_DEV_POWER_INSTANT                     = 186
MXSMLEX_FI_DEV_POWER_MIN_LIMIT                   = 187
MXSMLEX_FI_DEV_POWER_MAX_LIMIT                   = 188
MXSMLEX_FI_DEV_POWER_DEFAULT_LIMIT               = 189
MXSMLEX_FI_DEV_POWER_CURRENT_LIMIT               = 190
MXSMLEX_FI_DEV_ENERGY                            = 191
MXSMLEX_FI_DEV_POWER_REQUESTED_LIMIT             = 192

MXSMLEX_FI_DEV_TEMPERATURE_SHUTDOWN_TLIMIT       = 193
MXSMLEX_FI_DEV_TEMPERATURE_SLOWDOWN_TLIMIT       = 194
MXSMLEX_FI_DEV_TEMPERATURE_MEM_MAX_TLIMIT        = 195
MXSMLEX_FI_DEV_TEMPERATURE_GPU_MAX_TLIMIT        = 196

MXSMLEX_FI_DEV_IS_MIG_MODE_INDEPENDENT_MIG_QUERY_CAPABLE   = 199

MXSMLEX_FI_MAX = 200 # One greater than the largest field ID defined above


## Enums needed for the method nvmlDeviceGetVirtualizationMode and nvmlDeviceSetVirtualizationMode
MXSMLEX_GPU_VIRTUALIZATION_MODE_NONE        = 0  # Represents Bare Metal GPU
MXSMLEX_GPU_VIRTUALIZATION_MODE_PASSTHROUGH = 1  # Device is associated with GPU-Passthorugh
MXSMLEX_GPU_VIRTUALIZATION_MODE_VGPU        = 2  # Device is associated with vGPU inside virtual machine.
MXSMLEX_GPU_VIRTUALIZATION_MODE_HOST_VGPU   = 3  # Device is associated with VGX hypervisor in vGPU mode
MXSMLEX_GPU_VIRTUALIZATION_MODE_HOST_VSGA   = 4  # Device is associated with VGX hypervisor in vSGA mode

## Lib loading ##
nvmlLib = None
libLoadLock = threading.Lock()
_nvmlLib_refcount = 0 # Incremented on each nvmlInit and decremented on nvmlShutdown

## vGPU Management
_nvmlVgpuTypeId_t   = c_uint
_nvmlVgpuInstance_t = c_uint

_nvmlVgpuVmIdType_t = c_uint
MXSMLEX_VGPU_VM_ID_DOMAIN_ID    = 0
MXSMLEX_VGPU_VM_ID_UUID         = 1

_nvmlGridLicenseFeatureCode_t = c_uint
MXSMLEX_GRID_LICENSE_FEATURE_CODE_UNKNOWN      = 0
MXSMLEX_GRID_LICENSE_FEATURE_CODE_VGPU         = 1
MXSMLEX_GRID_LICENSE_FEATURE_CODE_NVIDIA_RTX   = 2
MXSMLEX_GRID_LICENSE_FEATURE_CODE_VWORKSTATION = 2 # deprecated, use MXSMLEX_GRID_LICENSE_FEATURE_CODE_NVIDIA_RTX.
MXSMLEX_GRID_LICENSE_FEATURE_CODE_GAMING       = 3
MXSMLEX_GRID_LICENSE_FEATURE_CODE_COMPUTE      = 4

_nvmlGridLicenseExpiryStatus_t = c_uint8
MXSMLEX_GRID_LICENSE_EXPIRY_NOT_AVAILABLE    = 0,   # Expiry information not available
MXSMLEX_GRID_LICENSE_EXPIRY_INVALID          = 1,   # Invalid expiry or error fetching expiry
MXSMLEX_GRID_LICENSE_EXPIRY_VALID            = 2,   # Valid expiry
MXSMLEX_GRID_LICENSE_EXPIRY_NOT_APPLICABLE   = 3,   # Expiry not applicable
MXSMLEX_GRID_LICENSE_EXPIRY_PERMANENT        = 4,   # Permanent expiry

_nvmlVgpuCapability_t = c_uint
MXSMLEX_VGPU_CAP_NVLINK_P2P                    = 0  # vGPU P2P over NVLink is supported
MXSMLEX_VGPU_CAP_GPUDIRECT                     = 1  # GPUDirect capability is supported
MXSMLEX_VGPU_CAP_MULTI_VGPU_EXCLUSIVE          = 2  # vGPU profile cannot be mixed with other vGPU profiles in same VM
MXSMLEX_VGPU_CAP_EXCLUSIVE_TYPE                = 3  # vGPU profile cannot run on a GPU alongside other profiles of different type
MXSMLEX_VGPU_CAP_EXCLUSIVE_SIZE                = 4  # vGPU profile cannot run on a GPU alongside other profiles of different size
MXSMLEX_VGPU_CAP_COUNT                         = 5

_nvmlVgpuDriverCapability_t = c_uint
MXSMLEX_VGPU_DRIVER_CAP_HETEROGENEOUS_MULTI_VGPU          = 0  # Supports mixing of different vGPU profiles within one guest VM
MXSMLEX_VGPU_DRIVER_CAP_COUNT                             = 1

_nvmlDeviceVgpuCapability_t = c_uint
MXSMLEX_DEVICE_VGPU_CAP_FRACTIONAL_MULTI_VGPU             = 0  # Query if the fractional vGPU profiles on this GPU can be used in multi-vGPU configurations
MXSMLEX_DEVICE_VGPU_CAP_HETEROGENEOUS_TIMESLICE_PROFILES  = 1  # Query if the GPU supports concurrent execution of timesliced vGPU profiles of differing types
MXSMLEX_DEVICE_VGPU_CAP_HETEROGENEOUS_TIMESLICE_SIZES     = 2  # Query if the GPU supports concurrent execution of timesliced vGPU profiles of differing framebuffer sizes
MXSMLEX_DEVICE_VGPU_CAP_READ_DEVICE_BUFFER_BW             = 3  # Query the GPU's read_device_buffer expected bandwidth capacity in megabytes per second
MXSMLEX_DEVICE_VGPU_CAP_WRITE_DEVICE_BUFFER_BW            = 4  # Query the GPU's write_device_buffer expected bandwidth capacity in megabytes per second
MXSMLEX_DEVICE_VGPU_CAP_DEVICE_STREAMING                  = 5  # Query if vGPU profiles on the GPU supports migration data streaming
MXSMLEX_DEVICE_VGPU_CAP_MINI_QUARTER_GPU                  = 6  # Set/Get support of mini-quarter vGPU profiles
MXSMLEX_DEVICE_VGPU_CAP_COMPUTE_MEDIA_ENGINE_GPU          = 7  # Set/Get support for compute media engine vGPU profiles
MXSMLEX_DEVICE_VGPU_CAP_COUNT                             = 8

_nvmlVgpuGuestInfoState_t = c_uint
MXSMLEX_VGPU_INSTANCE_GUEST_INFO_STATE_UNINITIALIZED = 0
MXSMLEX_VGPU_INSTANCE_GUEST_INFO_STATE_INITIALIZED   = 1

_nvmlVgpuVmCompatibility_t = c_uint
MXSMLEX_VGPU_VM_COMPATIBILITY_NONE         = 0x0
MXSMLEX_VGPU_VM_COMPATIBILITY_COLD         = 0x1
MXSMLEX_VGPU_VM_COMPATIBILITY_HIBERNATE    = 0x2
MXSMLEX_VGPU_VM_COMPATIBILITY_SLEEP        = 0x4
MXSMLEX_VGPU_VM_COMPATIBILITY_LIVE         = 0x8

_nvmlVgpuPgpuCompatibilityLimitCode_t = c_uint
MXSMLEX_VGPU_COMPATIBILITY_LIMIT_NONE          = 0x0
MXSMLEX_VGPU_COMPATIBILITY_LIMIT_HOST_DRIVER   = 0x1
MXSMLEX_VGPU_COMPATIBILITY_LIMIT_GUEST_DRIVER  = 0x2
MXSMLEX_VGPU_COMPATIBILITY_LIMIT_GPU           = 0x4
MXSMLEX_VGPU_COMPATIBILITY_LIMIT_OTHER         = 0x80000000

_nvmlHostVgpuMode_t = c_uint
MXSMLEX_HOST_VGPU_MODE_NON_SRIOV   = 0
MXSMLEX_HOST_VGPU_MODE_SRIOV       = 1

_nvmlConfComputeGpusReadyState_t = c_uint
MXSMLEX_CC_ACCEPTING_CLIENT_REQUESTS_FALSE = 0
MXSMLEX_CC_ACCEPTING_CLIENT_REQUESTS_TRUE = 1

_nvmlConfComputeGpuCaps_t = c_uint
MXSMLEX_CC_SYSTEM_GPUS_CC_NOT_CAPABLE = 0
MXSMLEX_CC_SYSTEM_GPUS_CC_CAPABLE = 1

_nvmlConfComputeCpuCaps_t = c_uint
MXSMLEX_CC_SYSTEM_CPU_CAPS_NONE = 0
MXSMLEX_CC_SYSTEM_CPU_CAPS_AMD_SEV = 1
MXSMLEX_CC_SYSTEM_CPU_CAPS_INTEL_TDX = 2

_nvmlConfComputeDevToolsMode_t = c_uint
MXSMLEX_CC_SYSTEM_DEVTOOLS_MODE_OFF = 0
MXSMLEX_CC_SYSTEM_DEVTOOLS_MODE_ON = 1
 
MXSMLEX_CC_SYSTEM_ENVIRONMENT_UNAVAILABLE = 0
MXSMLEX_CC_SYSTEM_ENVIRONMENT_SIM = 1
MXSMLEX_CC_SYSTEM_ENVIRONMENT_PROD = 2
 
_nvmlConfComputeCcFeature_t = c_uint
MXSMLEX_CC_SYSTEM_FEATURE_DISABLED = 0
MXSMLEX_CC_SYSTEM_FEATURE_ENABLED = 1

_nvmlConfComputeCcKeyRotationThreshAttackerAdv_t = c_uint
MXSMLEX_CC_KEY_ROTATION_THRESH_ATTACKER_ADVANTAGE_MIN = 50
MXSMLEX_CC_KEY_ROTATION_THRESH_ATTACKER_ADVANTAGE_MAX = 75

# GSP firmware
MXSMLEX_GSP_FIRMWARE_VERSION_BUF_SIZE = 0x40

class MXSMLEXLibraryMismatchError(Exception):
    pass

## Error Checking ##
class NVMLError(Exception):
    _valClassMapping = dict()
    # List of currently known error codes
    _errcode_to_string = {
        MXSMLEX_ERROR_UNINITIALIZED:       "Uninitialized",
        MXSMLEX_ERROR_INVALID_ARGUMENT:    "Invalid Argument",
        MXSMLEX_ERROR_NOT_SUPPORTED:       "Not Supported",
        MXSMLEX_ERROR_NO_PERMISSION:       "Insufficient Permissions",
        MXSMLEX_ERROR_ALREADY_INITIALIZED: "Already Initialized",
        MXSMLEX_ERROR_NOT_FOUND:           "Not Found",
        MXSMLEX_ERROR_INSUFFICIENT_SIZE:   "Insufficient Size",
        MXSMLEX_ERROR_INSUFFICIENT_POWER:  "Insufficient External Power",
        MXSMLEX_ERROR_DRIVER_NOT_LOADED:   "Driver Not Loaded",
        MXSMLEX_ERROR_TIMEOUT:             "Timeout",
        MXSMLEX_ERROR_IRQ_ISSUE:           "Interrupt Request Issue",
        MXSMLEX_ERROR_LIBRARY_NOT_FOUND:   "MXSMLEX Shared Library Not Found",
        MXSMLEX_ERROR_FUNCTION_NOT_FOUND:  "Function Not Found",
        MXSMLEX_ERROR_CORRUPTED_INFOROM:   "Corrupted infoROM",
        MXSMLEX_ERROR_GPU_IS_LOST:         "GPU is lost",
        MXSMLEX_ERROR_RESET_REQUIRED:      "GPU requires restart",
        MXSMLEX_ERROR_OPERATING_SYSTEM:    "The operating system has blocked the request.",
        MXSMLEX_ERROR_LIB_RM_VERSION_MISMATCH: "RM has detected an MXSMLEX/RM version mismatch.",
        MXSMLEX_ERROR_MEMORY:              "Insufficient Memory",
        MXSMLEX_ERROR_UNKNOWN:             "Unknown Error",
        }
    def __new__(typ, value):
        '''
        Maps value to a proper subclass of NVMLError.
        See _extractNVMLErrorsAsClasses function for more details
        '''
        if typ == NVMLError:
            typ = NVMLError._valClassMapping.get(value, typ)
        obj = Exception.__new__(typ)
        obj.value = value
        return obj
    def __str__(self):
        try:
            if self.value not in NVMLError._errcode_to_string:
                NVMLError._errcode_to_string[self.value] = str(nvmlErrorString(self.value))
            return NVMLError._errcode_to_string[self.value]
        except NVMLError:
            return "MXSMLEX Error with code %d" % self.value
    def __eq__(self, other):
        return self.value == other.value

def nvmlExceptionClass(nvmlErrorCode):
    if nvmlErrorCode not in NVMLError._valClassMapping:
        raise ValueError('nvmlErrorCode %s is not valid' % nvmlErrorCode)
    return NVMLError._valClassMapping[nvmlErrorCode]

def _extractNVMLErrorsAsClasses():
    '''
    Generates a hierarchy of classes on top of NVMLError class.

    Each MXSMLEX Error gets a new NVMLError subclass. This way try,except blocks can filter appropriate
    exceptions more easily.

    NVMLError is a parent class. Each MXSMLEX_ERROR_* gets it's own subclass.
    e.g. MXSMLEX_ERROR_ALREADY_INITIALIZED will be turned into NVMLError_AlreadyInitialized
    '''
    this_module = sys.modules[__name__]
    nvmlErrorsNames = [x for x in dir(this_module) if x.startswith("MXSMLEX_ERROR_")]
    for err_name in nvmlErrorsNames:
        # e.g. Turn MXSMLEX_ERROR_ALREADY_INITIALIZED into NVMLError_AlreadyInitialized
        class_name = "NVMLError_" + string.capwords(err_name.replace("MXSMLEX_ERROR_", ""), "_").replace("_", "")
        err_val = getattr(this_module, err_name)
        def gen_new(val):
            def new(typ):
                obj = NVMLError.__new__(typ, val)
                return obj
            return new
        new_error_class = type(class_name, (NVMLError,), {'__new__': gen_new(err_val)})
        new_error_class.__module__ = __name__
        setattr(this_module, class_name, new_error_class)
        NVMLError._valClassMapping[err_val] = new_error_class
_extractNVMLErrorsAsClasses()

def _nvmlCheckReturn(ret):
    if (ret != MXSMLEX_SUCCESS):
        raise NVMLError(ret)
    return ret

## Function access ##
_nvmlGetFunctionPointer_cache = dict() # function pointers are cached to prevent unnecessary libLoadLock locking
def _nvmlGetFunctionPointer(name):
    global nvmlLib

    if name in _nvmlGetFunctionPointer_cache:
        return _nvmlGetFunctionPointer_cache[name]

    libLoadLock.acquire()
    try:
        # ensure library was loaded
        if (nvmlLib == None):
            raise NVMLError(MXSMLEX_ERROR_UNINITIALIZED)
        try:
            _nvmlGetFunctionPointer_cache[name] = getattr(nvmlLib, name)
            return _nvmlGetFunctionPointer_cache[name]
        except AttributeError:
            print("nvml error")
            #raise NVMLError(MXSMLEX_ERROR_FUNCTION_NOT_FOUND)
    finally:
        # lock is always freed
        libLoadLock.release()

## Alternative object
# Allows the object to be printed
# Allows mismatched types to be assigned
#  - like None when the Structure variant requires c_uint
class nvmlFriendlyObject(object):
    def __init__(self, dictionary):
        for x in dictionary:
            setattr(self, x, dictionary[x])
    def __str__(self):
        return self.__dict__.__str__()

def nvmlStructToFriendlyObject(struct):
    d = {}
    for x in struct._fields_:
        key = x[0]
        value = getattr(struct, key)
        # only need to convert from bytes if bytes, no need to check python version.
        d[key] = value.decode() if isinstance(value, bytes) else value
    obj = nvmlFriendlyObject(d)
    return obj

# pack the object so it can be passed to the MXSMLEX library
def nvmlFriendlyObjectToStruct(obj, model):
    for x in model._fields_:
        key = x[0]
        value = obj.__dict__[key]
        # any c_char_p in python3 needs to be bytes, default encoding works fine.
        if sys.version_info >= (3,):
            setattr(model, key, value.encode())
        else:
            setattr(model, key, value)
    return model

## Unit structures
class struct_c_nvmlUnit_t(Structure):
    pass # opaque handle
c_nvmlUnit_t = POINTER(struct_c_nvmlUnit_t)

class _PrintableStructure(Structure):
    """
    Abstract class that produces nicer __str__ output than ctypes.Structure.
    e.g. instead of:
      >>> print str(obj)
      <class_name object at 0x7fdf82fef9e0>
    this class will print
      class_name(field_name: formatted_value, field_name: formatted_value)

    _fmt_ dictionary of <str _field_ name> -> <str format>
    e.g. class that has _field_ 'hex_value', c_uint could be formatted with
      _fmt_ = {"hex_value" : "%08X"}
    to produce nicer output.
    Default fomratting string for all fields can be set with key "<default>" like:
      _fmt_ = {"<default>" : "%d MHz"} # e.g all values are numbers in MHz.
    If not set it's assumed to be just "%s"

    Exact format of returned str from this class is subject to change in the future.
    """
    _fmt_ = {}
    def __str__(self):
        result = []
        for x in self._fields_:
            key = x[0]
            value = getattr(self, key)
            fmt = "%s"
            if key in self._fmt_:
                fmt = self._fmt_[key]
            elif "<default>" in self._fmt_:
                fmt = self._fmt_["<default>"]
            result.append(("%s: " + fmt) % (key, value))
        return self.__class__.__name__ + "(" +  ", ".join(result) + ")"

    def __getattribute__(self, name):
        res = super(_PrintableStructure, self).__getattribute__(name)
        # need to convert bytes to unicode for python3 don't need to for python2
        # Python 2 strings are of both str and bytes
        # Python 3 strings are not of type bytes
        # ctypes should convert everything to the correct values otherwise
        if isinstance(res, bytes):
            if isinstance(res, str):
                return res
            return res.decode()
        return res

    def __setattr__(self, name, value):
        if isinstance(value, str):
            # encoding a python2 string returns the same value, since python2 strings are bytes already
            # bytes passed in python3 will be ignored.
            value = value.encode()
        super(_PrintableStructure, self).__setattr__(name, value)

class c_nvmlUnitInfo_t(_PrintableStructure):
    _fields_ = [
        ('name', c_char * 96),
        ('id', c_char * 96),
        ('serial', c_char * 96),
        ('firmwareVersion', c_char * 96),
    ]

class c_nvmlC2cModeInfo_v1_t(_PrintableStructure):
    _fields_ = [
        ('isC2cEnabled', c_uint)
    ]

nvmlC2cModeInfo_v1 = 0x1000008;

class c_nvmlLedState_t(_PrintableStructure):
    _fields_ = [
        ('cause', c_char * 256),
        ('color', _nvmlLedColor_t),
    ]

class c_nvmlPSUInfo_t(_PrintableStructure):
    _fields_ = [
        ('state', c_char * 256),
        ('current', c_uint),
        ('voltage', c_uint),
        ('power', c_uint),
    ]

class c_nvmlUnitFanInfo_t(_PrintableStructure):
    _fields_ = [
        ('speed', c_uint),
        ('state', _nvmlFanState_t),
    ]

class c_nvmlUnitFanSpeeds_t(_PrintableStructure):
    _fields_ = [
        ('fans', c_nvmlUnitFanInfo_t * 24),
        ('count', c_uint)
    ]

## Device structures
class struct_c_nvmlDevice_t(Structure):
    pass # opaque handle
c_nvmlDevice_t = POINTER(struct_c_nvmlDevice_t)

class nvmlPciInfoExt_v1_t(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('domain', c_uint),
        ('bus', c_uint),
        ('device', c_uint),
        ('pciDeviceId', c_uint),
        ('pciSubSystemId', c_uint),
        ('baseClass', c_uint),
        ('subClass', c_uint),
        ('busId', c_char * MXSMLEX_DEVICE_PCI_BUS_ID_BUFFER_SIZE),
    ]
    _fmt_ = {
            'version'        : "0x%04X",
            'domain'         : "0x%04X",
            'bus'            : "0x%02X",
            'device'         : "0x%02X",
            'pciDeviceId'    : "0x%08X",
            'pciSubSystemId' : "0x%08X",
            'baseClass'      : "0x%01X",
            'subClass'       : "0x%01X",
            }

nvmlPciInfoExt_v1 = 0x1000040

# Legacy pciInfo used for _v1 and _v2
class nvmlPciInfo_v2_t(_PrintableStructure):
    _fields_ = [
        ('busId', c_char * MXSMLEX_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE),
        ('domain', c_uint),
        ('bus', c_uint),
        ('device', c_uint),
        ('pciDeviceId', c_uint),

        # Added in 2.285
        ('pciSubSystemId', c_uint),
        ('reserved0', c_uint),
        ('reserved1', c_uint),
        ('reserved2', c_uint),
        ('reserved3', c_uint),
    ]
    _fmt_ = {
            'domain'         : "0x%04X",
            'bus'            : "0x%02X",
            'device'         : "0x%02X",
            'pciDeviceId'    : "0x%08X",
            'pciSubSystemId' : "0x%08X",
            }

class nvmlPciInfo_t(_PrintableStructure):
    _fields_ = [
        # Moved to the new busId location below
        ('busIdLegacy', c_char * MXSMLEX_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE),
        ('domain', c_uint),
        ('bus', c_uint),
        ('device', c_uint),
        ('pciDeviceId', c_uint),

        # Added in 2.285
        ('pciSubSystemId', c_uint),
        # New busId replaced the long deprecated and reserved fields with a
        # field of the same size in 9.0
        ('busId', c_char * MXSMLEX_DEVICE_PCI_BUS_ID_BUFFER_SIZE),
    ]
    _fmt_ = {
            'domain'         : "0x%08X",
            'bus'            : "0x%02X",
            'device'         : "0x%02X",
            'pciDeviceId'    : "0x%08X",
            'pciSubSystemId' : "0x%08X",
            }

class c_nvmlExcludedDeviceInfo_t(_PrintableStructure):
    _fields_ = [
        ('pci', nvmlPciInfo_t),
        ('uuid', c_char * MXSMLEX_DEVICE_UUID_BUFFER_SIZE)
    ]

class nvmlNvLinkUtilizationControl_t(_PrintableStructure):
    _fields_ = [
        ('units', _nvmlNvLinkUtilizationCountUnits_t),
        ('pktfilter', _nvmlNvLinkUtilizationCountPktTypes_t),
    ]

class c_nvmlMemory_t(_PrintableStructure):
    _fields_ = [
        ('total', c_ulonglong),
        ('free', c_ulonglong),
        ('used', c_ulonglong),
    ]
    _fmt_ = {'<default>': "%d B"}

class c_nvmlMemory_v2_t(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('total', c_ulonglong),
        ('reserved', c_ulonglong),
        ('free', c_ulonglong),
        ('used', c_ulonglong),
    ]
    _fmt_ = {'<default>': "%d B"}

nvmlMemory_v2 = 0x02000028

class c_nvmlBAR1Memory_t(_PrintableStructure):
    _fields_ = [
        ('bar1Total', c_ulonglong),
        ('bar1Free', c_ulonglong),
        ('bar1Used', c_ulonglong),
    ]
    _fmt_ = {'<default>': "%d B"}

class nvmlClkMonFaultInfo_t(Structure):
    _fields_ = [("clkApiDomain", c_uint),
                ("clkDomainFaultMask", c_uint)
    ]

class nvmlClkMonStatus_t(Structure):
    _fields_ = [("bGlobalStatus", c_uint),
                ("clkMonListSize", c_uint),
                ("clkMonList", nvmlClkMonFaultInfo_t)
    ]

# On Windows with the WDDM driver, usedGpuMemory is reported as None
# Code that processes this structure should check for None, I.E.
#
# if (info.usedGpuMemory == None):
#     # TODO handle the error
#     pass
# else:
#    print("Using %d MiB of memory" % (info.usedGpuMemory / 1024 / 1024))
# endif
#
# See MXSMLEX documentation for more information
class c_nvmlProcessInfo_v2_t(_PrintableStructure):
    _fields_ = [
        ('pid', c_uint),
        ('usedGpuMemory', c_ulonglong),
        ('gpuInstanceId', c_uint),
        ('computeInstanceId', c_uint),
    ]
    _fmt_ = {'usedGpuMemory': "%d B"}

c_nvmlProcessInfo_v3_t = c_nvmlProcessInfo_v2_t

c_nvmlProcessInfo_t = c_nvmlProcessInfo_v3_t

_nvmlProcessMode_t = c_uint
MXSMLEX_PROCESS_MODE_COMPUTE  = 0
MXSMLEX_PROCESS_MODE_GRAPHICS = 1
MXSMLEX_PROCESS_MODE_MPS      = 2

class c_nvmlProcessDetail_v1_t(Structure):
    _fields_ = [
        ('pid', c_uint),
        ('usedGpuMemory', c_ulonglong),
        ('gpuInstanceId', c_uint),
        ('computeInstanceId', c_uint),
        ('usedGpuCcProtectedMemory', c_ulonglong),
    ]

class c_nvmlProcessDetailList_v1_t(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('mode', _nvmlProcessMode_t),
        ('numProcArrayEntries', c_uint),
        ('procArray', POINTER(c_nvmlProcessDetail_v1_t)),
    ]
    _fmt_ = {'numProcArrayEntries': "%d B"}

c_nvmlProcessDetailList_t = c_nvmlProcessDetailList_v1_t

nvmlProcessDetailList_v1 = 0x1000018

class c_nvmlBridgeChipInfo_t(_PrintableStructure):
    _fields_ = [
        ('type', _nvmlBridgeChipType_t),
        ('fwVersion', c_uint),
    ]

class c_nvmlBridgeChipHierarchy_t(_PrintableStructure):
    _fields_ = [
        ('bridgeCount', c_uint),
        ('bridgeChipInfo', c_nvmlBridgeChipInfo_t * 128),
    ]

class c_nvmlEccErrorCounts_t(_PrintableStructure):
    _fields_ = [
        ('l1Cache', c_ulonglong),
        ('l2Cache', c_ulonglong),
        ('deviceMemory', c_ulonglong),
        ('registerFile', c_ulonglong),
    ]

class c_nvmlUtilization_t(_PrintableStructure):
    _fields_ = [
        ('gpu', c_uint),
        ('memory', c_uint),
    ]
    _fmt_ = {'<default>': "%d %%"}

# Added in 2.285
class c_nvmlHwbcEntry_t(_PrintableStructure):
    _fields_ = [
        ('hwbcId', c_uint),
        ('firmwareVersion', c_char * 32),
    ]

class c_nvmlValue_t(Union):
    _fields_ = [
        ('dVal', c_double),
        ('uiVal', c_uint),
        ('ulVal', c_ulong),
        ('ullVal', c_ulonglong),
        ('sllVal', c_longlong),
        ('siVal', c_int),
    ]

class c_nvmlSample_t(_PrintableStructure):
    _fields_ = [
        ('timeStamp', c_ulonglong),
        ('sampleValue', c_nvmlValue_t),
    ]

class c_nvmlViolationTime_t(_PrintableStructure):
    _fields_ = [
        ('referenceTime', c_ulonglong),
        ('violationTime', c_ulonglong),
    ]

class c_nvmlFieldValue_t(_PrintableStructure):
    _fields_ = [
        ('fieldId', c_uint32),
        ('scopeId', c_uint32),
        ('timestamp', c_int64),
        ('latencyUsec', c_int64),
        ('valueType', _nvmlValueType_t),
        ('nvmlReturn', _nvmlReturn_t),
        ('value', c_nvmlValue_t)
    ]

class c_nvmlVgpuHeterogeneousMode_v1_t(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('mode', c_uint),
    ]

VgpuHeterogeneousMode_v1 = 0x1000008

class c_nvmlVgpuPlacementId_v1_t(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('placementId', c_uint),
    ]

VgpuPlacementId_v1 = 0x1000008

class c_nvmlVgpuPlacementList_v1_t(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('count', c_uint),
        ('placementSize', c_uint),
        ('placementIds', POINTER(c_uint)),
    ]

VgpuPlacementList_v1 = 0x1000018

class c_nvmlVgpuInstanceUtilizationSample_t(_PrintableStructure):
    _fields_ = [
        ('vgpuInstance', _nvmlVgpuInstance_t),
        ('timeStamp', c_ulonglong),
        ('smUtil', c_nvmlValue_t),
        ('memUtil', c_nvmlValue_t),
        ('encUtil', c_nvmlValue_t),
        ('decUtil', c_nvmlValue_t),
    ]

class c_nvmlVgpuInstanceUtilizationInfo_v1_t(_PrintableStructure):
    _fields_ = [
        ('timeStamp', c_ulonglong),
        ('vgpuInstance', _nvmlVgpuInstance_t),
        ('smUtil', c_nvmlValue_t),
        ('memUtil', c_nvmlValue_t),
        ('encUtil', c_nvmlValue_t),
        ('decUtil', c_nvmlValue_t),
        ('jpgUtil', c_nvmlValue_t),
        ('ofaUtil', c_nvmlValue_t),
    ]

class c_nvmlVgpuInstancesUtilizationInfo_v1_t(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('sampleValType', _nvmlValueType_t),
        ('vgpuInstanceCount', c_uint),
        ('lastSeenTimeStamp', c_ulonglong),
        ('vgpuUtilArray', POINTER(c_nvmlVgpuInstanceUtilizationInfo_v1_t)),
    ]

VgpuInstancesUtilizationInfo_v1 = 0x01000020

class c_nvmlVgpuProcessUtilizationSample_t(_PrintableStructure):
    _fields_ = [
        ('vgpuInstance', _nvmlVgpuInstance_t),
        ('pid', c_uint),
        ('processName', c_char * MXSMLEX_VGPU_NAME_BUFFER_SIZE),
        ('timeStamp', c_ulonglong),
        ('smUtil', c_uint),
        ('memUtil', c_uint),
        ('encUtil', c_uint),
        ('decUtil', c_uint),
    ]

class c_nvmlVgpuProcessUtilizationInfo_v1_t(_PrintableStructure):
    _fields_ = [
        ('processName', c_char * MXSMLEX_VGPU_NAME_BUFFER_SIZE),
        ('timeStamp', c_ulonglong),
        ('vgpuInstance', _nvmlVgpuInstance_t),
        ('pid', c_uint),
        ('smUtil', c_uint),
        ('memUtil', c_uint),
        ('encUtil', c_uint),
        ('decUtil', c_uint),
        ('jpgUtil', c_uint),
        ('ofaUtil', c_uint),
    ]

class c_nvmlVgpuProcessesUtilizationInfo_v1_t(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('vgpuProcessCount', c_uint),
        ('lastSeenTimeStamp', c_ulonglong),
        ('vgpuProcUtilArray', POINTER(c_nvmlVgpuProcessUtilizationInfo_v1_t)),
    ]

VgpuProcessesUtilizationInfo_v1 = 0x01000018

class c_nvmlVgpuLicenseExpiry_t(_PrintableStructure):
    _fields_ = [
        ('year',    c_uint32),
        ('month',   c_uint16),
        ('day',     c_uint16),
        ('hour',    c_uint16),
        ('min',     c_uint16),
        ('sec',     c_uint16),
        ('status',  c_uint8),
    ]

MXSMLEX_GRID_LICENSE_STATE_UNKNOWN                 = 0
MXSMLEX_GRID_LICENSE_STATE_UNINITIALIZED           = 1
MXSMLEX_GRID_LICENSE_STATE_UNLICENSED_UNRESTRICTED = 2
MXSMLEX_GRID_LICENSE_STATE_UNLICENSED_RESTRICTED   = 3
MXSMLEX_GRID_LICENSE_STATE_UNLICENSED              = 4
MXSMLEX_GRID_LICENSE_STATE_LICENSED                = 5

class c_nvmlVgpuLicenseInfo_t(_PrintableStructure):
    _fields_ = [
        ('isLicensed',      c_uint8),
        ('licenseExpiry',   c_nvmlVgpuLicenseExpiry_t),
        ('currentState',    c_uint),
    ]

class c_nvmlEncoderSession_t(_PrintableStructure):
    _fields_ = [
        ('sessionId', c_uint),
        ('pid', c_uint),
        ('vgpuInstance', _nvmlVgpuInstance_t),
        ('codecType', c_uint),
        ('hResolution', c_uint),
        ('vResolution', c_uint),
        ('averageFps', c_uint),
        ('encodeLatency', c_uint),
    ]

class c_nvmlProcessUtilizationSample_t(_PrintableStructure):
    _fields_ = [
        ('pid', c_uint),
        ('timeStamp', c_ulonglong),
        ('smUtil', c_uint),
        ('memUtil', c_uint),
        ('encUtil', c_uint),
        ('decUtil', c_uint),
    ]

class c_nvmlProcessUtilizationInfo_v1_t(_PrintableStructure):
    _fields_ = [
        ('timeStamp', c_ulonglong),
        ('pid', c_uint),
        ('smUtil', c_uint),
        ('memUtil', c_uint),
        ('encUtil', c_uint),
        ('decUtil', c_uint),
        ('jpgUtil', c_uint),
        ('ofaUtil', c_uint),
    ]

class c_nvmlProcessesUtilizationInfo_v1_t(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('processSamplesCount', c_uint),
        ('lastSeenTimeStamp', c_ulonglong),
        ('procUtilArray', POINTER(c_nvmlProcessUtilizationInfo_v1_t)),
    ]

ProcessesUtilizationInfo_v1 = 0x01000018

class c_nvmlGridLicenseExpiry_t(_PrintableStructure):
    _fields_ = [
        ('year',    c_uint32),
        ('month',   c_uint16),
        ('day',     c_uint16),
        ('hour',    c_uint16),
        ('min',     c_uint16),
        ('sec',     c_uint16),
        ('status',  c_uint8),
    ]

class c_nvmlGridLicensableFeature_v4_t(_PrintableStructure):
    _fields_ = [
        ('featureCode',    _nvmlGridLicenseFeatureCode_t),
        ('featureState',   c_uint),
        ('licenseInfo',    c_char * MXSMLEX_GRID_LICENSE_BUFFER_SIZE),
        ('productName',    c_char * MXSMLEX_GRID_LICENSE_BUFFER_SIZE),
        ('featureEnabled', c_uint),
        ('licenseExpiry',  c_nvmlGridLicenseExpiry_t),
    ]

class c_nvmlGridLicensableFeatures_v4_t(_PrintableStructure):
    _fields_ = [
        ('isGridLicenseSupported',  c_int),
        ('licensableFeaturesCount', c_uint),
        ('gridLicensableFeatures',  c_nvmlGridLicensableFeature_v4_t * MXSMLEX_GRID_LICENSE_FEATURE_MAX_COUNT),
    ]

class c_nvmlGridLicensableFeature_v3_t(_PrintableStructure):
    _fields_ = [
        ('featureCode', _nvmlGridLicenseFeatureCode_t),
        ('featureState', c_uint),
        ('licenseInfo', c_char * MXSMLEX_GRID_LICENSE_BUFFER_SIZE),
        ('productName', c_char * MXSMLEX_GRID_LICENSE_BUFFER_SIZE),
        ('featureEnabled', c_uint),
    ]

class c_nvmlGridLicensableFeatures_v3_t(_PrintableStructure):
    _fields_ = [
        ('isGridLicenseSupported', c_int),
        ('licensableFeaturesCount', c_uint),
        ('gridLicensableFeatures', c_nvmlGridLicensableFeature_v3_t * MXSMLEX_GRID_LICENSE_FEATURE_MAX_COUNT),
    ]

class c_nvmlGridLicensableFeature_v2_t(_PrintableStructure):
    _fields_ = [
        ('featureCode', _nvmlGridLicenseFeatureCode_t),
        ('featureState', c_uint),
        ('licenseInfo', c_char * MXSMLEX_GRID_LICENSE_BUFFER_SIZE),
        ('productName', c_char * MXSMLEX_GRID_LICENSE_BUFFER_SIZE),
    ]

class c_nvmlGridLicensableFeatures_v2_t(_PrintableStructure):
    _fields_ = [
        ('isGridLicenseSupported', c_int),
        ('licensableFeaturesCount', c_uint),
        ('gridLicensableFeatures', c_nvmlGridLicensableFeature_v2_t * MXSMLEX_GRID_LICENSE_FEATURE_MAX_COUNT),
    ]

class c_nvmlGridLicensableFeature_t(_PrintableStructure):
    _fields_ = [
        ('featureCode', _nvmlGridLicenseFeatureCode_t),
        ('featureState', c_uint),
        ('licenseInfo', c_char * MXSMLEX_GRID_LICENSE_BUFFER_SIZE),
    ]

class c_nvmlGridLicensableFeatures_t(_PrintableStructure):
    _fields_ = [
        ('isGridLicenseSupported', c_int),
        ('licensableFeaturesCount', c_uint),
        ('gridLicensableFeatures', c_nvmlGridLicensableFeature_t * MXSMLEX_GRID_LICENSE_FEATURE_MAX_COUNT),
    ]

## Event structures
class struct_c_nvmlEventSet_t(Structure):
    pass # opaque handle
c_nvmlEventSet_t = POINTER(struct_c_nvmlEventSet_t)

nvmlEventTypeSingleBitEccError     = 0x0000000000000001
nvmlEventTypeDoubleBitEccError     = 0x0000000000000002
nvmlEventTypePState                = 0x0000000000000004
nvmlEventTypeXidCriticalError      = 0x0000000000000008
nvmlEventTypeClock                 = 0x0000000000000010
nvmlEventTypePowerSourceChange     = 0x0000000000000080
nvmlEventMigConfigChange           = 0x0000000000000100
nvmlEventTypeNone                  = 0x0000000000000000
nvmlEventTypeAll                   = (
                                        nvmlEventTypeNone
                                        | nvmlEventTypeSingleBitEccError
                                        | nvmlEventTypeDoubleBitEccError
                                        | nvmlEventTypePState
                                        | nvmlEventTypeClock
                                        | nvmlEventTypePowerSourceChange
                                        | nvmlEventTypeXidCriticalError
                                        | nvmlEventMigConfigChange
                                     )

## Clock Event Reasons defines
nvmlClocksEventReasonGpuIdle              = 0x0000000000000001
nvmlClocksEventReasonApplicationsClocksSetting = 0x0000000000000002
nvmlClocksEventReasonUserDefinedClocks         = nvmlClocksEventReasonApplicationsClocksSetting # deprecated, use nvmlClocksEventReasonApplicationsClocksSetting
nvmlClocksEventReasonSwPowerCap           = 0x0000000000000004
nvmlClocksEventReasonHwSlowdown           = 0x0000000000000008
nvmlClocksEventReasonSyncBoost            = 0x0000000000000010
nvmlClocksEventReasonSwThermalSlowdown    = 0x0000000000000020
nvmlClocksEventReasonHwThermalSlowdown    = 0x0000000000000040
nvmlClocksEventReasonHwPowerBrakeSlowdown = 0x0000000000000080
nvmlClocksEventReasonDisplayClockSetting  = 0x0000000000000100
nvmlClocksEventReasonNone                 = 0x0000000000000000
nvmlClocksEventReasonAll                  = (
                                                  nvmlClocksEventReasonNone |
                                                  nvmlClocksEventReasonGpuIdle |
                                                  nvmlClocksEventReasonApplicationsClocksSetting |
                                                  nvmlClocksEventReasonSwPowerCap |
                                                  nvmlClocksEventReasonHwSlowdown |
                                                  nvmlClocksEventReasonSyncBoost |
                                                  nvmlClocksEventReasonSwThermalSlowdown |
                                                  nvmlClocksEventReasonHwThermalSlowdown |
                                                  nvmlClocksEventReasonHwPowerBrakeSlowdown |
                                                  nvmlClocksEventReasonDisplayClockSetting
                                               )

## Following have been deprecated
nvmlClocksThrottleReasonGpuIdle              = 0x0000000000000001
nvmlClocksThrottleReasonApplicationsClocksSetting = 0x0000000000000002
nvmlClocksThrottleReasonUserDefinedClocks         = nvmlClocksThrottleReasonApplicationsClocksSetting # deprecated, use nvmlClocksThrottleReasonApplicationsClocksSetting
nvmlClocksThrottleReasonSwPowerCap           = 0x0000000000000004
nvmlClocksThrottleReasonHwSlowdown           = 0x0000000000000008
nvmlClocksThrottleReasonSyncBoost            = 0x0000000000000010
nvmlClocksThrottleReasonSwThermalSlowdown    = 0x0000000000000020
nvmlClocksThrottleReasonHwThermalSlowdown    = 0x0000000000000040
nvmlClocksThrottleReasonHwPowerBrakeSlowdown = 0x0000000000000080
nvmlClocksThrottleReasonDisplayClockSetting  = 0x0000000000000100
nvmlClocksThrottleReasonNone                 = 0x0000000000000000
nvmlClocksThrottleReasonAll                  = (
                                                  nvmlClocksThrottleReasonNone |
                                                  nvmlClocksThrottleReasonGpuIdle |
                                                  nvmlClocksThrottleReasonApplicationsClocksSetting |
                                                  nvmlClocksThrottleReasonSwPowerCap |
                                                  nvmlClocksThrottleReasonHwSlowdown |
                                                  nvmlClocksThrottleReasonSyncBoost |
                                                  nvmlClocksThrottleReasonSwThermalSlowdown |
                                                  nvmlClocksThrottleReasonHwThermalSlowdown |
                                                  nvmlClocksThrottleReasonHwPowerBrakeSlowdown |
                                                  nvmlClocksThrottleReasonDisplayClockSetting
                                               )

class c_nvmlEventData_t(_PrintableStructure):
    _fields_ = [
        ('device', c_nvmlDevice_t),
        ('eventType', c_ulonglong),
        ('eventData', c_ulonglong),
        ('gpuInstanceId', c_uint),
        ('computeInstanceId', c_uint)
    ]
    _fmt_ = {'eventType': "0x%08X"}

class c_nvmlAccountingStats_t(_PrintableStructure):
    _fields_ = [
        ('gpuUtilization', c_uint),
        ('memoryUtilization', c_uint),
        ('maxMemoryUsage', c_ulonglong),
        ('time', c_ulonglong),
        ('startTime', c_ulonglong),
        ('isRunning', c_uint),
        ('reserved', c_uint * 5)
    ]

class c_nvmlVgpuVersion_t(Structure):
    _fields_ = [("minVersion", c_uint),
                ("maxVersion", c_uint)
               ]

class c_nvmlVgpuMetadata_t(_PrintableStructure):
    _fields_ = [("version", c_uint),
                ("revision", c_uint),
                ("guestInfoState", _nvmlVgpuGuestInfoState_t),
                ("guestDriverVersion", c_char * MXSMLEX_SYSTEM_DRIVER_VERSION_BUFFER_SIZE),
                ("hostDriverVersion", c_char * MXSMLEX_SYSTEM_DRIVER_VERSION_BUFFER_SIZE),
                ("reserved", c_uint * 6),
                ("vgpuVirtualizationCaps", c_uint),
                ("guestVgpuVersion", c_uint),
                ("opaqueDataSize", c_uint),
                ("opaqueData", c_char * MXSMLEX_VGPU_METADATA_OPAQUE_DATA_SIZE)
               ]

class c_nvmlVgpuPgpuMetadata_t(_PrintableStructure):
    _fields_ = [("version", c_uint),
                ("revision", c_uint),
                ("hostDriverVersion", c_char * MXSMLEX_SYSTEM_DRIVER_VERSION_BUFFER_SIZE),
                ("pgpuVirtualizationCaps", c_uint),
                ("reserved", c_uint * 5),
                ("hostSupportedVgpuRange", c_nvmlVgpuVersion_t),
                ("opaqueDataSize", c_uint),
                ("opaqueData", c_char * MXSMLEX_VGPU_PGPU_METADATA_OPAQUE_DATA_SIZE)
               ]

class c_nvmlVgpuPgpuCompatibility_t(Structure):
    _fields_ = [("vgpuVmCompatibility", _nvmlVgpuVmCompatibility_t),
                ("compatibilityLimitCode", _nvmlVgpuPgpuCompatibilityLimitCode_t)
               ]

## vGPU scheduler policy defines
MXSMLEX_VGPU_SCHEDULER_POLICY_UNKNOWN      = 0
MXSMLEX_VGPU_SCHEDULER_POLICY_BEST_EFFORT  = 1
MXSMLEX_VGPU_SCHEDULER_POLICY_EQUAL_SHARE  = 2
MXSMLEX_VGPU_SCHEDULER_POLICY_FIXED_SHARE  = 3

## Supported vGPU scheduler policy count
MXSMLEX_SUPPORTED_VGPU_SCHEDULER_POLICY_COUNT  = 3

MXSMLEX_SCHEDULER_SW_MAX_LOG_ENTRIES           = 200

MXSMLEX_VGPU_SCHEDULER_ARR_DEFAULT   = 0
MXSMLEX_VGPU_SCHEDULER_ARR_DISABLE   = 1
MXSMLEX_VGPU_SCHEDULER_ARR_ENABLE    = 2

class c_nvmlVgpuSchedDataWithARR_t(_PrintableStructure):
    _fields_ = [
        ('avgFactor',   c_uint),
        ('timeslice',   c_uint),
    ]

class c_nvmlVgpuSchedData_t(_PrintableStructure):
    _fields_ = [
        ('timeslice',   c_uint),
    ]

class c_nvmlVgpuSchedulerParams_t(Union):
    _fields_ = [
        ('vgpuSchedDataWithARR', c_nvmlVgpuSchedDataWithARR_t),
        ('vgpuSchedData',        c_nvmlVgpuSchedData_t),
    ]

class c_nvmlVgpuSchedulerLogEntry_t(_PrintableStructure):
    _fields_ = [
        ('timestamp',                   c_ulonglong),
        ('timeRunTotal',                c_ulonglong),
        ('timeRun',                     c_ulonglong),
        ('swRunlistId',                 c_uint),
        ('targetTimeSlice',             c_ulonglong),
        ('cumulativePreemptionTime',    c_ulonglong),
    ]

class c_nvmlVgpuSchedulerLog_t(_PrintableStructure):
    _fields_ = [
        ('engineId',        c_uint),
        ('schedulerPolicy', c_uint),
        ('arrMode',         c_uint),
        ('schedulerParams', c_nvmlVgpuSchedulerParams_t),
        ('entriesCount',    c_uint),
        ('logEntries',      c_nvmlVgpuSchedulerLogEntry_t * MXSMLEX_SCHEDULER_SW_MAX_LOG_ENTRIES),
    ]

class c_nvmlVgpuSchedulerGetState_t(_PrintableStructure):
    _fields_ = [
        ('schedulerPolicy', c_uint),
        ('arrMode',         c_uint),
        ('schedulerParams', c_nvmlVgpuSchedulerParams_t),
    ]

class c_nvmlVgpuSchedSetDataWithARR_t(_PrintableStructure):
    _fields_ = [
        ('avgFactor',   c_uint),
        ('frequency',   c_uint),
    ]

class c_nvmlVgpuSchedSetData_t(_PrintableStructure):
    _fields_ = [
        ('timeslice',   c_uint),
    ]

class c_nvmlVgpuSchedulerSetParams_t(Union):
    _fields_ = [
        ('vgpuSchedDataWithARR', c_nvmlVgpuSchedSetDataWithARR_t),
        ('vgpuSchedData',        c_nvmlVgpuSchedSetData_t),
    ]

class c_nvmlVgpuSchedulerSetState_t(_PrintableStructure):
    _fields_ = [
        ('schedulerPolicy', c_uint),
        ('enableARRMode',   c_uint),
        ('schedulerParams', c_nvmlVgpuSchedulerSetParams_t),
    ]

class c_nvmlVgpuSchedulerCapabilities_t(_PrintableStructure):
    _fields_ = [
        ('supportedSchedulers', c_uint * MXSMLEX_SUPPORTED_VGPU_SCHEDULER_POLICY_COUNT),
        ('maxTimeslice',        c_uint),
        ('minTimeslice',        c_uint),
        ('isArrModeSupported',  c_uint),
        ('maxFrequencyForARR',  c_uint),
        ('minFrequencyForARR',  c_uint),
        ('maxAvgFactorForARR',  c_uint),
        ('minAvgFactorForARR',  c_uint),
    ]

class c_nvmlFBCStats_t(Structure):
    _fields_ = [("sessionsCount", c_uint),
                ("averageFPS", c_uint),
                ("averageLatency", c_uint)
               ]

class c_nvmlFBCSession_t(_PrintableStructure):
    _fields_ = [
        ('sessionId', c_uint),
        ('pid', c_uint),
        ('vgpuInstance', _nvmlVgpuInstance_t),
        ('displayOrdinal', c_uint),
        ('sessionType', c_uint),
        ('sessionFlags', c_uint),
        ('hMaxResolution', c_uint),
        ('vMaxResolution', c_uint),
        ('hResolution', c_uint),
        ('vResolution', c_uint),
        ('averageFPS', c_uint),
        ('averageLatency', c_uint),
    ]

MXSMLEX_DEVICE_MIG_DISABLE = 0x0
MXSMLEX_DEVICE_MIG_ENABLE  = 0x1

MXSMLEX_GPU_INSTANCE_PROFILE_1_SLICE      = 0x0
MXSMLEX_GPU_INSTANCE_PROFILE_2_SLICE      = 0x1
MXSMLEX_GPU_INSTANCE_PROFILE_3_SLICE      = 0x2
MXSMLEX_GPU_INSTANCE_PROFILE_4_SLICE      = 0x3
MXSMLEX_GPU_INSTANCE_PROFILE_7_SLICE      = 0x4
MXSMLEX_GPU_INSTANCE_PROFILE_8_SLICE      = 0x5
MXSMLEX_GPU_INSTANCE_PROFILE_6_SLICE      = 0x6
MXSMLEX_GPU_INSTANCE_PROFILE_1_SLICE_REV1 = 0x7
MXSMLEX_GPU_INSTANCE_PROFILE_2_SLICE_REV1 = 0x8
MXSMLEX_GPU_INSTANCE_PROFILE_1_SLICE_REV2 = 0x9
MXSMLEX_GPU_INSTANCE_PROFILE_COUNT        = 0xA

class c_nvmlGpuInstancePlacement_t(Structure):
    _fields_ = [("start", c_uint),
                ("size", c_uint)
               ]

class c_nvmlGpuInstanceProfileInfo_t(Structure):
    _fields_ = [("id", c_uint),
                ("isP2pSupported", c_uint),
                ("sliceCount", c_uint),
                ("instanceCount", c_uint),
                ("multiprocessorCount", c_uint),
                ("copyEngineCount", c_uint),
                ("decoderCount", c_uint),
                ("encoderCount", c_uint),
                ("jpegCount", c_uint),
                ("ofaCount", c_uint),
                ("memorySizeMB", c_ulonglong),
               ]

nvmlGpuInstanceProfileInfo_v2 = 0x02000098

class c_nvmlGpuInstanceProfileInfo_v2_t(_PrintableStructure):
    _fields_ = [("version", c_uint),
                ("id", c_uint),
                ("isP2pSupported", c_uint),
                ("sliceCount", c_uint),
                ("instanceCount", c_uint),
                ("multiprocessorCount", c_uint),
                ("copyEngineCount", c_uint),
                ("decoderCount", c_uint),
                ("encoderCount", c_uint),
                ("jpegCount", c_uint),
                ("ofaCount", c_uint),
                ("memorySizeMB", c_ulonglong),
                ("name", c_char * MXSMLEX_DEVICE_NAME_V2_BUFFER_SIZE)
               ]
    
    def __init__(self):
        super(c_nvmlGpuInstanceProfileInfo_v2_t, self).__init__(version=nvmlGpuInstanceProfileInfo_v2)

class c_nvmlGpuInstanceInfo_t(Structure):
    _fields_ = [("device", c_nvmlDevice_t),
                ("id", c_uint),
                ("profileId", c_uint),
                ("placement", c_nvmlGpuInstancePlacement_t)
               ]

class struct_c_nvmlGpuInstance_t(Structure):
    pass # opaque handle
c_nvmlGpuInstance_t = POINTER(struct_c_nvmlGpuInstance_t)

MXSMLEX_COMPUTE_INSTANCE_PROFILE_1_SLICE      = 0x0
MXSMLEX_COMPUTE_INSTANCE_PROFILE_2_SLICE      = 0x1
MXSMLEX_COMPUTE_INSTANCE_PROFILE_3_SLICE      = 0x2
MXSMLEX_COMPUTE_INSTANCE_PROFILE_4_SLICE      = 0x3
MXSMLEX_COMPUTE_INSTANCE_PROFILE_7_SLICE      = 0x4
MXSMLEX_COMPUTE_INSTANCE_PROFILE_8_SLICE      = 0x5
MXSMLEX_COMPUTE_INSTANCE_PROFILE_6_SLICE      = 0x6
MXSMLEX_COMPUTE_INSTANCE_PROFILE_1_SLICE_REV1 = 0x7
MXSMLEX_COMPUTE_INSTANCE_PROFILE_COUNT        = 0x8

MXSMLEX_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED = 0x0
MXSMLEX_COMPUTE_INSTANCE_ENGINE_PROFILE_COUNT = 0x1

class c_nvmlComputeInstancePlacement_t(Structure):
    _fields_ = [("start", c_uint),
                ("size", c_uint)
               ]

class c_nvmlComputeInstanceProfileInfo_t(Structure):
    _fields_ = [("id", c_uint),
                ("sliceCount", c_uint),
                ("instanceCount", c_uint),
                ("multiprocessorCount", c_uint),
                ("sharedCopyEngineCount", c_uint),
                ("sharedDecoderCount", c_uint),
                ("sharedEncoderCount", c_uint),
                ("sharedJpegCount", c_uint),
                ("sharedOfaCount", c_uint)
               ]

nvmlComputeInstanceProfileInfo_v2 = 0x02000088

class c_nvmlComputeInstanceProfileInfo_v2_t(_PrintableStructure):
    _fields_ = [("version", c_uint),
                ("id", c_uint),
                ("sliceCount", c_uint),
                ("instanceCount", c_uint),
                ("multiprocessorCount", c_uint),
                ("sharedCopyEngineCount", c_uint),
                ("sharedDecoderCount", c_uint),
                ("sharedEncoderCount", c_uint),
                ("sharedJpegCount", c_uint),
                ("sharedOfaCount", c_uint),
                ("name", c_char * MXSMLEX_DEVICE_NAME_V2_BUFFER_SIZE)
               ]

    def __init__(self):
        super(c_nvmlComputeInstanceProfileInfo_v2_t, self).__init__(version=nvmlComputeInstanceProfileInfo_v2)

class c_nvmlComputeInstanceInfo_t(Structure):
    _fields_ = [("device", c_nvmlDevice_t),
                ("gpuInstance", c_nvmlGpuInstance_t),
                ("id", c_uint),
                ("profileId", c_uint),
                ("placement", c_nvmlComputeInstancePlacement_t)
               ]

MXSMLEX_MAX_GPU_UTILIZATIONS = 8
MXSMLEX_GPU_UTILIZATION_DOMAIN_GPU    = 0
MXSMLEX_GPU_UTILIZATION_DOMAIN_FB     = 1
MXSMLEX_GPU_UTILIZATION_DOMAIN_VID    = 2
MXSMLEX_GPU_UTILIZATION_DOMAIN_BUS    = 3
class c_nvmlGpuDynamicPstatesUtilization_t(Structure):
    _fields_ = [("bIsPresent", c_uint, 1),
                ("percentage", c_uint),
                ("incThreshold", c_uint),
                ("decThreshold", c_uint)]
class c_nvmlGpuDynamicPstatesInfo_t(Structure):
    _fields_ = [("flags", c_uint),
                ("utilization", c_nvmlGpuDynamicPstatesUtilization_t * MXSMLEX_MAX_GPU_UTILIZATIONS)]

MXSMLEX_MAX_THERMAL_SENSORS_PER_GPU = 3

MXSMLEX_THERMAL_TARGET_NONE          = 0
MXSMLEX_THERMAL_TARGET_GPU           = 1
MXSMLEX_THERMAL_TARGET_MEMORY        = 2
MXSMLEX_THERMAL_TARGET_POWER_SUPPLY  = 4
MXSMLEX_THERMAL_TARGET_BOARD         = 8
MXSMLEX_THERMAL_TARGET_VCD_BOARD     = 9
MXSMLEX_THERMAL_TARGET_VCD_INLET     = 10
MXSMLEX_THERMAL_TARGET_VCD_OUTLET    = 11
MXSMLEX_THERMAL_TARGET_ALL           = 15
MXSMLEX_THERMAL_TARGET_UNKNOWN       = -1

MXSMLEX_THERMAL_CONTROLLER_NONE            = 0
MXSMLEX_THERMAL_CONTROLLER_GPU_INTERNAL    = 1
MXSMLEX_THERMAL_CONTROLLER_ADM1032         = 2
MXSMLEX_THERMAL_CONTROLLER_ADT7461         = 3
MXSMLEX_THERMAL_CONTROLLER_MAX6649         = 4
MXSMLEX_THERMAL_CONTROLLER_MAX1617         = 5
MXSMLEX_THERMAL_CONTROLLER_LM99            = 6
MXSMLEX_THERMAL_CONTROLLER_LM89            = 7
MXSMLEX_THERMAL_CONTROLLER_LM64            = 8
MXSMLEX_THERMAL_CONTROLLER_G781            = 9
MXSMLEX_THERMAL_CONTROLLER_ADT7473         = 10
MXSMLEX_THERMAL_CONTROLLER_SBMAX6649       = 11
MXSMLEX_THERMAL_CONTROLLER_VBIOSEVT        = 12
MXSMLEX_THERMAL_CONTROLLER_OS              = 13
MXSMLEX_THERMAL_CONTROLLER_NVSYSCON_CANOAS = 14
MXSMLEX_THERMAL_CONTROLLER_NVSYSCON_E551   = 15
MXSMLEX_THERMAL_CONTROLLER_MAX6649R        = 16
MXSMLEX_THERMAL_CONTROLLER_ADT7473S        = 17
MXSMLEX_THERMAL_CONTROLLER_UNKNOWN         = -1

class c_nvmlGpuThermalSensor_t(Structure):
    _fields_ = [("controller", c_int),
                ("defaultMinTemp", c_int),
                ("defaultMaxTemp", c_int),
                ("currentTemp", c_int),
                ("target", c_int)]
class c_nvmlGpuThermalSettings_t(Structure):
    _fields_ = [("count", c_uint),
                ("sensor", c_nvmlGpuThermalSensor_t * MXSMLEX_MAX_THERMAL_SENSORS_PER_GPU)]

class struct_c_nvmlComputeInstance_t(Structure):
    pass # opaque handle
c_nvmlComputeInstance_t = POINTER(struct_c_nvmlComputeInstance_t)

class c_nvmlDeviceAttributes(Structure):
    _fields_ = [("multiprocessorCount", c_uint),
                ("sharedCopyEngineCount", c_uint),
                ("sharedDecoderCount", c_uint),
                ("sharedEncoderCount", c_uint),
                ("sharedJpegCount", c_uint),
                ("sharedOfaCount", c_uint),
                ("gpuInstanceSliceCount", c_uint),
                ("computeInstanceSliceCount", c_uint),
                ("memorySizeMB", c_ulonglong),
               ]

class c_nvmlRowRemapperHistogramValues(Structure):
    _fields_ = [("max", c_uint),
                ("high", c_uint),
                ("partial", c_uint),
                ("low", c_uint),
                ("none", c_uint)
               ]

MXSMLEX_GPU_CERT_CHAIN_SIZE                = 0x1000
MXSMLEX_GPU_ATTESTATION_CERT_CHAIN_SIZE    = 0x1400
MXSMLEX_CC_GPU_CEC_NONCE_SIZE              = 0x20
MXSMLEX_CC_GPU_ATTESTATION_REPORT_SIZE     = 0x2000
MXSMLEX_CC_GPU_CEC_ATTESTATION_REPORT_SIZE = 0x1000
MXSMLEX_CC_CEC_ATTESTATION_REPORT_NOT_PRESENT = 0
MXSMLEX_CC_CEC_ATTESTATION_REPORT_PRESENT     = 1

class c_nvmlConfComputeSystemState_t(Structure):
    _fields_ = [('environment', c_uint),
                ('ccFeature', c_uint),
                ('devToolsMode', c_uint),
               ]

nvmlSystemConfComputeSettings_v1 = 0x1000014

class c_nvmlSystemConfComputeSettings_v1_t(Structure):
    _fields_ = [('version', c_uint),
                ('environment', c_uint),
                ('ccFeature', c_uint),
                ('devToolsMode', c_uint),
                ('multiGpuMode', c_uint),
               ]
    def __init__(self):
        super(c_nvmlSystemConfComputeSettings_v1_t, self).__init__(version=nvmlSystemConfComputeSettings_v1)

class c_nvmlConfComputeSystemCaps_t(Structure):
    _fields_ = [('cpuCaps', c_uint),
                ('gpusCaps', c_uint),
               ]

class c_nvmlConfComputeMemSizeInfo_t(Structure):
    _fields_ = [('protectedMemSizeKib', c_ulonglong),
                ('unprotectedMemSizeKib', c_ulonglong),
               ]

class c_nvmlConfComputeGpuCertificate_t(Structure):
    _fields_ = [('certChainSize', c_uint),
                ('attestationCertChainSize', c_uint),
                ('certChain', c_uint8 * MXSMLEX_GPU_CERT_CHAIN_SIZE),
                ('attestationCertChain', c_uint8 * MXSMLEX_GPU_ATTESTATION_CERT_CHAIN_SIZE),
               ]

class c_nvmlConfComputeGpuAttestationReport_t(Structure):
    _fields_ = [('isCecAttestationReportPresent', c_uint),
                ('attestationReportSize', c_uint),
                ('cecAttestationReportSize', c_uint),
                ('nonce', c_uint8 * MXSMLEX_CC_GPU_CEC_NONCE_SIZE),
                ('attestationReport', c_uint8 * MXSMLEX_CC_GPU_ATTESTATION_REPORT_SIZE),
                ('cecAttestationReport', c_uint8 * MXSMLEX_CC_GPU_CEC_ATTESTATION_REPORT_SIZE),
               ]

class c_nvmlConfComputeSetKeyRotationThresholdInfo_t(Structure):
    _fields_ = [('version', c_uint),
                ('maxAttackerAdvantage', c_ulong),
               ]
ConfComputeSetKeyRotationThresholdInfo_v1 = 0x1000010

class c_nvmlConfComputeGetKeyRotationThresholdInfo_t(Structure):
    _fields_ = [('version', c_uint),
                ('attackerAdvantage', c_ulong),
               ]
ConfComputeGetKeyRotationThresholdInfo_v1 = 0x1000010


## string/bytes conversion for ease of use
def convertStrBytes(func):
    '''
    In python 3, strings are unicode instead of bytes, and need to be converted for ctypes
    Args from caller: (1, 'string', <__main__.c_nvmlDevice_t at 0xFFFFFFFF>)
    Args passed to function: (1, b'string', <__main__.c_nvmlDevice_t at 0xFFFFFFFF)>
    ----
    Returned from function: b'returned string'
    Returned to caller: 'returned string'
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        # encoding a str returns bytes in python 2 and 3
        args = [arg.encode() if isinstance(arg, str) else arg for arg in args]
        res = func(*args, **kwargs)
        # In python 2, str and bytes are the same
        # In python 3, str is unicode and should be decoded.
        # Ctypes handles most conversions, this only effects c_char and char arrays.
        if isinstance(res, bytes):
            if isinstance(res, str):
                return res
            return res.decode()
        return res

    if sys.version_info >= (3,):
        return wrapper
    return func

def throwOnVersionMismatch(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NVMLError_FunctionNotFound:
            raise MXSMLEXLibraryMismatchError("Unversioned function called and the "
                                           "pyMXSMLEX version does not match the MXSMLEX lib version. "
                                           "Either use matching pyMXSMLEX and MXSMLEX lib versions or "
                                           "use a versioned function such as " + func.__name__ + "_v2")
    return wrapper

## C function wrappers ##
def nvmlInitWithFlags(flags):
    _LoadNvmlLibrary()

    #
    # Initialize the library
    #
    fn = _nvmlGetFunctionPointer("mxSmlExInit")
    ret = fn(flags)
    _nvmlCheckReturn(ret)

    # Atomically update refcount
    global _nvmlLib_refcount
    libLoadLock.acquire()
    _nvmlLib_refcount += 1
    libLoadLock.release()
    return None

def nvmlInit():
    nvmlInitWithFlags(0)
    return None

def _LoadNvmlLibrary():
    '''
    Load the library if it isn't loaded already
    '''
    global nvmlLib

    if (nvmlLib == None):
        # lock to ensure only one caller loads the library
        libLoadLock.acquire()

        try:
            # ensure the library still isn't loaded
            if (nvmlLib == None):
                try:
                    if (sys.platform[:3] == "win"):
                        # cdecl calling convention
                        try:
                            # Check for nvml.dll in System32 first for DCH drivers
                            nvmlLib = CDLL(os.path.join(os.getenv("WINDIR", "C:/Windows"), "System32/nvml.dll"))
                        except OSError as ose:
                            # If nvml.dll is not found in System32, it should be in ProgramFiles
                            # load nvml.dll from %ProgramFiles%/NVIDIA Corporation/NVSMI/nvml.dll
                            nvmlLib = CDLL(os.path.join(os.getenv("ProgramFiles", "C:/Program Files"), "NVIDIA Corporation/NVSMI/nvml.dll"))
                    else:
                        # assume linux
                        nvmlLib = CDLL("libmxsml.so")
                except OSError as ose:
                    _nvmlCheckReturn(MXSMLEX_ERROR_LIBRARY_NOT_FOUND)
                if (nvmlLib == None):
                    _nvmlCheckReturn(MXSMLEX_ERROR_LIBRARY_NOT_FOUND)
        finally:
            # lock is always freed
            libLoadLock.release()

def nvmlShutdown():
    #
    # Leave the library loaded, but shutdown the interface
    #
    fn = _nvmlGetFunctionPointer("mxSmlExShutdown")
    ret = fn()
    _nvmlCheckReturn(ret)

    # Atomically update refcount
    global _nvmlLib_refcount
    libLoadLock.acquire()
    if (0 < _nvmlLib_refcount):
        _nvmlLib_refcount -= 1
    libLoadLock.release()
    return None

# Added in 2.285
@convertStrBytes
def nvmlErrorString(result):
    fn = _nvmlGetFunctionPointer("mxSmlExErrorString")
    fn.restype = c_char_p # otherwise return is an int
    ret = fn(result)
    return ret

# Added in 2.285
@convertStrBytes
def nvmlSystemGetMXSMLEXVersion():
    c_version = create_string_buffer(MXSMLEX_SYSTEM_MXSMLEX_VERSION_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer("mxSmlExSystemGetMXSMLEXVersion")
    ret = fn(c_version, c_uint(MXSMLEX_SYSTEM_MXSMLEX_VERSION_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_version.value

def nvmlSystemGetCudaDriverVersion():
    c_cuda_version = c_int()
    fn = _nvmlGetFunctionPointer("mxSmlExSystemGetCudaDriverVersion")
    ret = fn(byref(c_cuda_version))
    _nvmlCheckReturn(ret)
    return c_cuda_version.value

def nvmlSystemGetCudaDriverVersion_v2():
    c_cuda_version = c_int()
    fn = _nvmlGetFunctionPointer("mxSmlExSystemGetCudaDriverVersion_v2")
    ret = fn(byref(c_cuda_version))
    _nvmlCheckReturn(ret)
    return c_cuda_version.value

# Added in 2.285
@convertStrBytes
def nvmlSystemGetProcessName(pid):
    c_name = create_string_buffer(1024)
    fn = _nvmlGetFunctionPointer("mxSmlExSystemGetProcessName")
    ret = fn(c_uint(pid), c_name, c_uint(1024))
    _nvmlCheckReturn(ret)
    return c_name.value

@convertStrBytes
def nvmlSystemGetDriverVersion():
    c_version = create_string_buffer(MXSMLEX_SYSTEM_DRIVER_VERSION_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer("mxSmlExSystemGetDriverVersion")
    ret = fn(c_version, c_uint(MXSMLEX_SYSTEM_DRIVER_VERSION_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_version.value

# Added in 2.285
def nvmlSystemGetHicVersion():
    c_count = c_uint(0)
    hics = None
    fn = _nvmlGetFunctionPointer("mxSmlExSystemGetHicVersion")

    # get the count
    ret = fn(byref(c_count), None)

    # this should only fail with insufficient size
    if ((ret != MXSMLEX_SUCCESS) and
        (ret != MXSMLEX_ERROR_INSUFFICIENT_SIZE)):
        raise NVMLError(ret)

    # If there are no hics
    if (c_count.value == 0):
        return []

    hic_array = c_nvmlHwbcEntry_t * c_count.value
    hics = hic_array()
    ret = fn(byref(c_count), hics)
    _nvmlCheckReturn(ret)
    return hics

## Unit get functions
def nvmlUnitGetCount():
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExUnitGetCount")
    ret = fn(byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlUnitGetHandleByIndex(index):
    c_index = c_uint(index)
    unit = c_nvmlUnit_t()
    fn = _nvmlGetFunctionPointer("mxSmlExUnitGetHandleByIndex")
    ret = fn(c_index, byref(unit))
    _nvmlCheckReturn(ret)
    return unit

def nvmlUnitGetUnitInfo(unit):
    c_info = c_nvmlUnitInfo_t()
    fn = _nvmlGetFunctionPointer("mxSmlExUnitGetUnitInfo")
    ret = fn(unit, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info

def nvmlUnitGetLedState(unit):
    c_state =  c_nvmlLedState_t()
    fn = _nvmlGetFunctionPointer("mxSmlExUnitGetLedState")
    ret = fn(unit, byref(c_state))
    _nvmlCheckReturn(ret)
    return c_state

def nvmlUnitGetPsuInfo(unit):
    c_info = c_nvmlPSUInfo_t()
    fn = _nvmlGetFunctionPointer("mxSmlExUnitGetPsuInfo")
    ret = fn(unit, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info

def nvmlUnitGetTemperature(unit, type):
    c_temp = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExUnitGetTemperature")
    ret = fn(unit, c_uint(type), byref(c_temp))
    _nvmlCheckReturn(ret)
    return c_temp.value

def nvmlUnitGetFanSpeedInfo(unit):
    c_speeds = c_nvmlUnitFanSpeeds_t()
    fn = _nvmlGetFunctionPointer("mxSmlExUnitGetFanSpeedInfo")
    ret = fn(unit, byref(c_speeds))
    _nvmlCheckReturn(ret)
    return c_speeds

# added to API
def nvmlUnitGetDeviceCount(unit):
    c_count = c_uint(0)
    # query the unit to determine device count
    fn = _nvmlGetFunctionPointer("mxSmlExUnitGetDevices")
    ret = fn(unit, byref(c_count), None)
    if (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        ret = MXSMLEX_SUCCESS
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlUnitGetDevices(unit):
    c_count = c_uint(nvmlUnitGetDeviceCount(unit))
    device_array = c_nvmlDevice_t * c_count.value
    c_devices = device_array()
    fn = _nvmlGetFunctionPointer("mxSmlExUnitGetDevices")
    ret = fn(unit, byref(c_count), c_devices)
    _nvmlCheckReturn(ret)
    return c_devices

## Device get functions
def nvmlDeviceGetCount():
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetCount")
    ret = fn(byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlDeviceGetHandleByIndex(index):
    c_index = c_uint(index)
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer("mxSmlExGetDeviceHandleByIndex")
    ret = fn(c_index, byref(device))
    _nvmlCheckReturn(ret)
    return device

@convertStrBytes
def nvmlDeviceGetHandleBySerial(serial):
    c_serial = c_char_p(serial)
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetHandleBySerial")
    ret = fn(c_serial, byref(device))
    _nvmlCheckReturn(ret)
    return device

@convertStrBytes
def nvmlDeviceGetHandleByUUID(uuid):
    c_uuid = c_char_p(uuid)
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetHandleByUUID")
    ret = fn(c_uuid, byref(device))
    _nvmlCheckReturn(ret)
    return device

@convertStrBytes
def nvmlDeviceGetHandleByPciBusId(pciBusId):
    c_busId = c_char_p(pciBusId)
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetHandleByPciBusId_v2")
    ret = fn(c_busId, byref(device))
    _nvmlCheckReturn(ret)
    return device

@convertStrBytes
def nvmlDeviceGetName(handle):
    c_name = create_string_buffer(MXSMLEX_DEVICE_NAME_V2_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetName")
    ret = fn(handle, c_name, c_uint(MXSMLEX_DEVICE_NAME_V2_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_name.value

def nvmlDeviceGetBoardId(handle):
    c_id = c_uint();
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetBoardId")
    ret = fn(handle, byref(c_id))
    _nvmlCheckReturn(ret)
    return c_id.value

def nvmlDeviceGetMultiGpuBoard(handle):
    c_multiGpu = c_uint();
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMultiGpuBoard")
    ret = fn(handle, byref(c_multiGpu))
    _nvmlCheckReturn(ret)
    return c_multiGpu.value

def nvmlDeviceGetBrand(handle):
    c_type = _nvmlBrandType_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetBrand")
    ret = fn(handle, byref(c_type))
    _nvmlCheckReturn(ret)
    return c_type.value

def nvmlDeviceGetC2cModeInfoV1(handle):
    c_info = c_nvmlC2cModeInfo_v1_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetC2cModeInfoV")
    ret = fn(handle, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info

def nvmlDeviceGetC2cModeInfoV(handle):
    return nvmlDeviceGetC2cModeInfoV1(handle)

@convertStrBytes
def nvmlDeviceGetBoardPartNumber(handle):
    c_part_number = create_string_buffer(MXSMLEX_DEVICE_PART_NUMBER_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetBoardPartNumber")
    ret = fn(handle, c_part_number, c_uint(MXSMLEX_DEVICE_PART_NUMBER_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_part_number.value

@convertStrBytes
def nvmlDeviceGetSerial(handle):
    c_serial = create_string_buffer(MXSMLEX_DEVICE_SERIAL_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetSerial")
    ret = fn(handle, c_serial, c_uint(MXSMLEX_DEVICE_SERIAL_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_serial.value

def nvmlDeviceGetModuleId(handle, moduleId):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetModuleId")
    ret = fn(handle, moduleId)
    return ret

def nvmlDeviceGetMemoryAffinity(handle, nodeSetSize, scope):
    affinity_array = c_ulonglong * nodeSetSize
    c_affinity = affinity_array()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMemoryAffinity")
    ret = fn(handle, nodeSetSize, byref(c_affinity), _nvmlAffinityScope_t(scope))
    _nvmlCheckReturn(ret)
    return c_affinity

def nvmlDeviceGetCpuAffinityWithinScope(handle, cpuSetSize, scope):
    affinity_array = c_ulonglong * cpuSetSize
    c_affinity = affinity_array()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetCpuAffinityWithinScope")
    ret = fn(handle, cpuSetSize, byref(c_affinity), _nvmlAffinityScope_t(scope))
    _nvmlCheckReturn(ret)
    return c_affinity

def nvmlDeviceGetCpuAffinity(handle, cpuSetSize):
    affinity_array = c_ulonglong * cpuSetSize
    c_affinity = affinity_array()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetCpuAffinity")
    ret = fn(handle, cpuSetSize, byref(c_affinity))
    _nvmlCheckReturn(ret)
    return c_affinity

def nvmlDeviceSetCpuAffinity(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetCpuAffinity")
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceClearCpuAffinity(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceClearCpuAffinity")
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetNumaNodeId(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetNumaNodeId")
    node = c_int()
    ret = fn(handle, byref(node))
    _nvmlCheckReturn(ret)
    return node.value

def nvmlDeviceGetMinorNumber(handle):
    c_minor_number = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMinorNumber")
    ret = fn(handle, byref(c_minor_number))
    _nvmlCheckReturn(ret)
    return c_minor_number.value

@convertStrBytes
def nvmlDeviceGetUUID(handle):
    c_uuid = create_string_buffer(MXSMLEX_DEVICE_UUID_V2_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetUUID")
    ret = fn(handle, c_uuid, c_uint(MXSMLEX_DEVICE_UUID_V2_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_uuid.value

@convertStrBytes
def nvmlDeviceGetInforomVersion(handle, infoRomObject):
    c_version = create_string_buffer(MXSMLEX_DEVICE_INFOROM_VERSION_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetInforomVersion")
    ret = fn(handle, _nvmlInforomObject_t(infoRomObject),
                 c_version, c_uint(MXSMLEX_DEVICE_INFOROM_VERSION_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_version.value

# Added in 4.304
@convertStrBytes
def nvmlDeviceGetInforomImageVersion(handle):
    c_version = create_string_buffer(MXSMLEX_DEVICE_INFOROM_VERSION_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetInforomImageVersion")
    ret = fn(handle, c_version, c_uint(MXSMLEX_DEVICE_INFOROM_VERSION_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_version.value

# Added in 4.304
def nvmlDeviceGetInforomConfigurationChecksum(handle):
    c_checksum = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetInforomConfigurationChecksum")
    ret = fn(handle, byref(c_checksum))
    _nvmlCheckReturn(ret)
    return c_checksum.value

# Added in 4.304
def nvmlDeviceValidateInforom(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceValidateInforom")
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetLastBBXFlushTime(handle):
    c_timestamp = c_ulonglong()
    c_durationUs = c_ulong()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetLastBBXFlushTime")
    ret = fn(handle, byref(c_timestamp), byref(c_durationUs))
    _nvmlCheckReturn(ret)
    return [c_timestamp.value, c_durationUs.value]

def nvmlDeviceGetDisplayMode(handle):
    c_mode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetDisplayMode")
    ret = fn(handle, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value

def nvmlDeviceGetDisplayActive(handle):
    c_mode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetDisplayActive")
    ret = fn(handle, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value


def nvmlDeviceGetPersistenceMode(handle):
    c_state = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPersistenceMode")
    ret = fn(handle, byref(c_state))
    _nvmlCheckReturn(ret)
    return c_state.value

def nvmlDeviceGetPciInfoExt(handle, c_info):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPciInfoExt")
    ret = fn(handle, c_info)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetPciInfo_v3(handle):
    c_info = nvmlPciInfo_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPciInfo_v3")
    ret = fn(handle, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info

def nvmlDeviceGetPciInfo(handle):
    return nvmlDeviceGetPciInfo_v3(handle)

def nvmlDeviceGetClockInfo(handle, type):
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetClockInfo")
    ret = fn(handle, _nvmlClockType_t(type), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value

# Added in 2.285
def nvmlDeviceGetMaxClockInfo(handle, type):
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMaxClockInfo")
    ret = fn(handle, _nvmlClockType_t(type), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value

# Added in 4.304
def nvmlDeviceGetApplicationsClock(handle, type):
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetApplicationsClock")
    ret = fn(handle, _nvmlClockType_t(type), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value

def nvmlDeviceGetMaxCustomerBoostClock(handle, type):
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMaxCustomerBoostClock")
    ret = fn(handle, _nvmlClockType_t(type), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value

def nvmlDeviceGetClock(handle, type, id):
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetClock")
    ret = fn(handle, _nvmlClockType_t(type), _nvmlClockId_t(id), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value

# Added in 5.319
def nvmlDeviceGetDefaultApplicationsClock(handle, type):
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetDefaultApplicationsClock")
    ret = fn(handle, _nvmlClockType_t(type), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value

# Added in 4.304
def nvmlDeviceGetSupportedMemoryClocks(handle):
    # first call to get the size
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetSupportedMemoryClocks")
    ret = fn(handle, byref(c_count), None)

    if (ret == MXSMLEX_SUCCESS):
        # special case, no clocks
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        clocks_array = c_uint * c_count.value
        c_clocks = clocks_array()

        # make the call again
        ret = fn(handle, byref(c_count), c_clocks)
        _nvmlCheckReturn(ret)

        procs = []
        for i in range(c_count.value):
            procs.append(c_clocks[i])

        return procs
    else:
        # error case
        raise NVMLError(ret)

# Added in 4.304
def nvmlDeviceGetSupportedGraphicsClocks(handle, memoryClockMHz):
    # first call to get the size
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetSupportedGraphicsClocks")
    ret = fn(handle, c_uint(memoryClockMHz), byref(c_count), None)

    if (ret == MXSMLEX_SUCCESS):
        # special case, no clocks
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        clocks_array = c_uint * c_count.value
        c_clocks = clocks_array()

        # make the call again
        ret = fn(handle, c_uint(memoryClockMHz), byref(c_count), c_clocks)
        _nvmlCheckReturn(ret)

        procs = []
        for i in range(c_count.value):
            procs.append(c_clocks[i])

        return procs
    else:
        # error case
        raise NVMLError(ret)

def nvmlDeviceGetFanSpeed(handle):
    c_speed = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetFanSpeed")
    ret = fn(handle, byref(c_speed))
    _nvmlCheckReturn(ret)
    return c_speed.value

def nvmlDeviceGetFanSpeed_v2(handle, fan):
    c_speed = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetFanSpeed_v2")
    ret = fn(handle, fan, byref(c_speed))
    _nvmlCheckReturn(ret)
    return c_speed.value

def nvmlDeviceGetTargetFanSpeed(handle, fan):
    c_speed = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetTargetFanSpeed")
    ret = fn(handle, fan, byref(c_speed))
    _nvmlCheckReturn(ret)
    return c_speed.value

def nvmlDeviceGetNumFans(device):
    c_numFans = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetNumFans")
    ret = fn(device, byref(c_numFans))
    _nvmlCheckReturn(ret)
    return c_numFans.value

def nvmlDeviceSetDefaultFanSpeed_v2(handle, index):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetDefaultFanSpeed_v2");
    ret = fn(handle, index)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetMinMaxFanSpeed(handle, minSpeed, maxSpeed):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMinMaxFanSpeed")
    ret = fn(handle, minSpeed, maxSpeed)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetFanControlPolicy_v2(handle, fan, fanControlPolicy):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetFanControlPolicy_v2")
    ret = fn(handle, fan, fanControlPolicy)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceSetFanControlPolicy(handle, fan, fanControlPolicy):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetFanControlPolicy")
    ret = fn(handle, fan, _nvmlFanControlPolicy_t(fanControlPolicy))
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetTemperature(handle, sensor):
    c_temp = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetTemperature")
    ret = fn(handle, _nvmlTemperatureSensors_t(sensor), byref(c_temp))
    _nvmlCheckReturn(ret)
    return c_temp.value

def nvmlDeviceGetTemperatureThreshold(handle, threshold):
    c_temp = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetTemperatureThreshold")
    ret = fn(handle, _nvmlTemperatureThresholds_t(threshold), byref(c_temp))
    _nvmlCheckReturn(ret)
    return c_temp.value

def nvmlDeviceSetTemperatureThreshold(handle, threshold, temp):
    c_temp = c_uint()
    c_temp.value = temp
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetTemperatureThreshold")
    ret = fn(handle, _nvmlTemperatureThresholds_t(threshold), byref(c_temp))
    _nvmlCheckReturn(ret)
    return None

# DEPRECATED use nvmlDeviceGetPerformanceState
def nvmlDeviceGetPowerState(handle):
    c_pstate = _nvmlPstates_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPowerState")
    ret = fn(handle, byref(c_pstate))
    _nvmlCheckReturn(ret)
    return c_pstate.value

def nvmlDeviceGetPerformanceState(handle):
    c_pstate = _nvmlPstates_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPerformanceState")
    ret = fn(handle, byref(c_pstate))
    _nvmlCheckReturn(ret)
    return c_pstate.value

def nvmlDeviceGetPowerManagementMode(handle):
    c_pcapMode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPowerManagementMode")
    ret = fn(handle, byref(c_pcapMode))
    _nvmlCheckReturn(ret)
    return c_pcapMode.value

def nvmlDeviceGetPowerManagementLimit(handle):
    c_limit = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPowerManagementLimit")
    ret = fn(handle, byref(c_limit))
    _nvmlCheckReturn(ret)
    return c_limit.value

# Added in 4.304
def nvmlDeviceGetPowerManagementLimitConstraints(handle):
    c_minLimit = c_uint()
    c_maxLimit = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPowerManagementLimitConstraints")
    ret = fn(handle, byref(c_minLimit), byref(c_maxLimit))
    _nvmlCheckReturn(ret)
    return [c_minLimit.value, c_maxLimit.value]

# Added in 4.304
def nvmlDeviceGetPowerManagementDefaultLimit(handle):
    c_limit = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPowerManagementDefaultLimit")
    ret = fn(handle, byref(c_limit))
    _nvmlCheckReturn(ret)
    return c_limit.value


# Added in 331
def nvmlDeviceGetEnforcedPowerLimit(handle):
    c_limit = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetEnforcedPowerLimit")
    ret = fn(handle, byref(c_limit))
    _nvmlCheckReturn(ret)
    return c_limit.value

def nvmlDeviceGetPowerUsage(handle):
    c_watts = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPowerUsage")
    ret = fn(handle, byref(c_watts))
    _nvmlCheckReturn(ret)
    return c_watts.value

def nvmlDeviceGetTotalEnergyConsumption(handle):
    c_millijoules = c_uint64()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetTotalEnergyConsumption")
    ret = fn(handle, byref(c_millijoules))
    _nvmlCheckReturn(ret)
    return c_millijoules.value

# Added in 4.304
def nvmlDeviceGetGpuOperationMode(handle):
    c_currState = _nvmlGpuOperationMode_t()
    c_pendingState = _nvmlGpuOperationMode_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGpuOperationMode")
    ret = fn(handle, byref(c_currState), byref(c_pendingState))
    _nvmlCheckReturn(ret)
    return [c_currState.value, c_pendingState.value]

# Added in 4.304
def nvmlDeviceGetCurrentGpuOperationMode(handle):
    return nvmlDeviceGetGpuOperationMode(handle)[0]

# Added in 4.304
def nvmlDeviceGetPendingGpuOperationMode(handle):
    return nvmlDeviceGetGpuOperationMode(handle)[1]

def nvmlDeviceGetMemoryInfo(handle, version=None):
    if not version:
        c_memory = c_nvmlMemory_t()
        fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMemoryInfo")
    else:
        c_memory = c_nvmlMemory_v2_t()
        c_memory.version = version
        fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMemoryInfo_v2")
    ret = fn(handle, byref(c_memory))
    _nvmlCheckReturn(ret)
    return c_memory

def nvmlDeviceGetBAR1MemoryInfo(handle):
    c_bar1_memory = c_nvmlBAR1Memory_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetBAR1MemoryInfo")
    ret = fn(handle, byref(c_bar1_memory))
    _nvmlCheckReturn(ret)
    return c_bar1_memory

def nvmlDeviceGetComputeMode(handle):
    c_mode = _nvmlComputeMode_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetComputeMode")
    ret = fn(handle, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value

def nvmlDeviceGetCudaComputeCapability(handle):
    c_major = c_int()
    c_minor = c_int()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetComputeCapability")
    ret = fn(handle, byref(c_major), byref(c_minor))
    _nvmlCheckReturn(ret)
    return (c_major.value, c_minor.value)

def nvmlDeviceGetEccMode(handle):
    c_currState = _nvmlEnableState_t()
    c_pendingState = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetEccMode")
    ret = fn(handle, byref(c_currState), byref(c_pendingState))
    _nvmlCheckReturn(ret)
    return [c_currState.value, c_pendingState.value]

# added to API
def nvmlDeviceGetCurrentEccMode(handle):
    return nvmlDeviceGetEccMode(handle)[0]

# added to API
def nvmlDeviceGetPendingEccMode(handle):
    return nvmlDeviceGetEccMode(handle)[1]

def nvmlDeviceGetDefaultEccMode(handle):
    c_defaultState = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetDefaultEccMode")
    ret = fn(handle, byref(c_defaultState))
    _nvmlCheckReturn(ret)
    return [c_defaultState.value]

def nvmlDeviceGetTotalEccErrors(handle, errorType, counterType):
    c_count = c_ulonglong()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetTotalEccErrors")
    ret = fn(handle, _nvmlMemoryErrorType_t(errorType),
                 _nvmlEccCounterType_t(counterType), byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

# This is deprecated, instead use nvmlDeviceGetMemoryErrorCounter
def nvmlDeviceGetDetailedEccErrors(handle, errorType, counterType):
    c_counts = c_nvmlEccErrorCounts_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetDetailedEccErrors")
    ret = fn(handle, _nvmlMemoryErrorType_t(errorType),
                 _nvmlEccCounterType_t(counterType), byref(c_counts))
    _nvmlCheckReturn(ret)
    return c_counts

# Added in 4.304
def nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, locationType):
    c_count = c_ulonglong()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMemoryErrorCounter")
    ret = fn(handle,
             _nvmlMemoryErrorType_t(errorType),
             _nvmlEccCounterType_t(counterType),
             _nvmlMemoryLocation_t(locationType),
             byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlDeviceGetUtilizationRates(handle):
    c_util = c_nvmlUtilization_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetUtilizationRates")
    ret = fn(handle, byref(c_util))
    _nvmlCheckReturn(ret)
    return c_util

def nvmlDeviceGetEncoderUtilization(handle):
    c_util = c_uint()
    c_samplingPeriod = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetEncoderUtilization")
    ret = fn(handle, byref(c_util), byref(c_samplingPeriod))
    _nvmlCheckReturn(ret)
    return [c_util.value, c_samplingPeriod.value]

def nvmlDeviceGetDecoderUtilization(handle):
    c_util = c_uint()
    c_samplingPeriod = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetDecoderUtilization")
    ret = fn(handle, byref(c_util), byref(c_samplingPeriod))
    _nvmlCheckReturn(ret)
    return [c_util.value, c_samplingPeriod.value]

def nvmlDeviceGetJpgUtilization(handle):
    c_util = c_uint()
    c_samplingPeriod = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetJpgUtilization")
    ret = fn(handle, byref(c_util), byref(c_samplingPeriod))
    _nvmlCheckReturn(ret)
    return [c_util.value, c_samplingPeriod.value]

def nvmlDeviceGetOfaUtilization(handle):
    c_util = c_uint()
    c_samplingPeriod = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetOfaUtilization")
    ret = fn(handle, byref(c_util), byref(c_samplingPeriod))
    _nvmlCheckReturn(ret)
    return [c_util.value, c_samplingPeriod.value]

def nvmlDeviceGetPcieReplayCounter(handle):
    c_replay = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPcieReplayCounter")
    ret = fn(handle, byref(c_replay))
    _nvmlCheckReturn(ret)
    return c_replay.value

def nvmlDeviceGetDriverModel(handle):
    c_currModel = _nvmlDriverModel_t()
    c_pendingModel = _nvmlDriverModel_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetDriverModel")
    ret = fn(handle, byref(c_currModel), byref(c_pendingModel))
    _nvmlCheckReturn(ret)
    return [c_currModel.value, c_pendingModel.value]

# added to API
def nvmlDeviceGetCurrentDriverModel(handle):
    return nvmlDeviceGetDriverModel(handle)[0]

# added to API
def nvmlDeviceGetPendingDriverModel(handle):
    return nvmlDeviceGetDriverModel(handle)[1]

# Added in 2.285
@convertStrBytes
def nvmlDeviceGetVbiosVersion(handle):
    c_version = create_string_buffer(MXSMLEX_DEVICE_VBIOS_VERSION_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetVbiosVersion")
    ret = fn(handle, c_version, c_uint(MXSMLEX_DEVICE_VBIOS_VERSION_BUFFER_SIZE))
    _nvmlCheckReturn(ret)
    return c_version.value

# Added in 2.285
def nvmlDeviceGetComputeRunningProcesses_v2(handle):
    # first call to get the size
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetComputeRunningProcesses_v2")
    ret = fn(handle, byref(c_count), None)
    if (ret == MXSMLEX_SUCCESS):
        # special case, no running processes
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        # oversize the array incase more processes are created
        c_count.value = c_count.value * 2 + 5
        proc_array = c_nvmlProcessInfo_v2_t * c_count.value
        c_procs = proc_array()
        # make the call again
        ret = fn(handle, byref(c_count), c_procs)
        _nvmlCheckReturn(ret)
        procs = []
        for i in range(c_count.value):
            # use an alternative struct for this object
            obj = nvmlStructToFriendlyObject(c_procs[i])
            if (obj.usedGpuMemory == MXSMLEX_VALUE_NOT_AVAILABLE_ulonglong.value):
                # special case for WDDM on Windows, see comment above
                obj.usedGpuMemory = None
            procs.append(obj)
        return procs
    else:
        # error case
        raise NVMLError(ret)

# Added in 2.285
def nvmlDeviceGetComputeRunningProcesses_v3(handle):
    # first call to get the size
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetComputeRunningProcesses_v3")
    ret = fn(handle, byref(c_count), None)

    if (ret == MXSMLEX_SUCCESS):
        # special case, no running processes
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        # oversize the array incase more processes are created
        c_count.value = c_count.value * 2 + 5
        proc_array = c_nvmlProcessInfo_v3_t * c_count.value
        c_procs = proc_array()

        # make the call again
        ret = fn(handle, byref(c_count), c_procs)
        _nvmlCheckReturn(ret)

        procs = []
        for i in range(c_count.value):
            # use an alternative struct for this object
            obj = nvmlStructToFriendlyObject(c_procs[i])
            if (obj.usedGpuMemory == MXSMLEX_VALUE_NOT_AVAILABLE_ulonglong.value):
                # special case for WDDM on Windows, see comment above
                obj.usedGpuMemory = None
            procs.append(obj)

        return procs
    else:
        # error case
        raise NVMLError(ret)

@throwOnVersionMismatch
def nvmlDeviceGetComputeRunningProcesses(handle):
    return nvmlDeviceGetComputeRunningProcesses_v3(handle)

def nvmlDeviceGetGraphicsRunningProcesses_v2(handle):
    # first call to get the size
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGraphicsRunningProcesses_v2")
    ret = fn(handle, byref(c_count), None)
    if (ret == MXSMLEX_SUCCESS):
        # special case, no running processes
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        # oversize the array incase more processes are created
        c_count.value = c_count.value * 2 + 5
        proc_array = c_nvmlProcessInfo_v2_t * c_count.value
        c_procs = proc_array()
        # make the call again
        ret = fn(handle, byref(c_count), c_procs)
        _nvmlCheckReturn(ret)
        procs = []
        for i in range(c_count.value):
            # use an alternative struct for this object
            obj = nvmlStructToFriendlyObject(c_procs[i])
            if (obj.usedGpuMemory == MXSMLEX_VALUE_NOT_AVAILABLE_ulonglong.value):
                # special case for WDDM on Windows, see comment above
                obj.usedGpuMemory = None
            procs.append(obj)
        return procs
    else:
        # error case
        raise NVMLError(ret)

def nvmlDeviceGetGraphicsRunningProcesses_v3(handle):
    # first call to get the size
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGraphicsRunningProcesses_v3")
    ret = fn(handle, byref(c_count), None)

    if (ret == MXSMLEX_SUCCESS):
        # special case, no running processes
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        # oversize the array incase more processes are created
        c_count.value = c_count.value * 2 + 5
        proc_array = c_nvmlProcessInfo_v3_t * c_count.value
        c_procs = proc_array()

        # make the call again
        ret = fn(handle, byref(c_count), c_procs)
        _nvmlCheckReturn(ret)

        procs = []
        for i in range(c_count.value):
            # use an alternative struct for this object
            obj = nvmlStructToFriendlyObject(c_procs[i])
            if (obj.usedGpuMemory == MXSMLEX_VALUE_NOT_AVAILABLE_ulonglong.value):
                # special case for WDDM on Windows, see comment above
                obj.usedGpuMemory = None
            procs.append(obj)

        return procs
    else:
        # error case
        raise NVMLError(ret)

@throwOnVersionMismatch
def nvmlDeviceGetGraphicsRunningProcesses(handle):
    return nvmlDeviceGetGraphicsRunningProcesses_v3(handle)

@throwOnVersionMismatch
def nvmlDeviceGetMPSComputeRunningProcesses(handle):
    return nvmlDeviceGetMPSComputeRunningProcesses_v3(handle)

def nvmlDeviceGetMPSComputeRunningProcesses_v2(handle):
    # first call to get the size
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMPSComputeRunningProcesses_v2")
    ret = fn(handle, byref(c_count), None)

    if (ret == MXSMLEX_SUCCESS):
        # special case, no running processes
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        # oversize the array incase more processes are created
        c_count.value = c_count.value * 2 + 5
        proc_array = c_nvmlProcessInfo_v2_t * c_count.value
        c_procs = proc_array()

        # make the call again
        ret = fn(handle, byref(c_count), c_procs)
        _nvmlCheckReturn(ret)

        procs = []
        for i in range(c_count.value):
            # use an alternative struct for this object
            obj = nvmlStructToFriendlyObject(c_procs[i])
            if (obj.usedGpuMemory == MXSMLEX_VALUE_NOT_AVAILABLE_ulonglong.value):
                # special case for WDDM on Windows, see comment above
                obj.usedGpuMemory = None
            procs.append(obj)

        return procs
    else:
        # error case
        raise NVMLError(ret)

def nvmlDeviceGetMPSComputeRunningProcesses_v3(handle):
    # first call to get the size
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMPSComputeRunningProcesses_v3")
    ret = fn(handle, byref(c_count), None)

    if (ret == MXSMLEX_SUCCESS):
        # special case, no running processes
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        # oversize the array incase more processes are created
        c_count.value = c_count.value * 2 + 5
        proc_array = c_nvmlProcessInfo_v3_t * c_count.value
        c_procs = proc_array()

        # make the call again
        ret = fn(handle, byref(c_count), c_procs)
        _nvmlCheckReturn(ret)

        procs = []
        for i in range(c_count.value):
            # use an alternative struct for this object
            obj = nvmlStructToFriendlyObject(c_procs[i])
            if (obj.usedGpuMemory == MXSMLEX_VALUE_NOT_AVAILABLE_ulonglong.value):
                # special case for WDDM on Windows, see comment above
                obj.usedGpuMemory = None
            procs.append(obj)

        return procs
    else:
        # error case
        raise NVMLError(ret)

def nvmlDeviceGetRunningProcessDetailList(handle, version, mode):
    c_processDetailList = c_nvmlProcessDetailList_t()
    c_processDetailList.version = version
    c_processDetailList.mode = mode

    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetRunningProcessDetailList")

    # first call to get the size
    ret = fn(handle, byref(c_processDetailList))
    if (ret == MXSMLEX_SUCCESS):
        # special case, no running processes
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        c_procs = c_nvmlProcessDetail_v1_t * c_processDetailList.numProcArrayEntries
        c_processDetailList.procArray = cast((c_procs)(), POINTER(c_nvmlProcessDetail_v1_t))

        # make the call again
        ret = fn(handle, byref(c_processDetailList))
        _nvmlCheckReturn(ret)

        procs = []
        for i in range(c_processDetailList.numProcArrayEntries):
            # use an alternative struct for this object
            obj = c_processDetailList.procArray[i]
            if (obj.usedGpuMemory == MXSMLEX_VALUE_NOT_AVAILABLE_ulonglong.value):
                obj.usedGpuMemory = None
            if (obj.usedGpuCcProtectedMemory == MXSMLEX_VALUE_NOT_AVAILABLE_ulonglong.value):
                obj.usedGpuCcProtectedMemory = None
            procs.append(obj)

        return procs
    else:
        # error case
        raise NVMLError(ret)

def nvmlDeviceGetAutoBoostedClocksEnabled(handle):
    c_isEnabled = _nvmlEnableState_t()
    c_defaultIsEnabled = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetAutoBoostedClocksEnabled")
    ret = fn(handle, byref(c_isEnabled), byref(c_defaultIsEnabled))
    _nvmlCheckReturn(ret)
    return [c_isEnabled.value, c_defaultIsEnabled.value]
    #Throws MXSMLEX_ERROR_NOT_SUPPORTED if hardware doesn't support setting auto boosted clocks

## Set functions
def nvmlUnitSetLedState(unit, color):
    fn = _nvmlGetFunctionPointer("mxSmlExUnitSetLedState")
    ret = fn(unit, _nvmlLedColor_t(color))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetPersistenceMode(handle, mode):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetPersistenceMode")
    ret = fn(handle, _nvmlEnableState_t(mode))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetComputeMode(handle, mode):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetComputeMode")
    ret = fn(handle, _nvmlComputeMode_t(mode))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetEccMode(handle, mode):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetEccMode")
    ret = fn(handle, _nvmlEnableState_t(mode))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceClearEccErrorCounts(handle, counterType):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceClearEccErrorCounts")
    ret = fn(handle, _nvmlEccCounterType_t(counterType))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetDriverModel(handle, model):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetDriverModel")
    ret = fn(handle, _nvmlDriverModel_t(model))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetAutoBoostedClocksEnabled(handle, enabled):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetAutoBoostedClocksEnabled")
    ret = fn(handle, _nvmlEnableState_t(enabled))
    _nvmlCheckReturn(ret)
    return None
    #Throws MXSMLEX_ERROR_NOT_SUPPORTED if hardware doesn't support setting auto boosted clocks

def nvmlDeviceSetDefaultAutoBoostedClocksEnabled(handle, enabled, flags):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetDefaultAutoBoostedClocksEnabled")
    ret = fn(handle, _nvmlEnableState_t(enabled), c_uint(flags))
    _nvmlCheckReturn(ret)
    return None
    #Throws MXSMLEX_ERROR_NOT_SUPPORTED if hardware doesn't support setting auto boosted clocks

def nvmlDeviceSetGpuLockedClocks(handle, minGpuClockMHz, maxGpuClockMHz):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetGpuLockedClocks")
    ret = fn(handle, c_uint(minGpuClockMHz), c_uint(maxGpuClockMHz))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceResetGpuLockedClocks(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceResetGpuLockedClocks")
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetMemoryLockedClocks(handle, minMemClockMHz, maxMemClockMHz):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetMemoryLockedClocks")
    ret = fn(handle, c_uint(minMemClockMHz), c_uint(maxMemClockMHz))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceResetMemoryLockedClocks(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceResetMemoryLockedClocks")
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetClkMonStatus(handle, c_clkMonInfo):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetClkMonStatus")
    ret = fn(handle, c_clkMonInfo)
    return ret

# Added in 4.304
def nvmlDeviceSetApplicationsClocks(handle, maxMemClockMHz, maxGraphicsClockMHz):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetApplicationsClocks")
    ret = fn(handle, c_uint(maxMemClockMHz), c_uint(maxGraphicsClockMHz))
    _nvmlCheckReturn(ret)
    return None

# Added in 4.304
def nvmlDeviceResetApplicationsClocks(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceResetApplicationsClocks")
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None

# Added in 4.304
def nvmlDeviceSetPowerManagementLimit(handle, limit):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetPowerManagementLimit")
    ret = fn(handle, c_uint(limit))
    _nvmlCheckReturn(ret)
    return None

# Added in 4.304
def nvmlDeviceSetGpuOperationMode(handle, mode):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetGpuOperationMode")
    ret = fn(handle, _nvmlGpuOperationMode_t(mode))
    _nvmlCheckReturn(ret)
    return None

# Added in 2.285
def nvmlEventSetCreate():
    fn = _nvmlGetFunctionPointer("mxSmlExEventSetCreate")
    eventSet = c_nvmlEventSet_t()
    ret = fn(byref(eventSet))
    _nvmlCheckReturn(ret)
    return eventSet

# Added in 2.285
def nvmlDeviceRegisterEvents(handle, eventTypes, eventSet):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceRegisterEvents")
    ret = fn(handle, c_ulonglong(eventTypes), eventSet)
    _nvmlCheckReturn(ret)
    return None

# Added in 2.285
def nvmlDeviceGetSupportedEventTypes(handle):
    c_eventTypes = c_ulonglong()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetSupportedEventTypes")
    ret = fn(handle, byref(c_eventTypes))
    _nvmlCheckReturn(ret)
    return c_eventTypes.value

# raises MXSMLEX_ERROR_TIMEOUT exception on timeout
def nvmlEventSetWait_v2(eventSet, timeoutms):
    fn = _nvmlGetFunctionPointer("mxSmlExEventSetWait_v2")
    data = c_nvmlEventData_t()
    ret = fn(eventSet, byref(data), c_uint(timeoutms))
    _nvmlCheckReturn(ret)
    return data

def nvmlEventSetWait(eventSet, timeoutms):
    return nvmlEventSetWait_v2(eventSet, timeoutms)

# Added in 2.285
def nvmlEventSetFree(eventSet):
    fn = _nvmlGetFunctionPointer("mxSmlExEventSetFree")
    ret = fn(eventSet)
    _nvmlCheckReturn(ret)
    return None

# Added in 3.295
def nvmlDeviceOnSameBoard(handle1, handle2):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceOnSameBoard")
    onSameBoard = c_int()
    ret = fn(handle1, handle2, byref(onSameBoard))
    _nvmlCheckReturn(ret)
    return (onSameBoard.value != 0)

# Added in 3.295
def nvmlDeviceGetCurrPcieLinkGeneration(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetCurrPcieLinkGeneration")
    gen = c_uint()
    ret = fn(handle, byref(gen))
    _nvmlCheckReturn(ret)
    return gen.value

# Added in 3.295
def nvmlDeviceGetMaxPcieLinkGeneration(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMaxPcieLinkGeneration")
    gen = c_uint()
    ret = fn(handle, byref(gen))
    _nvmlCheckReturn(ret)
    return gen.value

# Added in 3.295
def nvmlDeviceGetCurrPcieLinkWidth(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetCurrPcieLinkWidth")
    width = c_uint()
    ret = fn(handle, byref(width))
    _nvmlCheckReturn(ret)
    return width.value

# Added in 3.295
def nvmlDeviceGetMaxPcieLinkWidth(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMaxPcieLinkWidth")
    width = c_uint()
    ret = fn(handle, byref(width))
    _nvmlCheckReturn(ret)
    return width.value

def nvmlDeviceGetGpuMaxPcieLinkGeneration(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGpuMaxPcieLinkGeneration")
    gen = c_uint()
    ret = fn(handle, byref(gen))
    _nvmlCheckReturn(ret)
    return gen.value

# Added in 4.304
def nvmlDeviceGetSupportedClocksThrottleReasons(handle):
    c_reasons= c_ulonglong()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetSupportedClocksThrottleReasons")
    ret = fn(handle, byref(c_reasons))
    _nvmlCheckReturn(ret)
    return c_reasons.value

def nvmlDeviceGetSupportedClocksEventReasons(handle):
    c_reasons= c_ulonglong()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetSupportedClocksEventReasons")
    ret = fn(handle, byref(c_reasons))
    _nvmlCheckReturn(ret)
    return c_reasons.value

# Added in 4.304
def nvmlDeviceGetCurrentClocksThrottleReasons(handle):
    c_reasons= c_ulonglong()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetCurrentClocksThrottleReasons")
    ret = fn(handle, byref(c_reasons))
    _nvmlCheckReturn(ret)
    return c_reasons.value

def nvmlDeviceGetCurrentClocksEventReasons(handle):
    c_reasons= c_ulonglong()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetCurrentClocksEventReasons")
    ret = fn(handle, byref(c_reasons))
    _nvmlCheckReturn(ret)
    return c_reasons.value

# Added in 5.319
def nvmlDeviceGetIndex(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetIndex")
    c_index = c_uint()
    ret = fn(handle, byref(c_index))
    _nvmlCheckReturn(ret)
    return c_index.value

# Added in 5.319
def nvmlDeviceGetAccountingMode(handle):
    c_mode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetAccountingMode")
    ret = fn(handle, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value

def nvmlDeviceSetAccountingMode(handle, mode):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetAccountingMode")
    ret = fn(handle, _nvmlEnableState_t(mode))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceClearAccountingPids(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceClearAccountingPids")
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetAccountingStats(handle, pid):
    stats = c_nvmlAccountingStats_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetAccountingStats")
    ret = fn(handle, c_uint(pid), byref(stats))
    _nvmlCheckReturn(ret)
    if (stats.maxMemoryUsage == MXSMLEX_VALUE_NOT_AVAILABLE_ulonglong.value):
        # special case for WDDM on Windows, see comment above
        stats.maxMemoryUsage = None
    return stats

def nvmlDeviceGetAccountingPids(handle):
    count = c_uint(nvmlDeviceGetAccountingBufferSize(handle))
    pids = (c_uint * count.value)()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetAccountingPids")
    ret = fn(handle, byref(count), pids)
    _nvmlCheckReturn(ret)
    return list(map(int, pids[0:count.value]))

def nvmlDeviceGetAccountingBufferSize(handle):
    bufferSize = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetAccountingBufferSize")
    ret = fn(handle, byref(bufferSize))
    _nvmlCheckReturn(ret)
    return int(bufferSize.value)

def nvmlDeviceGetRetiredPages(device, sourceFilter):
    c_source = _nvmlPageRetirementCause_t(sourceFilter)
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetRetiredPages")

    # First call will get the size
    ret = fn(device, c_source, byref(c_count), None)

    # this should only fail with insufficient size
    if ((ret != MXSMLEX_SUCCESS) and
        (ret != MXSMLEX_ERROR_INSUFFICIENT_SIZE)):
        raise NVMLError(ret)

    # call again with a buffer
    # oversize the array for the rare cases where additional pages
    # are retired between MXSMLEX calls
    c_count.value = c_count.value * 2 + 5
    page_array = c_ulonglong * c_count.value
    c_pages = page_array()
    ret = fn(device, c_source, byref(c_count), c_pages)
    _nvmlCheckReturn(ret)
    return list(map(int, c_pages[0:c_count.value]))

def nvmlDeviceGetRetiredPages_v2(device, sourceFilter):
    c_source = _nvmlPageRetirementCause_t(sourceFilter)
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetRetiredPages_v2")

    # First call will get the size
    ret = fn(device, c_source, byref(c_count), None)

    # this should only fail with insufficient size
    if ((ret != MXSMLEX_SUCCESS) and
        (ret != MXSMLEX_ERROR_INSUFFICIENT_SIZE)):
        raise NVMLError(ret)

    # call again with a buffer
    # oversize the array for the rare cases where additional pages
    # are retired between MXSMLEX calls
    c_count.value = c_count.value * 2 + 5
    page_array = c_ulonglong * c_count.value
    c_pages = page_array()
    times_array = c_ulonglong * c_count.value
    c_times = times_array()
    ret = fn(device, c_source, byref(c_count), c_pages, c_times)
    _nvmlCheckReturn(ret)
    return [ { 'address': int(c_pages[i]), 'timestamp': int(c_times[i]) } for i in range(c_count.value) ];

def nvmlDeviceGetRetiredPagesPendingStatus(device):
    c_pending = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetRetiredPagesPendingStatus")
    ret = fn(device, byref(c_pending))
    _nvmlCheckReturn(ret)
    return int(c_pending.value)

def nvmlDeviceGetAPIRestriction(device, apiType):
    c_permission = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetAPIRestriction")
    ret = fn(device, _nvmlRestrictedAPI_t(apiType), byref(c_permission))
    _nvmlCheckReturn(ret)
    return int(c_permission.value)

def nvmlDeviceSetAPIRestriction(handle, apiType, isRestricted):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetAPIRestriction")
    ret = fn(handle, _nvmlRestrictedAPI_t(apiType), _nvmlEnableState_t(isRestricted))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetBridgeChipInfo(handle):
    bridgeHierarchy = c_nvmlBridgeChipHierarchy_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetBridgeChipInfo")
    ret = fn(handle, byref(bridgeHierarchy))
    _nvmlCheckReturn(ret)
    return bridgeHierarchy

def nvmlDeviceGetSamples(device, sampling_type, timeStamp):
    c_sampling_type = _nvmlSamplingType_t(sampling_type)
    c_time_stamp = c_ulonglong(timeStamp)
    c_sample_count = c_uint(0)
    c_sample_value_type = _nvmlValueType_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetSamples")

    ## First Call gets the size
    ret = fn(device, c_sampling_type, c_time_stamp, byref(c_sample_value_type), byref(c_sample_count), None)

    # Stop if this fails
    if (ret != MXSMLEX_SUCCESS):
        raise NVMLError(ret)

    sampleArray = c_sample_count.value * c_nvmlSample_t
    c_samples = sampleArray()
    ret = fn(device, c_sampling_type, c_time_stamp,  byref(c_sample_value_type), byref(c_sample_count), c_samples)
    _nvmlCheckReturn(ret)
    return (c_sample_value_type.value, c_samples[0:c_sample_count.value])

def nvmlDeviceGetViolationStatus(device, perfPolicyType):
    c_perfPolicy_type = _nvmlPerfPolicyType_t(perfPolicyType)
    c_violTime = c_nvmlViolationTime_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetViolationStatus")

    ## Invoke the method to get violation time
    ret = fn(device, c_perfPolicy_type, byref(c_violTime))
    _nvmlCheckReturn(ret)
    return c_violTime

def nvmlDeviceGetPcieThroughput(device, counter):
    c_util = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPcieThroughput")
    ret = fn(device, _nvmlPcieUtilCounter_t(counter), byref(c_util))
    _nvmlCheckReturn(ret)
    return c_util.value

def nvmlSystemGetTopologyGpuSet(cpuNumber):
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer("mxSmlExSystemGetTopologyGpuSet")

    # First call will get the size
    ret = fn(cpuNumber, byref(c_count), None)

    if ret != MXSMLEX_SUCCESS:
        raise NVMLError(ret)
    # call again with a buffer
    device_array = c_nvmlDevice_t * c_count.value
    c_devices = device_array()
    ret = fn(cpuNumber, byref(c_count), c_devices)
    _nvmlCheckReturn(ret)
    return list(c_devices[0:c_count.value])

def nvmlDeviceGetTopologyNearestGpus(device, level):
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetTopologyNearestGpus")

    # First call will get the size
    ret = fn(device, level, byref(c_count), None)

    if ret != MXSMLEX_SUCCESS:
        raise NVMLError(ret)

    # call again with a buffer
    device_array = c_nvmlDevice_t * c_count.value
    c_devices = device_array()
    ret = fn(device, level, byref(c_count), c_devices)
    _nvmlCheckReturn(ret)
    return list(c_devices[0:c_count.value])

def nvmlDeviceGetTopologyCommonAncestor(device1, device2):
    c_level = _nvmlGpuTopologyLevel_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetTopologyCommonAncestor")
    ret = fn(device1, device2, byref(c_level))
    _nvmlCheckReturn(ret)
    return c_level.value

def nvmlDeviceGetNvLinkUtilizationCounter(device, link, counter):
    c_rxcounter = c_ulonglong()
    c_txcounter = c_ulonglong()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetNvLinkUtilizationCounter")
    ret = fn(device, link, counter, byref(c_rxcounter), byref(c_txcounter))
    _nvmlCheckReturn(ret)
    return (c_rxcounter.value, c_txcounter.value)

def nvmlDeviceFreezeNvLinkUtilizationCounter(device, link, counter, freeze):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceFreezeNvLinkUtilizationCounter")
    ret = fn(device, link, counter, freeze)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceResetNvLinkUtilizationCounter(device, link, counter):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceResetNvLinkUtilizationCounter")
    ret = fn(device, link, counter)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceSetNvLinkUtilizationControl(device, link, counter, control, reset):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetNvLinkUtilizationControl")
    ret = fn(device, link, counter, byref(control), reset)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetNvLinkUtilizationControl(device, link, counter):
    c_control = nvmlNvLinkUtilizationControl_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetNvLinkUtilizationControl")
    ret = fn(device, link, counter, byref(c_control))
    _nvmlCheckReturn(ret)
    return c_control

def nvmlDeviceGetNvLinkCapability(device, link, capability):
    c_capResult = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetNvLinkCapability")
    ret = fn(device, link, capability, byref(c_capResult))
    _nvmlCheckReturn(ret)
    return c_capResult.value

def nvmlDeviceGetNvLinkErrorCounter(device, link, counter):
    c_result = c_ulonglong()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetNvLinkErrorCounter")
    ret = fn(device, link, counter, byref(c_result))
    _nvmlCheckReturn(ret)
    return c_result.value

def nvmlDeviceResetNvLinkErrorCounters(device, link):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceResetNvLinkErrorCounters")
    ret = fn(device, link)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetNvLinkRemotePciInfo(device, link):
    c_pci = nvmlPciInfo_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetNvLinkRemotePciInfo_v2")
    ret = fn(device, link, byref(c_pci))
    _nvmlCheckReturn(ret)
    return c_pci

def nvmlDeviceGetNvLinkRemoteDeviceType(handle, link):
    c_type = _nvmlNvLinkDeviceType_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetNvLinkRemoteDeviceType")
    ret = fn(handle, link, byref(c_type))
    _nvmlCheckReturn(ret)
    return c_type.value

def nvmlDeviceGetNvLinkState(device, link):
    c_isActive = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetNvLinkState")
    ret = fn(device, link, byref(c_isActive))
    _nvmlCheckReturn(ret)
    return c_isActive.value

def nvmlDeviceGetNvLinkVersion(device, link):
    c_version = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetNvLinkVersion")
    ret = fn(device, link, byref(c_version))
    _nvmlCheckReturn(ret)
    return c_version.value

def nvmlDeviceModifyDrainState(pciInfo, newState):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceModifyDrainState")
    ret = fn(pointer(pciInfo), newState)
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceQueryDrainState(pciInfo):
    c_newState = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceQueryDrainState")
    ret = fn(pointer(pciInfo), byref(c_newState))
    _nvmlCheckReturn(ret)
    return c_newState.value

def nvmlDeviceRemoveGpu(pciInfo):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceRemoveGpu")
    ret = fn(pointer(pciInfo))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceDiscoverGpus(pciInfo):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceDiscoverGpus")
    ret = fn(pointer(pciInfo))
    _nvmlCheckReturn(ret)
    return None

def nvmlDeviceGetFieldValues(handle, fieldIds):
    values_arr = c_nvmlFieldValue_t * len(fieldIds)
    values = values_arr()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetFieldValues")

    for i, fieldId in enumerate(fieldIds):
        try:
            (values[i].fieldId, values[i].scopeId) = fieldId
        except TypeError:
            values[i].fieldId = fieldId

    ret = fn(handle, c_int32(len(fieldIds)), byref(values))
    _nvmlCheckReturn(ret)
    return values

def nvmlDeviceClearFieldValues(handle, fieldIds):
    values_arr = c_nvmlFieldValue_t * len(fieldIds)
    values = values_arr()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceClearFieldValues")

    for i, fieldId in enumerate(fieldIds):
        try:
            (values[i].fieldId, values[i].scopeId) = fieldId
        except TypeError:
            values[i].fieldId = fieldId

    ret = fn(handle, c_int32(len(fieldIds)), byref(values))
    _nvmlCheckReturn(ret)
    return values

def nvmlDeviceGetVirtualizationMode(handle):
    c_virtualization_mode = c_ulonglong()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetVirtualizationMode")
    ret = fn(handle, byref(c_virtualization_mode))
    _nvmlCheckReturn(ret)
    return c_virtualization_mode.value

def nvmlDeviceSetVirtualizationMode(handle, virtualization_mode):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetVirtualizationMode")
    return fn(handle, virtualization_mode)

def nvmlDeviceGetVgpuHeterogeneousMode(handle):
    c_vgpuHeterogeneousMode = c_nvmlVgpuHeterogeneousMode_v1_t(0)
    c_vgpuHeterogeneousMode.version = VgpuHeterogeneousMode_v1
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetVgpuHeterogeneousMode")
    ret = fn(handle, byref(c_vgpuHeterogeneousMode))
    _nvmlCheckReturn(ret)
    return c_vgpuHeterogeneousMode.mode

def nvmlDeviceSetVgpuHeterogeneousMode(handle, heterogeneous_mode):
    c_vgpuHeterogeneousMode = c_nvmlVgpuHeterogeneousMode_v1_t(0)
    c_vgpuHeterogeneousMode.version = VgpuHeterogeneousMode_v1
    c_vgpuHeterogeneousMode.mode = heterogeneous_mode
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetVgpuHeterogeneousMode")
    ret = fn(handle, byref(c_vgpuHeterogeneousMode))
    _nvmlCheckReturn(ret)
    return ret

def nvmlVgpuInstanceGetPlacementId(vgpuInstance):
    c_placement = c_nvmlVgpuPlacementId_v1_t(0)
    c_placement.version = VgpuPlacementId_v1
    fn = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetPlacementId")
    ret = fn(vgpuInstance, byref(c_placement))
    _nvmlCheckReturn(ret)
    return c_placement.placementId

def nvmlDeviceGetVgpuTypeSupportedPlacements(handle, vgpuTypeId):
    c_vgpu_placements = c_nvmlVgpuPlacementList_v1_t()
    c_max_instances = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetMaxInstances")
    ret = fn(handle, vgpuTypeId, byref(c_max_instances))
    _nvmlCheckReturn(ret)
    c_placements = c_uint * c_max_instances.value
    c_vgpu_placements.version = VgpuPlacementList_v1
    c_vgpu_placements.placementIds = c_placements()
    fn  = _nvmlGetFunctionPointer("mxSmlExDeviceGetVgpuTypeSupportedPlacements")
    ret = fn(handle, vgpuTypeId, byref(c_vgpu_placements))
    _nvmlCheckReturn(ret)
    return c_vgpu_placements

def nvmlDeviceGetVgpuTypeCreatablePlacements(handle, vgpuTypeId):
    c_vgpu_placements = c_nvmlVgpuPlacementList_v1_t()
    c_max_instances = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetMaxInstances")
    ret = fn(handle, vgpuTypeId, byref(c_max_instances))
    _nvmlCheckReturn(ret)
    c_placements = c_uint * c_max_instances.value
    c_vgpu_placements.version = VgpuPlacementList_v1
    c_vgpu_placements.placementIds = c_placements()
    fn  = _nvmlGetFunctionPointer("mxSmlExDeviceGetVgpuTypeCreatablePlacements")
    ret = fn(handle, vgpuTypeId, byref(c_vgpu_placements))
    _nvmlCheckReturn(ret)
    return c_vgpu_placements

def nvmlGetVgpuDriverCapabilities(capability):
    c_capResult = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExGetVgpuDriverCapabilities")
    ret = fn(_nvmlVgpuDriverCapability_t(capability), byref(c_capResult))
    _nvmlCheckReturn(ret)
    return c_capResult.value

def nvmlDeviceGetVgpuCapabilities(handle, capability):
    c_capResult = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetVgpuCapabilities")
    ret = fn(handle, _nvmlDeviceVgpuCapability_t(capability), byref(c_capResult))
    _nvmlCheckReturn(ret)
    return c_capResult.value

def nvmlDeviceSetVgpuCapabilities(handle, capability, state):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetVgpuCapabilities")
    ret = fn(handle, _nvmlDeviceVgpuCapability_t(capability), state)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetSupportedVgpus(handle):
    # first call to get the size
    c_vgpu_count = c_uint(0)

    fn =  _nvmlGetFunctionPointer("mxSmlExDeviceGetSupportedVgpus")
    ret = fn(handle, byref(c_vgpu_count), None)

    if (ret == MXSMLEX_SUCCESS):
        # special case, no supported vGPUs
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        vgpu_type_ids_array = _nvmlVgpuTypeId_t * c_vgpu_count.value
        c_vgpu_type_ids = vgpu_type_ids_array()

        # make the call again
        ret = fn(handle, byref(c_vgpu_count), c_vgpu_type_ids)
        _nvmlCheckReturn(ret)
        vgpus = []
        for i in range(c_vgpu_count.value):
            vgpus.append(c_vgpu_type_ids[i])
        return vgpus
    else:
        # error case
        raise NVMLError(ret)

def nvmlDeviceGetCreatableVgpus(handle):
    # first call to get the size
    c_vgpu_count = c_uint(0)

    fn =  _nvmlGetFunctionPointer("mxSmlExDeviceGetCreatableVgpus")
    ret = fn(handle, byref(c_vgpu_count), None)

    if (ret == MXSMLEX_SUCCESS):
        # special case, no supported vGPUs
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        vgpu_type_ids_array = _nvmlVgpuTypeId_t * c_vgpu_count.value
        c_vgpu_type_ids = vgpu_type_ids_array()

        # make the call again
        ret = fn(handle, byref(c_vgpu_count), c_vgpu_type_ids)
        _nvmlCheckReturn(ret)
        vgpus = []
        for i in range(c_vgpu_count.value):
            vgpus.append(c_vgpu_type_ids[i])
        return vgpus
    else:
        # error case
        raise NVMLError(ret)

def nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId):
    c_profile_id = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetGpuInstanceProfileId")
    ret = fn(vgpuTypeId, byref(c_profile_id))
    _nvmlCheckReturn(ret)
    return (c_profile_id.value)

@convertStrBytes
def nvmlVgpuTypeGetClass(vgpuTypeId):
    c_class = create_string_buffer(MXSMLEX_DEVICE_NAME_BUFFER_SIZE)
    c_buffer_size = c_uint(MXSMLEX_DEVICE_NAME_BUFFER_SIZE)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetClass")
    ret = fn(vgpuTypeId, c_class, byref(c_buffer_size))
    _nvmlCheckReturn(ret)
    return c_class.value

@convertStrBytes
def nvmlVgpuTypeGetName(vgpuTypeId):
    c_name = create_string_buffer(MXSMLEX_DEVICE_NAME_BUFFER_SIZE)
    c_buffer_size = c_uint(MXSMLEX_DEVICE_NAME_BUFFER_SIZE)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetName")
    ret = fn(vgpuTypeId, c_name, byref(c_buffer_size))
    _nvmlCheckReturn(ret)
    return c_name.value

def nvmlVgpuTypeGetDeviceID(vgpuTypeId):
    c_device_id    = c_ulonglong(0)
    c_subsystem_id = c_ulonglong(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetDeviceID")
    ret = fn(vgpuTypeId, byref(c_device_id), byref(c_subsystem_id))
    _nvmlCheckReturn(ret)
    return (c_device_id.value, c_subsystem_id.value)

def nvmlVgpuTypeGetFramebufferSize(vgpuTypeId):
    c_fb_size = c_ulonglong(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetFramebufferSize")
    ret = fn(vgpuTypeId, byref(c_fb_size))
    _nvmlCheckReturn(ret)
    return c_fb_size.value

def nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId):
    c_num_heads = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetNumDisplayHeads")
    ret = fn(vgpuTypeId, byref(c_num_heads))
    _nvmlCheckReturn(ret)
    return c_num_heads.value

def nvmlVgpuTypeGetResolution(vgpuTypeId):
    c_xdim = c_uint(0)
    c_ydim = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetResolution")
    ret = fn(vgpuTypeId, 0, byref(c_xdim), byref(c_ydim))
    _nvmlCheckReturn(ret)
    return (c_xdim.value, c_ydim.value)

@convertStrBytes
def nvmlVgpuTypeGetLicense(vgpuTypeId):
    c_license = create_string_buffer(MXSMLEX_GRID_LICENSE_BUFFER_SIZE)
    c_buffer_size = c_uint(MXSMLEX_GRID_LICENSE_BUFFER_SIZE)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetLicense")
    ret = fn(vgpuTypeId, c_license, c_buffer_size)
    _nvmlCheckReturn(ret)
    return c_license.value

def nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId):
    c_frl_config = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetFrameRateLimit")
    ret = fn(vgpuTypeId, byref(c_frl_config))
    _nvmlCheckReturn(ret)
    return c_frl_config.value

def nvmlVgpuTypeGetGspHeapSize(vgpuTypeId):
    c_gsp_heap = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetGspHeapSize")
    ret = fn(vgpuTypeId, byref(c_gsp_heap))
    _nvmlCheckReturn(ret)
    return c_gsp_heap.value

def nvmlVgpuTypeGetFbReservation(vgpuTypeId):
    c_fb_reservation = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetFbReservation")
    ret = fn(vgpuTypeId, byref(c_fb_reservation))
    _nvmlCheckReturn(ret)
    return c_fb_reservation.value

def nvmlVgpuTypeGetMaxInstances(handle, vgpuTypeId):
    c_max_instances = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetMaxInstances")
    ret = fn(handle, vgpuTypeId, byref(c_max_instances))
    _nvmlCheckReturn(ret)
    return c_max_instances.value

def nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId):
    c_max_instances_per_vm = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetMaxInstancesPerVm")
    ret = fn(vgpuTypeId, byref(c_max_instances_per_vm))
    _nvmlCheckReturn(ret)
    return c_max_instances_per_vm.value

def nvmlDeviceGetActiveVgpus(handle):
    # first call to get the size
    c_vgpu_count = c_uint(0)

    fn  = _nvmlGetFunctionPointer("mxSmlExDeviceGetActiveVgpus")
    ret = fn(handle, byref(c_vgpu_count), None)

    if (ret == MXSMLEX_SUCCESS):
        # special case, no active vGPUs
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        vgpu_instance_array = _nvmlVgpuInstance_t * c_vgpu_count.value
        c_vgpu_instances = vgpu_instance_array()

        # make the call again
        ret = fn(handle, byref(c_vgpu_count), c_vgpu_instances)
        _nvmlCheckReturn(ret)
        vgpus = []
        for i in range(c_vgpu_count.value):
            vgpus.append(c_vgpu_instances[i])
        return vgpus
    else:
        # error case
        raise NVMLError(ret)

@convertStrBytes
def nvmlVgpuInstanceGetVmID(vgpuInstance):
    c_vm_id = create_string_buffer(MXSMLEX_DEVICE_UUID_BUFFER_SIZE)
    c_buffer_size = c_uint(MXSMLEX_GRID_LICENSE_BUFFER_SIZE)
    c_vm_id_type  = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetVmID")
    ret = fn(vgpuInstance, byref(c_vm_id), c_buffer_size, byref(c_vm_id_type))
    _nvmlCheckReturn(ret)
    return (c_vm_id.value, c_vm_id_type.value)

@convertStrBytes
def nvmlVgpuInstanceGetUUID(vgpuInstance):
    c_uuid = create_string_buffer(MXSMLEX_DEVICE_UUID_BUFFER_SIZE)
    c_buffer_size = c_uint(MXSMLEX_DEVICE_UUID_BUFFER_SIZE)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetUUID")
    ret = fn(vgpuInstance, byref(c_uuid), c_buffer_size)
    _nvmlCheckReturn(ret)
    return c_uuid.value

@convertStrBytes
def nvmlVgpuInstanceGetMdevUUID(vgpuInstance):
    c_uuid = create_string_buffer(MXSMLEX_DEVICE_UUID_BUFFER_SIZE)
    c_buffer_size = c_uint(MXSMLEX_DEVICE_UUID_BUFFER_SIZE)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetMdevUUID")
    ret = fn(vgpuInstance, byref(c_uuid), c_buffer_size)
    _nvmlCheckReturn(ret)
    return c_uuid.value

@convertStrBytes
def nvmlVgpuInstanceGetVmDriverVersion(vgpuInstance):
    c_driver_version = create_string_buffer(MXSMLEX_SYSTEM_DRIVER_VERSION_BUFFER_SIZE)
    c_buffer_size = c_uint(MXSMLEX_SYSTEM_DRIVER_VERSION_BUFFER_SIZE)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetVmDriverVersion")
    ret = fn(vgpuInstance, byref(c_driver_version), c_buffer_size)
    _nvmlCheckReturn(ret)
    return c_driver_version.value

def nvmlVgpuInstanceGetLicenseStatus(vgpuInstance):
    c_license_status = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetLicenseStatus")
    ret = fn(vgpuInstance, byref(c_license_status))
    _nvmlCheckReturn(ret)
    return c_license_status.value

def nvmlVgpuInstanceGetLicenseInfo_v2(vgpuInstance):
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetLicenseInfo_v2")
    c_license_info = c_nvmlVgpuLicenseInfo_t()
    ret = fn(vgpuInstance, byref(c_license_info))
    _nvmlCheckReturn(ret)
    return c_license_info

def nvmlVgpuInstanceGetLicenseInfo(vgpuInstance):
    return nvmlVgpuInstanceGetLicenseInfo_v2(vgpuInstance)

def nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance):
    c_frl = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetFrameRateLimit")
    ret = fn(vgpuInstance, byref(c_frl))
    _nvmlCheckReturn(ret)
    return c_frl.value

def nvmlVgpuInstanceGetEccMode(vgpuInstance):
    c_mode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetEccMode")
    ret = fn(vgpuInstance, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value

def nvmlVgpuInstanceGetType(vgpuInstance):
    c_vgpu_type = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetType")
    ret = fn(vgpuInstance, byref(c_vgpu_type))
    _nvmlCheckReturn(ret)
    return c_vgpu_type.value

def nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance):
    c_encoder_capacity = c_ulonglong(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetEncoderCapacity")
    ret = fn(vgpuInstance, byref(c_encoder_capacity))
    _nvmlCheckReturn(ret)
    return c_encoder_capacity.value

def nvmlVgpuInstanceSetEncoderCapacity(vgpuInstance, encoder_capacity):
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceSetEncoderCapacity")
    return fn(vgpuInstance, encoder_capacity)

def nvmlVgpuInstanceGetFbUsage(vgpuInstance):
    c_fb_usage = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetFbUsage")
    ret = fn(vgpuInstance, byref(c_fb_usage))
    _nvmlCheckReturn(ret)
    return c_fb_usage.value

def nvmlVgpuTypeGetCapabilities(vgpuTypeId, capability):
    c_cap_result = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuTypeGetCapabilities")
    ret = fn(vgpuTypeId, _nvmlVgpuCapability_t(capability), byref(c_cap_result))
    _nvmlCheckReturn(ret)
    return (c_cap_result.value)

def nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance):
    c_id = c_uint(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetGpuInstanceId")
    ret = fn(vgpuInstance, byref(c_id))
    _nvmlCheckReturn(ret)
    return (c_id.value)

@convertStrBytes
def nvmlVgpuInstanceGetGpuPciId(vgpuInstance):
    c_vgpuPciId = create_string_buffer(MXSMLEX_DEVICE_PCI_BUS_ID_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetGpuPciId")
    ret = fn(vgpuInstance, c_vgpuPciId, byref(c_uint(MXSMLEX_DEVICE_PCI_BUS_ID_BUFFER_SIZE)))
    _nvmlCheckReturn(ret)
    return c_vgpuPciId.value

def nvmlDeviceGetVgpuUtilization(handle, timeStamp):
    # first call to get the size
    c_vgpu_count = c_uint(0)
    c_time_stamp = c_ulonglong(timeStamp)
    c_sample_value_type = _nvmlValueType_t()

    fn  = _nvmlGetFunctionPointer("mxSmlExDeviceGetVgpuUtilization")
    ret = fn(handle, c_time_stamp, byref(c_sample_value_type), byref(c_vgpu_count), None)

    if (ret == MXSMLEX_SUCCESS):
        # special case, no active vGPUs
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        sampleArray = c_vgpu_count.value * c_nvmlVgpuInstanceUtilizationSample_t
        c_samples = sampleArray()

        # make the call again
        ret = fn(handle, c_time_stamp, byref(c_sample_value_type), byref(c_vgpu_count), c_samples)
        _nvmlCheckReturn(ret)

        return c_samples[0:c_vgpu_count.value]
    else:
        # error case
        raise NVMLError(ret)

def nvmlDeviceGetVgpuInstancesUtilizationInfo(handle, timeStamp):
    # first call to get the size
    c_time_stamp = c_ulonglong(timeStamp)
    c_vgpuUtilInfo = c_nvmlVgpuInstancesUtilizationInfo_v1_t(0)
    c_vgpuUtilInfo.version = VgpuInstancesUtilizationInfo_v1
    c_vgpuUtilInfo.sampleValType = _nvmlValueType_t()
    c_vgpuUtilInfo.vgpuInstanceCount = c_uint(0)
    c_vgpuUtilInfo.lastSeenTimeStamp = c_time_stamp

    fn  = _nvmlGetFunctionPointer("mxSmlExDeviceGetVgpuInstancesUtilizationInfo")
    ret = fn(handle, byref(c_vgpuUtilInfo))

    if (ret == MXSMLEX_SUCCESS):
        # special case, no active vGPUs
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        sampleArray = c_vgpuUtilInfo.vgpuInstanceCount * c_nvmlVgpuInstanceUtilizationInfo_v1_t
        c_samples = sampleArray()
        c_vgpuUtilInfo.vgpuUtilArray = c_samples

        # make the call again
        ret = fn(handle, byref(c_vgpuUtilInfo))
        _nvmlCheckReturn(ret)

        return c_samples[0:c_vgpuUtilInfo.vgpuInstanceCount]
    else:
        # error case
        raise NVMLError(ret)

def nvmlDeviceGetP2PStatus(device1, device2, p2pIndex):
    c_p2pstatus = _nvmlGpuP2PStatus_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetP2PStatus")
    ret = fn(device1, device2,p2pIndex, byref(c_p2pstatus))
    _nvmlCheckReturn(ret)
    return c_p2pstatus.value

def nvmlDeviceGetGridLicensableFeatures_v4(handle):
    c_get_grid_licensable_features = c_nvmlGridLicensableFeatures_v4_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGridLicensableFeatures_v4")
    ret = fn(handle, byref(c_get_grid_licensable_features))
    _nvmlCheckReturn(ret)

    return (c_get_grid_licensable_features)

def nvmlDeviceGetGridLicensableFeatures(handle):
    return nvmlDeviceGetGridLicensableFeatures_v4(handle)

def nvmlDeviceGetGspFirmwareVersion(handle, version):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGspFirmwareVersion")
    ret = fn(handle, version)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetGspFirmwareMode(handle, isEnabled, defaultMode):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGspFirmwareMode")
    ret = fn(handle, isEnabled, defaultMode)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetEncoderCapacity(handle, encoderQueryType):
    c_encoder_capacity = c_ulonglong(0)
    c_encoderQuery_type = _nvmlEncoderQueryType_t(encoderQueryType)

    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetEncoderCapacity")
    ret = fn(handle, c_encoderQuery_type, byref(c_encoder_capacity))
    _nvmlCheckReturn(ret)
    return c_encoder_capacity.value

def nvmlDeviceGetVgpuProcessUtilization(handle, timeStamp):
    # first call to get the size
    c_vgpu_count = c_uint(0)
    c_time_stamp = c_ulonglong(timeStamp)

    fn  = _nvmlGetFunctionPointer("mxSmlExDeviceGetVgpuProcessUtilization")
    ret = fn(handle, c_time_stamp, byref(c_vgpu_count), None)

    if (ret == MXSMLEX_SUCCESS):
        # special case, no active vGPUs
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        sampleArray = c_vgpu_count.value * c_nvmlVgpuProcessUtilizationSample_t
        c_samples = sampleArray()

        # make the call again
        ret = fn(handle, c_time_stamp, byref(c_vgpu_count), c_samples)
        _nvmlCheckReturn(ret)

        return c_samples[0:c_vgpu_count.value]
    else:
        # error case
        raise NVMLError(ret)

def nvmlDeviceGetVgpuProcessesUtilizationInfo(handle, timeStamp):
    # first call to get the size
    c_time_stamp = c_ulonglong(timeStamp)
    c_vgpuProcUtilInfo = c_nvmlVgpuProcessesUtilizationInfo_v1_t(0)
    c_vgpuProcUtilInfo.version = VgpuProcessesUtilizationInfo_v1
    c_vgpuProcUtilInfo.vgpuProcessCount = c_uint(0)
    c_vgpuProcUtilInfo.lastSeenTimeStamp = c_time_stamp

    fn  = _nvmlGetFunctionPointer("mxSmlExDeviceGetVgpuProcessesUtilizationInfo")
    ret = fn(handle, byref(c_vgpuProcUtilInfo))

    if (ret == MXSMLEX_SUCCESS):
        # special case, no active vGPUs
        return []
    elif (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        sampleArray = c_vgpuProcUtilInfo.vgpuProcessCount * c_nvmlVgpuProcessUtilizationInfo_v1_t
        c_samples = sampleArray()
        c_vgpuProcUtilInfo.vgpuProcUtilArray = c_samples

        # make the call again
        ret = fn(handle, byref(c_vgpuProcUtilInfo))
        _nvmlCheckReturn(ret)

        return c_samples[0:c_vgpuProcUtilInfo.vgpuProcessCount]
    else:
        # error case
        raise NVMLError(ret)

def nvmlDeviceGetEncoderStats(handle):
    c_encoderCount = c_ulonglong(0)
    c_encodeFps = c_ulonglong(0)
    c_encoderLatency = c_ulonglong(0)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetEncoderStats")
    ret = fn(handle, byref(c_encoderCount), byref(c_encodeFps), byref(c_encoderLatency))
    _nvmlCheckReturn(ret)
    return (c_encoderCount.value, c_encodeFps.value, c_encoderLatency.value)

def nvmlDeviceGetEncoderSessions(handle):
    # first call to get the size
    c_session_count = c_uint(0)

    fn  = _nvmlGetFunctionPointer("mxSmlExDeviceGetEncoderSessions")
    ret = fn(handle, byref(c_session_count), None)

    if (ret == MXSMLEX_SUCCESS):
        if (c_session_count.value != 0):
            # typical case
            session_array = c_nvmlEncoderSession_t * c_session_count.value
            c_sessions = session_array()

            # make the call again
            ret = fn(handle, byref(c_session_count), c_sessions)
            _nvmlCheckReturn(ret)
            sessions = []
            for i in range(c_session_count.value):
                sessions.append(c_sessions[i])
            return sessions
        else:
            return []  # no active sessions
    else:
        # error case
        raise NVMLError(ret)

def nvmlDeviceGetFBCStats(handle):
    c_fbcStats = c_nvmlFBCStats_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetFBCStats")
    ret = fn(handle, byref(c_fbcStats))
    _nvmlCheckReturn(ret)
    return c_fbcStats

def nvmlDeviceGetFBCSessions(handle):
    # first call to get the size
    c_session_count = c_uint(0)

    fn  = _nvmlGetFunctionPointer("mxSmlExDeviceGetFBCSessions")
    ret = fn(handle, byref(c_session_count), None)

    if (ret == MXSMLEX_SUCCESS):
        if (c_session_count.value != 0):
            # typical case
            session_array = c_nvmlFBCSession_t * c_session_count.value
            c_sessions = session_array()

            # make the call again
            ret = fn(handle, byref(c_session_count), c_sessions)
            _nvmlCheckReturn(ret)
            sessions = []
            for i in range(c_session_count.value):
                sessions.append(c_sessions[i])
            return sessions
        else:
            return []  # no active sessions
    else:
        # error case
        raise NVMLError(ret)

def nvmlVgpuInstanceGetEncoderStats(vgpuInstance):
    c_encoderCount    = c_ulonglong(0)
    c_encodeFps       = c_ulonglong(0)
    c_encoderLatency  = c_ulonglong(0)
    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetEncoderStats")
    ret = fn(vgpuInstance, byref(c_encoderCount), byref(c_encodeFps), byref(c_encoderLatency))
    _nvmlCheckReturn(ret)
    return (c_encoderCount.value, c_encodeFps.value, c_encoderLatency.value)

def nvmlVgpuInstanceGetEncoderSessions(vgpuInstance):
    # first call to get the size
    c_session_count = c_uint(0)

    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetEncoderSessions")
    ret = fn(vgpuInstance, byref(c_session_count), None)

    if (ret == MXSMLEX_SUCCESS):
        if (c_session_count.value != 0):
            # typical case
            session_array = c_nvmlEncoderSession_t * c_session_count.value
            c_sessions = session_array()

            # make the call again
            ret = fn(vgpuInstance, byref(c_session_count), c_sessions)
            _nvmlCheckReturn(ret)
            sessions = []
            for i in range(c_session_count.value):
                sessions.append(c_sessions[i])
            return sessions
        else:
            return []  # no active sessions
    else:
        # error case
        raise NVMLError(ret)

def nvmlVgpuInstanceGetFBCStats(vgpuInstance):
    c_fbcStats = c_nvmlFBCStats_t()
    fn = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetFBCStats")
    ret = fn(vgpuInstance, byref(c_fbcStats))
    _nvmlCheckReturn(ret)
    return c_fbcStats

def nvmlVgpuInstanceGetFBCSessions(vgpuInstance):
    # first call to get the size
    c_session_count = c_uint(0)

    fn  = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetFBCSessions")
    ret = fn(vgpuInstance, byref(c_session_count), None)

    if (ret == MXSMLEX_SUCCESS):
        if (c_session_count.value != 0):
            # typical case
            session_array = c_nvmlFBCSession_t * c_session_count.value
            c_sessions = session_array()

            # make the call again
            ret = fn(vgpuInstance, byref(c_session_count), c_sessions)
            _nvmlCheckReturn(ret)
            sessions = []
            for i in range(c_session_count.value):
                sessions.append(c_sessions[i])
            return sessions
        else:
            return []  # no active sessions
    else:
        # error case
        raise NVMLError(ret)

def nvmlDeviceGetProcessUtilization(handle, timeStamp):
    # first call to get the size
    c_count = c_uint(0)
    c_time_stamp = c_ulonglong(timeStamp)

    fn  = _nvmlGetFunctionPointer("mxSmlExDeviceGetProcessUtilization")
    ret = fn(handle, None, byref(c_count), c_time_stamp)

    if (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        sampleArray = c_count.value * c_nvmlProcessUtilizationSample_t
        c_samples = sampleArray()

        # make the call again
        ret = fn(handle, c_samples, byref(c_count), c_time_stamp)
        _nvmlCheckReturn(ret)

        return c_samples[0:c_count.value]
    else:
        # error case
        raise NVMLError(ret)

def nvmlDeviceGetProcessesUtilizationInfo(handle, timeStamp):
    # first call to get the size
    c_time_stamp = c_ulonglong(timeStamp)
    c_processesUtilInfo = c_nvmlProcessesUtilizationInfo_v1_t(0)
    c_processesUtilInfo.version = ProcessesUtilizationInfo_v1
    c_processesUtilInfo.processSamplesCount = c_uint(0)
    c_processesUtilInfo.lastSeenTimeStamp = c_time_stamp

    fn  = _nvmlGetFunctionPointer("mxSmlExDeviceGetProcessesUtilizationInfo")
    ret = fn(handle, byref(c_processesUtilInfo))

    if (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        # typical case
        sampleArray = c_processesUtilInfo.processSamplesCount * c_nvmlProcessUtilizationInfo_v1_t
        c_samples = sampleArray()
        c_processesUtilInfo.procUtilArray = c_samples

        # make the call again
        ret = fn(handle, byref(c_processesUtilInfo))
        _nvmlCheckReturn(ret)

        return c_samples[0:c_processesUtilInfo.processSamplesCount]
    else:
        # error case
        raise NVMLError(ret)

def nvmlVgpuInstanceGetMetadata(vgpuInstance):
    fn = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetMetadata")
    c_vgpuMetadata = c_nvmlVgpuMetadata_t()
    c_bufferSize = c_uint(0)
    # Make the first MXSMLEX API call to get the c_bufferSize value.
    # We have already allocated required buffer above.
    ret = fn(vgpuInstance, byref(c_vgpuMetadata), byref(c_bufferSize))
    if (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        ret = fn(vgpuInstance, byref(c_vgpuMetadata), byref(c_bufferSize))
        _nvmlCheckReturn(ret)
    else:
        raise NVMLError(ret)
    return c_vgpuMetadata

def nvmlDeviceGetVgpuMetadata(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetVgpuMetadata")
    c_vgpuPgpuMetadata = c_nvmlVgpuPgpuMetadata_t()
    c_bufferSize = c_uint(0)
    # Make the first MXSMLEX API call to get the c_bufferSize value.
    # We have already allocated required buffer above.
    ret = fn(handle, byref(c_vgpuPgpuMetadata), byref(c_bufferSize))
    if (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        ret = fn(handle, byref(c_vgpuPgpuMetadata), byref(c_bufferSize))
        _nvmlCheckReturn(ret)
    else:
        raise NVMLError(ret)
    return c_vgpuPgpuMetadata

def nvmlGetVgpuCompatibility(vgpuMetadata, pgpuMetadata):
    fn = _nvmlGetFunctionPointer("mxSmlExGetVgpuCompatibility")
    c_vgpuPgpuCompatibility = c_nvmlVgpuPgpuCompatibility_t()
    ret = fn(byref(vgpuMetadata), byref(pgpuMetadata), byref(c_vgpuPgpuCompatibility))
    _nvmlCheckReturn(ret)
    return c_vgpuPgpuCompatibility

@convertStrBytes
def nvmlDeviceGetPgpuMetadataString(handle):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPgpuMetadataString")
    c_pgpuMetadata = create_string_buffer(MXSMLEX_VGPU_PGPU_METADATA_OPAQUE_DATA_SIZE)
    c_bufferSize = c_uint(0)
    # Make the first MXSMLEX API call to get the c_bufferSize value.
    # We have already allocated required buffer above.
    ret = fn(handle, byref(c_pgpuMetadata), byref(c_bufferSize))
    if (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        ret = fn(handle, byref(c_pgpuMetadata), byref(c_bufferSize))
        _nvmlCheckReturn(ret)
    else:
        raise NVMLError(ret)
    return (c_pgpuMetadata.value, c_bufferSize.value)

def nvmlDeviceGetVgpuSchedulerLog(handle):
    c_vgpu_sched_log = c_nvmlVgpuSchedulerLog_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetVgpuSchedulerLog")
    ret = fn(handle, byref(c_vgpu_sched_log))
    _nvmlCheckReturn(ret)
    return c_vgpu_sched_log

def nvmlDeviceGetVgpuSchedulerState(handle):
    c_vgpu_sched_state = c_nvmlVgpuSchedulerGetState_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetVgpuSchedulerState")
    ret = fn(handle, byref(c_vgpu_sched_state))
    _nvmlCheckReturn(ret)
    return c_vgpu_sched_state

def nvmlDeviceGetVgpuSchedulerCapabilities(handle):
    c_vgpu_sched_caps = c_nvmlVgpuSchedulerCapabilities_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetVgpuSchedulerCapabilities")
    ret = fn(handle, byref(c_vgpu_sched_caps))
    _nvmlCheckReturn(ret)
    return c_vgpu_sched_caps

def nvmlDeviceSetVgpuSchedulerState(handle, sched_state):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetVgpuSchedulerState")
    ret = fn(handle, byref(sched_state))
    _nvmlCheckReturn(ret)
    return ret

def nvmlSetVgpuVersion(vgpuVersion):
    fn = _nvmlGetFunctionPointer("mxSmlExSetVgpuVersion")
    ret = fn(byref(vgpuVersion))
    _nvmlCheckReturn(ret)
    return ret

def nvmlGetVgpuVersion(supported, current):
    fn = _nvmlGetFunctionPointer("mxSmlExGetVgpuVersion")
    ret = fn(byref(supported), byref(current))
    _nvmlCheckReturn(ret)
    return ret

def nvmlVgpuInstanceGetAccountingMode(vgpuInstance):
    c_mode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetAccountingMode")
    ret = fn(vgpuInstance, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value

def nvmlVgpuInstanceGetAccountingPids(vgpuInstance):
    c_pidCount = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetAccountingPids")
    ret = fn(vgpuInstance, byref(c_pidCount), None)
    if (ret == MXSMLEX_ERROR_INSUFFICIENT_SIZE):
        sampleArray = c_pidCount.value * c_uint
        c_pidArray = sampleArray()
        ret = fn(vgpuInstance, byref(c_pidCount), byref(c_pidArray))
        _nvmlCheckReturn(ret)
    else:
        raise NVMLError(ret)
    return (c_pidCount, c_pidArray)

def nvmlVgpuInstanceGetAccountingStats(vgpuInstance, pid):
    c_accountingStats = c_nvmlAccountingStats_t()
    fn = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceGetAccountingStats")
    ret = fn(vgpuInstance, pid, byref(c_accountingStats))
    _nvmlCheckReturn(ret)
    return c_accountingStats

def nvmlVgpuInstanceClearAccountingPids(vgpuInstance):
    fn = _nvmlGetFunctionPointer("mxSmlExVgpuInstanceClearAccountingPids")
    ret = fn(vgpuInstance)
    _nvmlCheckReturn(ret)
    return ret

def nvmlGetExcludedDeviceCount():
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExGetExcludedDeviceCount")
    ret = fn(byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlGetExcludedDeviceInfoByIndex(index):
    c_index = c_uint(index)
    info = c_nvmlExcludedDeviceInfo_t()
    fn = _nvmlGetFunctionPointer("mxSmlExGetExcludedDeviceInfoByIndex")
    ret = fn(c_index, byref(info))
    _nvmlCheckReturn(ret)
    return info

def nvmlDeviceGetHostVgpuMode(handle):
    c_host_vgpu_mode = _nvmlHostVgpuMode_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetHostVgpuMode")
    ret = fn(handle, byref(c_host_vgpu_mode))
    _nvmlCheckReturn(ret)
    return c_host_vgpu_mode.value

def nvmlDeviceSetMigMode(device, mode):
    c_activationStatus = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetMigMode")
    ret = fn(device, mode, byref(c_activationStatus))
    _nvmlCheckReturn(ret)
    return c_activationStatus.value

def nvmlDeviceGetMigMode(device):
    c_currentMode = c_uint()
    c_pendingMode = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMigMode")
    ret = fn(device, byref(c_currentMode), byref(c_pendingMode))
    _nvmlCheckReturn(ret)
    return [c_currentMode.value, c_pendingMode.value]

def nvmlDeviceGetGpuInstanceProfileInfo(device, profile, version=2):
    if version == 2:
        c_info = c_nvmlGpuInstanceProfileInfo_v2_t()
        fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGpuInstanceProfileInfoV")
    elif version == 1:
        c_info = c_nvmlGpuInstanceProfileInfo_t()
        fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGpuInstanceProfileInfo")
    else:
        raise NVMLError(MXSMLEX_ERROR_FUNCTION_NOT_FOUND)
    ret = fn(device, profile, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info

# Define function alias for the API exposed by MXSMLEX
nvmlDeviceGetGpuInstanceProfileInfoV = nvmlDeviceGetGpuInstanceProfileInfo

def nvmlDeviceGetGpuInstanceRemainingCapacity(device, profileId):
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGpuInstanceRemainingCapacity")
    ret = fn(device, profileId, byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlDeviceGetGpuInstancePossiblePlacements(device, profileId, placementsRef, countRef):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGpuInstancePossiblePlacements_v2")
    ret = fn(device, profileId, placementsRef, countRef)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceCreateGpuInstance(device, profileId):
    c_instance = c_nvmlGpuInstance_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceCreateGpuInstance")
    ret = fn(device, profileId, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance

def nvmlDeviceCreateGpuInstanceWithPlacement(device, profileId, placement):
    c_instance = c_nvmlGpuInstance_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceCreateGpuInstanceWithPlacement")
    ret = fn(device, profileId, placement, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance

def nvmlGpuInstanceDestroy(gpuInstance):
    fn = _nvmlGetFunctionPointer("mxSmlExGpuInstanceDestroy")
    ret = fn(gpuInstance)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetGpuInstances(device, profileId, gpuInstancesRef, countRef):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGpuInstances")
    ret = fn(device, profileId, gpuInstancesRef, countRef)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetGpuInstanceById(device, gpuInstanceId):
    c_instance = c_nvmlGpuInstance_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGpuInstanceById")
    ret = fn(device, gpuInstanceId, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance

def nvmlGpuInstanceGetInfo(gpuInstance):
    c_info = c_nvmlGpuInstanceInfo_t()
    fn = _nvmlGetFunctionPointer("mxSmlExGpuInstanceGetInfo")
    ret = fn(gpuInstance, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info

def nvmlGpuInstanceGetComputeInstanceProfileInfo(device, profile, engProfile, version=2):
    if version == 2:
        c_info = c_nvmlComputeInstanceProfileInfo_v2_t()
        fn = _nvmlGetFunctionPointer("mxSmlExGpuInstanceGetComputeInstanceProfileInfoV")
    elif version == 1:
        c_info = c_nvmlComputeInstanceProfileInfo_t()
        fn = _nvmlGetFunctionPointer("mxSmlExGpuInstanceGetComputeInstanceProfileInfo")
    else:
        raise NVMLError(MXSMLEX_ERROR_FUNCTION_NOT_FOUND) 
    ret = fn(device, profile, engProfile, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info

# Define function alias for the API exposed by MXSMLEX
nvmlGpuInstanceGetComputeInstanceProfileInfoV = nvmlGpuInstanceGetComputeInstanceProfileInfo

def nvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance, profileId):
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExGpuInstanceGetComputeInstanceRemainingCapacity")
    ret = fn(gpuInstance, profileId, byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlGpuInstanceGetComputeInstancePossiblePlacements(gpuInstance, profileId, placementsRef, countRef):
    fn = _nvmlGetFunctionPointer("mxSmlExGpuInstanceGetComputeInstancePossiblePlacements")
    ret = fn(gpuInstance, profileId, placementsRef, countRef)
    _nvmlCheckReturn(ret)
    return ret

def nvmlGpuInstanceCreateComputeInstance(gpuInstance, profileId):
    c_instance = c_nvmlComputeInstance_t()
    fn = _nvmlGetFunctionPointer("mxSmlExGpuInstanceCreateComputeInstance")
    ret = fn(gpuInstance, profileId, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance

def nvmlGpuInstanceCreateComputeInstanceWithPlacement(gpuInstance, profileId, placement):
    c_instance = c_nvmlComputeInstance_t()
    fn = _nvmlGetFunctionPointer("mxSmlExGpuInstanceCreateComputeInstanceWithPlacement")
    ret = fn(gpuInstance, profileId, placement, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance

def nvmlComputeInstanceDestroy(computeInstance):
    fn = _nvmlGetFunctionPointer("mxSmlExComputeInstanceDestroy")
    ret = fn(computeInstance)
    _nvmlCheckReturn(ret)
    return ret

def nvmlGpuInstanceGetComputeInstances(gpuInstance, profileId, computeInstancesRef, countRef):
    fn = _nvmlGetFunctionPointer("mxSmlExGpuInstanceGetComputeInstances")
    ret = fn(gpuInstance, profileId, computeInstancesRef, countRef)
    _nvmlCheckReturn(ret)
    return ret

def nvmlGpuInstanceGetComputeInstanceById(gpuInstance, computeInstanceId):
    c_instance = c_nvmlComputeInstance_t()
    fn = _nvmlGetFunctionPointer("mxSmlExGpuInstanceGetComputeInstanceById")
    ret = fn(gpuInstance, computeInstanceId, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance

def nvmlComputeInstanceGetInfo_v2(computeInstance):
    c_info = c_nvmlComputeInstanceInfo_t()
    fn = _nvmlGetFunctionPointer("mxSmlExComputeInstanceGetInfo_v2")
    ret = fn(computeInstance, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info

def nvmlComputeInstanceGetInfo(computeInstance):
    return nvmlComputeInstanceGetInfo_v2(computeInstance)

def nvmlDeviceIsMigDeviceHandle(device):
    c_isMigDevice = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceIsMigDeviceHandle")
    ret = fn(device, byref(c_isMigDevice))
    _nvmlCheckReturn(ret)
    return c_isMigDevice

def nvmlDeviceGetGpuInstanceId(device):
    c_gpuInstanceId = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGpuInstanceId")
    ret = fn(device, byref(c_gpuInstanceId))
    _nvmlCheckReturn(ret)
    return c_gpuInstanceId.value

def nvmlDeviceGetComputeInstanceId(device):
    c_computeInstanceId = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetComputeInstanceId")
    ret = fn(device, byref(c_computeInstanceId))
    _nvmlCheckReturn(ret)
    return c_computeInstanceId.value

def nvmlDeviceGetMaxMigDeviceCount(device):
    c_count = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMaxMigDeviceCount")
    ret = fn(device, byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value

def nvmlDeviceGetMigDeviceHandleByIndex(device, index):
    c_index = c_uint(index)
    migDevice = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMigDeviceHandleByIndex")
    ret = fn(device, c_index, byref(migDevice))
    _nvmlCheckReturn(ret)
    return migDevice

def nvmlDeviceGetDeviceHandleFromMigDeviceHandle(migDevice):
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetDeviceHandleFromMigDeviceHandle")
    ret = fn(migDevice, byref(device))
    _nvmlCheckReturn(ret)
    return device

def nvmlDeviceGetAttributes_v2(device):
    c_attrs = c_nvmlDeviceAttributes()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetAttributes_v2")
    ret = fn(device, byref(c_attrs))
    _nvmlCheckReturn(ret)
    return c_attrs

def nvmlDeviceGetAttributes(device):
    return nvmlDeviceGetAttributes_v2(device)

def nvmlDeviceGetRemappedRows(device):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetRemappedRows")
    c_corr = c_uint()
    c_unc = c_uint()
    c_bpending = c_uint()
    c_bfailure = c_uint()
    ret = fn(device, byref(c_corr), byref(c_unc), byref(c_bpending), byref(c_bfailure))
    _nvmlCheckReturn(ret)
    return (c_corr.value, c_unc.value, c_bpending.value, c_bfailure.value)

def nvmlDeviceGetRowRemapperHistogram(device):
    c_vals = c_nvmlRowRemapperHistogramValues()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetRowRemapperHistogram")
    ret = fn(device, byref(c_vals))
    _nvmlCheckReturn(ret)
    return c_vals

def nvmlDeviceGetArchitecture(device):
    arch = _nvmlDeviceArchitecture_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetArchitecture")
    ret = fn(device, byref(arch))
    _nvmlCheckReturn(ret)
    return arch.value

def nvmlDeviceGetBusType(device):
    c_busType = _nvmlBusType_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetBusType")
    ret = fn(device, byref(c_busType))
    _nvmlCheckReturn(ret)
    return c_busType.value

def nvmlDeviceGetIrqNum(device):
    c_irqNum = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetIrqNum")
    ret = fn(device, byref(c_irqNum))
    _nvmlCheckReturn(ret)
    return c_irqNum.value

def nvmlDeviceGetNumGpuCores(device):
    c_numCores = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetNumGpuCores")
    ret = fn(device, byref(c_numCores))
    _nvmlCheckReturn(ret)
    return c_numCores.value

def nvmlDeviceGetPowerSource(device):
    c_powerSource = _nvmlPowerSource_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPowerSource")
    ret = fn(device, byref(c_powerSource))
    _nvmlCheckReturn(ret)
    return c_powerSource.value

def nvmlDeviceGetMemoryBusWidth(device):
    c_memBusWidth = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMemoryBusWidth")
    ret = fn(device, byref(c_memBusWidth))
    _nvmlCheckReturn(ret)
    return c_memBusWidth.value

def nvmlDeviceGetPcieLinkMaxSpeed(device):
    c_speed = _nvmlPcieLinkMaxSpeed_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPcieLinkMaxSpeed")
    ret = fn(device, byref(c_speed))
    _nvmlCheckReturn(ret)
    return c_speed.value

def nvmlDeviceGetAdaptiveClockInfoStatus(device):
    c_adaptiveClockInfoStatus = _nvmlAdaptiveClockInfoStatus_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetAdaptiveClockInfoStatus")
    ret = fn(device, byref(c_adaptiveClockInfoStatus))
    _nvmlCheckReturn(ret)
    return c_adaptiveClockInfoStatus.value

def nvmlDeviceGetPcieSpeed(device):
    c_speed = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetPcieSpeed")
    ret = fn(device, byref(c_speed))
    _nvmlCheckReturn(ret)
    return c_speed.value

def nvmlDeviceGetDynamicPstatesInfo(device, c_dynamicpstatesinfo):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetDynamicPstatesInfo");
    ret = fn(device, c_dynamicpstatesinfo)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceSetFanSpeed_v2(handle, index, speed):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetFanSpeed_v2");
    ret = fn(handle, index, speed)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetThermalSettings(device, sensorindex, c_thermalsettings):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetThermalSettings");
    ret = fn(device, sensorindex, c_thermalsettings)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetMinMaxClockOfPState(device, type, pstate, minClockMHz, maxClockMHz):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMinMaxClockOfPState");
    ret = fn(device, _nvmlClockType_t(type), _nvmlClockType_t(pstate), minClockMHz, maxClockMHz)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetSupportedPerformanceStates(device):
    pstates = []
    c_count = c_uint(MXSMLEX_MAX_GPU_PERF_PSTATES)
    c_size = sizeof(c_uint)*c_count.value

    # NOTE: use 'c_uint' to represent the size of the nvmlPstate_t enumeration.
    pstates_array = _nvmlPstates_t * c_count.value
    c_pstates = pstates_array()

    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetSupportedPerformanceStates")
    ret = fn(device, c_pstates, c_size)
    _nvmlCheckReturn(ret)

    for value in c_pstates:
        if value != MXSMLEX_PSTATE_UNKNOWN:
            pstates.append(value)

    return pstates

def nvmlDeviceGetGpcClkVfOffset(device):
    offset = c_int32()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGpcClkVfOffset")
    ret = fn(device, byref(offset))
    _nvmlCheckReturn(ret)
    return offset.value

def nvmlDeviceSetGpcClkVfOffset(device, offset):
    c_offset = c_int32(offset)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetGpcClkVfOffset")
    ret = fn(device, c_offset)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetGpcClkMinMaxVfOffset(device, minOffset, maxOffset):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGpcClkMinMaxVfOffset")
    ret = fn(device, minOffset, maxOffset)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetMemClkVfOffset(device):
    offset = c_int32()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMemClkVfOffset")
    ret = fn(device, byref(offset))
    _nvmlCheckReturn(ret)
    return offset.value

def nvmlDeviceSetMemClkVfOffset(device, offset):
    c_offset = c_int32(offset)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetMemClkVfOffset")
    ret = fn(device, c_offset)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetMemClkMinMaxVfOffset(device, minOffset, maxOffset):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetMemClkMinMaxVfOffset")
    ret = fn(device, minOffset, maxOffset)
    _nvmlCheckReturn(ret)
    return ret

def nvmlSystemSetConfComputeGpusReadyState(state):
    c_state = c_uint(state)
    fn = _nvmlGetFunctionPointer("mxSmlExSystemSetConfComputeGpusReadyState")
    ret = fn(c_state)
    _nvmlCheckReturn(ret)
    return ret

def nvmlSystemGetConfComputeGpusReadyState():
    c_state = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExSystemGetConfComputeGpusReadyState")
    ret = fn(byref(c_state))
    _nvmlCheckReturn(ret)
    return c_state.value

def nvmlSystemGetConfComputeCapabilities():
    c_ccSysCaps = c_nvmlConfComputeSystemCaps_t()
    fn = _nvmlGetFunctionPointer("mxSmlExSystemGetConfComputeCapabilities")
    ret = fn(byref(c_ccSysCaps))
    _nvmlCheckReturn(ret)
    return c_ccSysCaps

def nvmlSystemGetConfComputeState():
    c_state = c_nvmlConfComputeSystemState_t()
    fn = _nvmlGetFunctionPointer("mxSmlExSystemGetConfComputeState")
    ret = fn(byref(c_state))
    _nvmlCheckReturn(ret)
    return c_state

def nvmlSystemGetConfComputeSettings(settings):
    fn = _nvmlGetFunctionPointer("mxSmlExSystemGetConfComputeSettings")
    return fn(settings)

def nvmlDeviceSetConfComputeUnprotectedMemSize(device, c_ccMemSize):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetConfComputeUnprotectedMemSize")
    ret = fn(device, c_ccMemSize)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetConfComputeMemSizeInfo(device):
    c_ccMemSize = c_nvmlConfComputeMemSizeInfo_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetConfComputeMemSizeInfo")
    ret = fn(device, byref(c_ccMemSize))
    _nvmlCheckReturn(ret)
    return c_ccMemSize

def nvmlDeviceGetConfComputeProtectedMemoryUsage(device):
    c_memory = c_nvmlMemory_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetConfComputeProtectedMemoryUsage")
    ret = fn(device, byref(c_memory))
    _nvmlCheckReturn(ret)
    return c_memory

def nvmlDeviceGetConfComputeGpuCertificate(device):
    c_cert = c_nvmlConfComputeGpuCertificate_t()
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetConfComputeGpuCertificate")
    ret = fn(device, byref(c_cert))
    _nvmlCheckReturn(ret)
    return c_cert

def nvmlDeviceGetConfComputeGpuAttestationReport(device, c_nonce):
    c_attestReport = c_nvmlConfComputeGpuAttestationReport_t()
    c_nonce_arr = (c_uint8 * len(c_nonce))(*(c_nonce))
    setattr(c_attestReport, 'nonce', c_nonce_arr)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetConfComputeGpuAttestationReport")
    ret = fn(device, byref(c_attestReport))
    _nvmlCheckReturn(ret)
    return c_attestReport

def nvmlSystemSetConfComputeKeyRotationThresholdInfo(max_atk_adv):
    c_keyRotationThrInfo = c_nvmlConfComputeSetKeyRotationThresholdInfo_t(0)
    c_keyRotationThrInfo.version = ConfComputeSetKeyRotationThresholdInfo_v1
    c_keyRotationThrInfo.maxAttackerAdvantage = max_atk_adv
    fn = _nvmlGetFunctionPointer("mxSmlExSystemSetConfComputeKeyRotationThresholdInfo")
    ret = fn(byref(c_keyRotationThrInfo))
    _nvmlCheckReturn(ret)
    return ret

def nvmlSystemGetConfComputeKeyRotationThresholdInfo():
    c_keyRotationThrInfo = c_nvmlConfComputeGetKeyRotationThresholdInfo_t(0)
    c_keyRotationThrInfo.version = ConfComputeGetKeyRotationThresholdInfo_v1
    fn = _nvmlGetFunctionPointer("mxSmlExSystemGetConfComputeKeyRotationThresholdInfo")
    ret = fn(byref(c_keyRotationThrInfo))
    _nvmlCheckReturn(ret)
    return c_keyRotationThrInfo

## GPM ##
#########

## Enums/defines

#### GPM Metric Identifiers
MXSMLEX_GPM_METRIC_GRAPHICS_UTIL           = 1 # Percentage of time any compute/graphics app was active on the GPU. 0.0 - 100.0
MXSMLEX_GPM_METRIC_SM_UTIL                 = 2 # Percentage of SMs that were busy. 0.0 - 100.0
MXSMLEX_GPM_METRIC_SM_OCCUPANCY            = 3 # Percentage of warps that were active vs theoretical maximum. 0.0 - 100.0
MXSMLEX_GPM_METRIC_INTEGER_UTIL            = 4 # Percentage of time the GPU's SMs were doing integer operations. 0.0 - 100.0
MXSMLEX_GPM_METRIC_ANY_TENSOR_UTIL         = 5 # Percentage of time the GPU's SMs were doing ANY tensor operations. 0.0 - 100.0
MXSMLEX_GPM_METRIC_DFMA_TENSOR_UTIL        = 6 # Percentage of time the GPU's SMs were doing DFMA tensor operations. 0.0 - 100.0
MXSMLEX_GPM_METRIC_HMMA_TENSOR_UTIL        = 7 # Percentage of time the GPU's SMs were doing HMMA tensor operations. 0.0 - 100.0
MXSMLEX_GPM_METRIC_IMMA_TENSOR_UTIL        = 9 # Percentage of time the GPU's SMs were doing IMMA tensor operations. 0.0 - 100.0
MXSMLEX_GPM_METRIC_DRAM_BW_UTIL            = 10 # Percentage of DRAM bw used vs theoretical maximum. 0.0 - 100.0
MXSMLEX_GPM_METRIC_FP64_UTIL               = 11 # Percentage of time the GPU's SMs were doing non-tensor FP64 math. 0.0 - 100.0
MXSMLEX_GPM_METRIC_FP32_UTIL               = 12 # Percentage of time the GPU's SMs were doing non-tensor FP32 math. 0.0 - 100.0
MXSMLEX_GPM_METRIC_FP16_UTIL               = 13 # Percentage of time the GPU's SMs were doing non-tensor FP16 math. 0.0 - 100.0
MXSMLEX_GPM_METRIC_PCIE_TX_PER_SEC         = 20 # PCIe traffic from this GPU in MiB/sec
MXSMLEX_GPM_METRIC_PCIE_RX_PER_SEC         = 21 # PCIe traffic to this GPU in MiB/sec
MXSMLEX_GPM_METRIC_NVDEC_0_UTIL            = 30 # Percent utilization of NVDEC 0. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVDEC_1_UTIL            = 31 # Percent utilization of NVDEC 1. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVDEC_2_UTIL            = 32 # Percent utilization of NVDEC 2. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVDEC_3_UTIL            = 33 # Percent utilization of NVDEC 3. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVDEC_4_UTIL            = 34 # Percent utilization of NVDEC 4. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVDEC_5_UTIL            = 35 # Percent utilization of NVDEC 5. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVDEC_6_UTIL            = 36 # Percent utilization of NVDEC 6. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVDEC_7_UTIL            = 37 # Percent utilization of NVDEC 7. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVJPG_0_UTIL            = 40 # Percent utilization of NVJPG 0. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVJPG_1_UTIL            = 41 # Percent utilization of NVJPG 1. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVJPG_2_UTIL            = 42 # Percent utilization of NVJPG 2. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVJPG_3_UTIL            = 43 # Percent utilization of NVJPG 3. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVJPG_4_UTIL            = 44 # Percent utilization of NVJPG 4. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVJPG_5_UTIL            = 45 # Percent utilization of NVJPG 5. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVJPG_6_UTIL            = 46 # Percent utilization of NVJPG 6. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVJPG_7_UTIL            = 47 # Percent utilization of NVJPG 7. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVOFA_0_UTIL            = 50 # Percent utilization of NVOFA 0. 0.0 - 100.0
MXSMLEX_GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC = 60 # NvLink read bandwidth for all links in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC = 61 # NvLink write bandwidth for all links in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L0_RX_PER_SEC    = 62 # NvLink read bandwidth for link 0 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L0_TX_PER_SEC    = 63 # NvLink write bandwidth for link 0 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L1_RX_PER_SEC    = 64 # NvLink read bandwidth for link 1 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L1_TX_PER_SEC    = 65 # NvLink write bandwidth for link 1 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L2_RX_PER_SEC    = 66 # NvLink read bandwidth for link 2 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L2_TX_PER_SEC    = 67 # NvLink write bandwidth for link 2 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L3_RX_PER_SEC    = 68 # NvLink read bandwidth for link 3 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L3_TX_PER_SEC    = 69 # NvLink write bandwidth for link 3 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L4_RX_PER_SEC    = 70 # NvLink read bandwidth for link 4 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L4_TX_PER_SEC    = 71 # NvLink write bandwidth for link 4 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L5_RX_PER_SEC    = 72 # NvLink read bandwidth for link 5 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L5_TX_PER_SEC    = 73 # NvLink write bandwidth for link 5 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L6_RX_PER_SEC    = 74 # NvLink read bandwidth for link 6 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L6_TX_PER_SEC    = 75 # NvLink write bandwidth for link 6 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L7_RX_PER_SEC    = 76 # NvLink read bandwidth for link 7 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L7_TX_PER_SEC    = 77 # NvLink write bandwidth for link 7 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L8_RX_PER_SEC    = 78 # NvLink read bandwidth for link 8 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L8_TX_PER_SEC    = 79 # NvLink write bandwidth for link 8 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L9_RX_PER_SEC    = 80 # NvLink read bandwidth for link 9 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L9_TX_PER_SEC    = 81 # NvLink write bandwidth for link 9 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L10_RX_PER_SEC   = 82 # NvLink read bandwidth for link 10 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L10_TX_PER_SEC   = 83 # NvLink write bandwidth for link 10 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L11_RX_PER_SEC   = 84 # NvLink read bandwidth for link 11 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L11_TX_PER_SEC   = 85 # NvLink write bandwidth for link 11 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L12_RX_PER_SEC   = 86 # NvLink read bandwidth for link 12 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L12_TX_PER_SEC   = 87 # NvLink write bandwidth for link 12 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L13_RX_PER_SEC   = 88 # NvLink read bandwidth for link 13 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L13_TX_PER_SEC   = 89 # NvLink write bandwidth for link 13 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L14_RX_PER_SEC   = 90 # NvLink read bandwidth for link 14 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L14_TX_PER_SEC   = 91 # NvLink write bandwidth for link 14 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L15_RX_PER_SEC   = 92 # NvLink read bandwidth for link 15 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L15_TX_PER_SEC   = 93 # NvLink write bandwidth for link 15 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L16_RX_PER_SEC   = 94 # NvLink read bandwidth for link 16 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L16_TX_PER_SEC   = 95 # NvLink write bandwidth for link 16 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L17_RX_PER_SEC   = 96 # NvLink read bandwidth for link 17 in MiB/sec
MXSMLEX_GPM_METRIC_NVLINK_L17_TX_PER_SEC   = 97 # NvLink write bandwidth for link 17 in MiB/sec
MXSMLEX_GPM_METRIC_MAX                     = 98

## Structs

class c_nvmlUnitInfo_t(_PrintableStructure):
    _fields_ = [
        ('name', c_char * 96),
        ('id', c_char * 96),
        ('serial', c_char * 96),
        ('firmwareVersion', c_char * 96),
    ]

class struct_c_nvmlGpmSample_t(Structure):
    pass # opaque handle
c_nvmlGpmSample_t = POINTER(struct_c_nvmlGpmSample_t)

class c_metricInfo_t(Structure):
    _fields_ = [
        ("shortName", c_char_p),
        ("longName", c_char_p),
        ("unit", c_char_p),
    ]

class c_nvmlGpmMetric_t(_PrintableStructure):
    _fields_ = [
        ('metricId', c_uint),
        ('nvmlReturn', _nvmlReturn_t),
        ('value', c_double),
        ('metricInfo', c_metricInfo_t)
    ]

class c_nvmlGpmMetricsGet_t(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('numMetrics', c_uint),
        ('sample1', c_nvmlGpmSample_t),
        ('sample2', c_nvmlGpmSample_t),
        ('metrics', c_nvmlGpmMetric_t * MXSMLEX_GPM_METRIC_MAX)
    ]

MXSMLEX_GPM_METRICS_GET_VERSION = 1

class c_nvmlGpmSupport_t(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('isSupportedDevice', c_uint),
    ]

MXSMLEX_GPM_SUPPORT_VERSION = 1

## Functions

def nvmlGpmMetricsGet(metricsGet):
    fn = _nvmlGetFunctionPointer("mxSmlExGpmMetricsGet")
    ret = fn(byref(metricsGet))
    _nvmlCheckReturn(ret)
    return metricsGet

def nvmlGpmSampleFree(gpmSample):
    fn = _nvmlGetFunctionPointer("mxSmlExGpmSampleFree")
    ret = fn(gpmSample)
    _nvmlCheckReturn(ret)
    return

def nvmlGpmSampleAlloc():
    gpmSample = c_nvmlGpmSample_t()
    fn = _nvmlGetFunctionPointer("mxSmlExGpmSampleAlloc")
    ret = fn(byref(gpmSample))
    _nvmlCheckReturn(ret)
    return gpmSample

def nvmlGpmSampleGet(device, gpmSample):
    fn = _nvmlGetFunctionPointer("mxSmlExGpmSampleGet")
    ret = fn(device, gpmSample)
    _nvmlCheckReturn(ret)
    return gpmSample

def nvmlGpmMigSampleGet(device, gpuInstanceId, gpmSample):
    fn = _nvmlGetFunctionPointer("mxSmlExGpmMigSampleGet")
    ret = fn(device, gpuInstanceId, gpmSample)
    _nvmlCheckReturn(ret)
    return gpmSample

def nvmlGpmQueryDeviceSupport(device):
    gpmSupport = c_nvmlGpmSupport_t()
    gpmSupport.version = MXSMLEX_GPM_SUPPORT_VERSION
    fn = _nvmlGetFunctionPointer("mxSmlExGpmQueryDeviceSupport")
    ret = fn(device, byref(gpmSupport))
    _nvmlCheckReturn(ret)
    return gpmSupport

def nvmlGpmSetStreamingEnabled(device, state):
    c_state = c_uint(state)
    fn = _nvmlGetFunctionPointer("mxSmlExGpmSetStreamingEnabled")
    ret = fn(device, c_state)
    _nvmlCheckReturn(ret)
    return ret

def nvmlGpmQueryIfStreamingEnabled(device):
    c_state = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExGpmQueryIfStreamingEnabled")
    ret = fn(device, byref(c_state))
    _nvmlCheckReturn(ret)
    return c_state.value

# Low Power Structure and Function

MXSMLEX_NVLINK_POWER_STATE_HIGH_SPEED    = 0x0
MXSMLEX_NVLINK_POWER_STATE_LOW           = 0x1

MXSMLEX_NVLINK_LOW_POWER_THRESHOLD_MIN   = 0x1
MXSMLEX_NVLINK_LOW_POWER_THRESHOLD_MAX   = 0x1FFF
MXSMLEX_NVLINK_LOW_POWER_THRESHOLD_RESET = 0xFFFFFFFF

class c_nvmlNvLinkPowerThres_t(Structure):
    _fields_ = [
        ("lowPwrThreshold", c_uint),
    ]

def nvmlDeviceSetNvLinkDeviceLowPowerThreshold(device, l1threshold):
    c_info = c_nvmlNvLinkPowerThres_t()
    c_info.lowPwrThreshold = l1threshold
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetNvLinkDeviceLowPowerThreshold")
    ret = fn(device, byref(c_info))
    _nvmlCheckReturn(ret)
    return ret 

_nvmlGpuFabricState_t = c_uint
MXSMLEX_GPU_FABRIC_STATE_NOT_SUPPORTED = 0
MXSMLEX_GPU_FABRIC_STATE_NOT_STARTED   = 1
MXSMLEX_GPU_FABRIC_STATE_IN_PROGRESS   = 2
MXSMLEX_GPU_FABRIC_STATE_COMPLETED     = 3

class c_nvmlGpuFabricInfo_t(_PrintableStructure):
    _fields_ = [
        ("clusterUuid", c_char * MXSMLEX_DEVICE_UUID_BUFFER_SIZE),
        ("status", _nvmlReturn_t),
        ("cliqueId", c_uint32),
        ("state", _nvmlGpuFabricState_t)
    ]

nvmlGpuFabricInfo_v2 = 0x02000024

class c_nvmlGpuFabricInfoV_t(_PrintableStructure):
    _fields_ = [
        ("version", c_uint),
        ("clusterUuid", c_char * MXSMLEX_DEVICE_UUID_BUFFER_SIZE),
        ("status", _nvmlReturn_t),
        ("cliqueId", c_uint32),
        ("state", _nvmlGpuFabricState_t),
        ("healthMask", c_uint32)
    ]

    def __init__(self):
        super(c_nvmlGpuFabricInfoV_t, self).__init__(version=nvmlGpuFabricInfo_v2)

def nvmlDeviceGetGpuFabricInfo(device, gpuFabricInfo):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGpuFabricInfo");
    ret = fn(device, gpuFabricInfo)
    _nvmlCheckReturn(ret)
    return ret

def nvmlDeviceGetGpuFabricInfoV(device, gpuFabricInfo):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetGpuFabricInfoV");
    ret = fn(device, gpuFabricInfo)
    _nvmlCheckReturn(ret)
    return ret

######################
## Enums/defines
#### MXSMLEX GPU NVLINK BW MODE
MXSMLEX_GPU_NVLINK_BW_MODE_FULL      = 0x0
MXSMLEX_GPU_NVLINK_BW_MODE_OFF       = 0x1
MXSMLEX_GPU_NVLINK_BW_MODE_MIN       = 0x2
MXSMLEX_GPU_NVLINK_BW_MODE_HALF      = 0x3
MXSMLEX_GPU_NVLINK_BW_MODE_3QUARTER  = 0x4
MXSMLEX_GPU_NVLINK_BW_MODE_COUNT     = 0x5

def nvmlSystemSetNvlinkBwMode(mode):
    fn = _nvmlGetFunctionPointer("mxSmlExSystemSetNvlinkBwMode")
    ret = fn(mode)
    _nvmlCheckReturn(ret)
    return ret

def nvmlSystemGetNvlinkBwMode():
    mode = c_uint()
    fn = _nvmlGetFunctionPointer("mxSmlExSystemGetNvlinkBwMode")
    ret = fn(byref(mode))
    _nvmlCheckReturn(ret)
    return mode.value

_nvmlPowerScopeType_t = c_uint
MXSMLEX_POWER_SCOPE_GPU     = 0
MXSMLEX_POWER_SCOPE_MODULE  = 1
MXSMLEX_POWER_SCOPE_MEMORY  = 2

class c_nvmlPowerValue_v2_t(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('powerScope', _nvmlPowerScopeType_t),
        ('powerValueMw', c_uint),
    ]
    _fmt_ = {'<default>': "%d B"}

nvmlPowerValue_v2 = 0x0200000C

def nvmlDeviceSetPowerManagementLimit_v2(device, powerScope, powerLimit, version=nvmlPowerValue_v2):
    c_powerScope = _nvmlPowerScopeType_t(powerScope)
    c_powerValue = c_nvmlPowerValue_v2_t()
    c_powerValue.version = c_uint(version)
    c_powerValue.powerScope = c_powerScope
    c_powerValue.powerValueMw = c_uint(powerLimit)
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceSetPowerManagementLimit_v2")
    ret = fn(device, byref(c_powerValue))
    return ret

class c_nvmlEccSramErrorStatus_v1_t(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('aggregateUncParity', c_ulonglong),
        ('aggregateUncSecDed', c_ulonglong),
        ('aggregateCor', c_ulonglong),
        ('volatileUncParity', c_ulonglong),
        ('volatileUncSecDed', c_ulonglong),
        ('volatileCor', c_ulonglong),
        ('aggregateUncBucketL2', c_ulonglong),
        ('aggregateUncBucketSm', c_ulonglong),
        ('aggregateUncBucketPcie', c_ulonglong),
        ('aggregateUncBucketMcu', c_ulonglong),
        ('aggregateUncBucketOther', c_ulonglong),
        ('bThresholdExceeded', c_uint)
    ]

    def __init__(self):
        super(c_nvmlEccSramErrorStatus_v1_t, self).__init__(version=nvmlEccSramErrorStatus_v1)

nvmlEccSramErrorStatus_v1 = 0x1000068

def nvmlDeviceGetSramEccErrorStatus(device, status):
    fn = _nvmlGetFunctionPointer("mxSmlExDeviceGetSramEccErrorStatus")
    ret = fn(device, status)
    _nvmlCheckReturn(ret)
    return ret

