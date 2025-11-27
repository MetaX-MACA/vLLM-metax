# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# This file is a pure Python wrapper for the NCCL library.
# The main purpose is to use NCCL combined with CUDA graph.
# Before writing this script, we tried the following approach:
# 1. We tried to use `cupy`, it calls NCCL correctly, but `cupy` itself
#  often gets stuck when initializing the NCCL communicator.
# 2. We tried to use `torch.distributed`, but `torch.distributed.all_reduce`
#  contains many other potential cuda APIs, that are not allowed during
#  capturing the CUDA graph. For further details, please check
# https://discuss.pytorch.org/t/pytorch-cudagraph-with-nccl-operation-failed/ .
#
# Another rejected idea is to write a C/C++ binding for NCCL. It is usually
# doable, but we often encounter issues related with nccl versions, and need
# to switch between different versions of NCCL. See
# https://github.com/NVIDIA/nccl/issues/1234 for more details.
# A C/C++ binding is not flexible enough to handle this. It requires
# recompilation of the code every time we want to switch between different
# versions. This current implementation, with a **pure** Python wrapper, is
# more flexible. We can easily switch between different versions of NCCL by
# changing the environment variable `VLLM_NCCL_SO_PATH`, or the `so_file`
# variable in the code.


# -------------------------------------------------------
# Note: This patch is for compatibility on Metax platform,
#       replaced some libraries' interface with maca's
# -------------------------------------------------------

import ctypes
import platform
from typing import Any

import vllm
from vllm.distributed.device_communicators.pynccl_wrapper import (
    Function,
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    logger,
    ncclComm_t,
    ncclDataType_t,
    ncclRedOp_t,
    ncclResult_t,
    ncclWindow_t,
    ncclUniqueId,
)

from vllm.platforms import current_platform
from vllm_metax.utils.mccl import find_mccl_library


class NCCLLibrary:
    exported_functions = [
        # const char* ncclGetErrorString(ncclResult_t result)
        Function("mcclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
        # ncclResult_t  ncclGetVersion(int *version);
        Function("mcclGetVersion", ncclResult_t, [ctypes.POINTER(ctypes.c_int)]),
        # ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
        Function("mcclGetUniqueId", ncclResult_t, [ctypes.POINTER(ncclUniqueId)]),
        # ncclResult_t  ncclCommInitRank(
        #   ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
        # note that ncclComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function(
            "mcclCommInitRank",
            ncclResult_t,
            [ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int],
        ),
        # ncclResult_t  ncclAllReduce(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        #   cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "mcclAllReduce",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclRedOp_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t  ncclReduce(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, int root,
        #   ncclComm_t comm,  cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "mcclReduce",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclRedOp_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t  ncclAllGather(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclComm_t comm,
        #   cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "mcclAllGather",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t  ncclReduceScatter(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        #   cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "mcclReduceScatter",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclRedOp_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t  ncclSend(
        #   const void* sendbuff, size_t count, ncclDataType_t datatype,
        #   int dest, ncclComm_t comm, cudaStream_t stream);
        Function(
            "mcclSend",
            ncclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t  ncclRecv(
        #   void* recvbuff, size_t count, ncclDataType_t datatype,
        #   int src, ncclComm_t comm, cudaStream_t stream);
        Function(
            "mcclRecv",
            ncclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t ncclBroadcast(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, int root, ncclComm_t comm,
        #   cudaStream_t stream);
        Function(
            "mcclBroadcast",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # be cautious! this is a collective call, it will block until all
        # processes in the communicator have called this function.
        # because Python object destruction can happen in random order,
        # it is better not to call it at all.
        # ncclResult_t  ncclCommDestroy(ncclComm_t comm);
        Function("mcclCommDestroy", ncclResult_t, [ncclComm_t]),
        # ncclResult_t ncclGroupStart();
        Function("mcclGroupStart", ncclResult_t, []),
        # ncclResult_t ncclGroupEnd();
        Function("mcclGroupEnd", ncclResult_t, []),
        # # ncclResult_t ncclCommWindowRegister(
        # #   ncclComm_t comm, void* buff, size_t size,
        # #   ncclWindow_t* win, int winFlags);
        # Function(
        #     "mcclCommWindowRegister",
        #     ncclResult_t,
        #     [
        #         ncclComm_t,
        #         buffer_type,
        #         ctypes.c_size_t,
        #         ctypes.POINTER(ncclWindow_t),
        #         ctypes.c_int,
        #     ],
        # ),
        # # ncclResult_t ncclCommWindowDeregister(
        # #   ncclComm_t comm, ncclWindow_t win);
        # Function("mcclCommWindowDeregister", ncclResult_t,
        #          [ncclComm_t, ncclWindow_t]),
    ]
    # \------------------------- Metax Modification -------------------------/

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: str | None = None):
        # /------------------------  Metax Modification -------------------------\
        so_file = so_file or find_mccl_library()
        # \------------------------- Metax Modification -------------------------/
        try:
            if so_file not in NCCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                NCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = NCCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load NCCL library from %s. "
                "It is expected if you are not running on NVIDIA/AMD GPUs."
                "Otherwise, the nccl library might not exist, be corrupted "
                "or it does not support the current platform %s. "
                "If you already have the library, please set the "
                "environment variable VLLM_NCCL_SO_PATH"
                " to point to the correct nccl library path.",
                so_file,
                platform.platform(),
            )
            raise e

        if so_file not in NCCLLibrary.path_to_dict_mapping:
            _funcs: dict[str, Any] = {}
            for func in NCCLLibrary.exported_functions:
                try:
                    f = getattr(self.lib, func.name)
                    f.restype = func.restype
                    f.argtypes = func.argtypes
                    _funcs[func.name] = f
                except AttributeError:
                    if func.name in [
                        "ncclCommWindowRegister",
                        "ncclCommWindowDeregister",
                    ]:
                        if envs.VLLM_USE_NCCL_SYMM_MEM:
                            logger.warning_once(
                                "The symbol %s is not found in the NCCL "
                                "library %s. To enable VLLM_USE_NCCL_SYMM_MEM "
                                " please update your NCCL version to >= "
                                "2.27.03.",
                                func.name,
                                so_file,
                            )
                        if current_platform.is_rocm():
                            # Having an exception here on ROCm platform is
                            # not allowed during graph capturing
                            continue
                    raise
            NCCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = NCCLLibrary.path_to_dict_mapping[so_file]

    def ncclGetErrorString(self, result: ncclResult_t) -> str:
        return self._funcs["mcclGetErrorString"](result).decode("utf-8")

    def NCCL_CHECK(self, result: ncclResult_t) -> None:
        if result != 0:
            error_str = self.ncclGetErrorString(result)
            raise RuntimeError(f"NCCL error: {error_str}")

    def ncclGetRawVersion(self) -> int:
        version = ctypes.c_int()
        self.NCCL_CHECK(self._funcs["mcclGetVersion"](ctypes.byref(version)))
        # something like 21903
        return version.value

    def ncclGetVersion(self) -> str:
        version_str = str(self.ncclGetRawVersion())
        # something like 21903 --> "2.19.3"
        major = version_str[0].lstrip("0")
        minor = version_str[1:3].lstrip("0")
        patch = version_str[3:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def ncclGetUniqueId(self) -> ncclUniqueId:
        unique_id = ncclUniqueId()
        self.NCCL_CHECK(self._funcs["mcclGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def unique_id_from_bytes(self, data: bytes) -> ncclUniqueId:
        if len(data) != 128:
            raise ValueError(
                f"Expected 128 bytes for ncclUniqueId, got {len(data)} bytes"
            )
        unique_id = ncclUniqueId()
        ctypes.memmove(ctypes.addressof(unique_id.internal), data, 128)
        return unique_id

    def ncclCommInitRank(
        self, world_size: int, unique_id: ncclUniqueId, rank: int
    ) -> ncclComm_t:
        comm = ncclComm_t()
        self.NCCL_CHECK(
            self._funcs["mcclCommInitRank"](
                ctypes.byref(comm), world_size, unique_id, rank
            )
        )
        return comm

    def ncclAllReduce(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # and `op` should be `ncclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(
            self._funcs["mcclAllReduce"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def ncclReduce(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        root: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # and `op` should be `ncclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(
            self._funcs["mcclReduce"](
                sendbuff, recvbuff, count, datatype, op, root, comm, stream
            )
        )

    def ncclReduceScatter(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # and `op` should be `ncclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(
            self._funcs["mcclReduceScatter"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def ncclAllGather(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # which is an aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(
            self._funcs["mcclAllGather"](
                sendbuff, recvbuff, count, datatype, comm, stream
            )
        )

    def ncclSend(
        self,
        sendbuff: buffer_type,
        count: int,
        datatype: int,
        dest: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["mcclSend"](sendbuff, count, datatype, dest, comm, stream)
        )

    def ncclRecv(
        self,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        src: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["mcclRecv"](recvbuff, count, datatype, src, comm, stream)
        )

    def ncclBroadcast(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        root: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["mcclBroadcast"](
                sendbuff, recvbuff, count, datatype, root, comm, stream
            )
        )

    def ncclCommDestroy(self, comm: ncclComm_t) -> None:
        self.NCCL_CHECK(self._funcs["mcclCommDestroy"](comm))

    def ncclGroupStart(self) -> None:
        self.NCCL_CHECK(self._funcs["mcclGroupStart"]())

    def ncclGroupEnd(self) -> None:
        self.NCCL_CHECK(self._funcs["mcclGroupEnd"]())

    def ncclCommWindowRegister(
        self, comm: ncclComm_t, buff: buffer_type, size: int, win_flags: int
    ) -> ncclWindow_t:
        window = ncclWindow_t()
        # self.NCCL_CHECK(self._funcs["mcclCommWindowRegister"](
        #     comm, buff, size, ctypes.byref(window), win_flags))
        return window

    def ncclCommWindowDeregister(self, comm: ncclComm_t, window: ncclWindow_t) -> None:
        # self.NCCL_CHECK(self._funcs["mcclCommWindowDeregister"](comm, window))
        return


import vllm.distributed.device_communicators.pynccl_wrapper

vllm.distributed.device_communicators.pynccl_wrapper.NCCLLibrary = NCCLLibrary
