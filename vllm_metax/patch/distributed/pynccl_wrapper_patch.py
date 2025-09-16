# SPDX-License-Identifier: Apache-2.0

import ctypes
import platform
from typing import Any, Dict, Optional

import vllm
from vllm.distributed.device_communicators.pynccl_wrapper import (
    Function, NCCLLibrary, buffer_type, cudaStream_t, logger, ncclComm_t,
    ncclDataType_t, ncclDataTypeEnum, ncclRedOp_t, ncclResult_t, ncclUniqueId)
from vllm.logger import init_logger

from vllm_metax.utils import find_mccl_library

logger = init_logger(__name__)


class NCCLLibrary:
    exported_functions = [
        # const char* ncclGetErrorString(ncclResult_t result)
        Function("mcclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
        # ncclResult_t  ncclGetVersion(int *version);
        Function("mcclGetVersion", ncclResult_t,
                 [ctypes.POINTER(ctypes.c_int)]),
        # ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
        Function("mcclGetUniqueId", ncclResult_t,
                 [ctypes.POINTER(ncclUniqueId)]),
        # ncclResult_t  ncclCommInitRank(
        #   ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
        # note that ncclComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function("mcclCommInitRank", ncclResult_t, [
            ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId,
            ctypes.c_int
        ]),
        # ncclResult_t  ncclAllReduce(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        #   cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function("mcclAllReduce", ncclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t,
            ncclRedOp_t, ncclComm_t, cudaStream_t
        ]),

        # ncclResult_t  ncclReduce(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, int root,
        #   ncclComm_t comm,  cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function("mcclReduce", ncclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t,
            ncclRedOp_t, ctypes.c_int, ncclComm_t, cudaStream_t
        ]),

        # ncclResult_t  ncclAllGather(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclComm_t comm,
        #   cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function("mcclAllGather", ncclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t,
            ncclComm_t, cudaStream_t
        ]),

        # ncclResult_t  ncclReduceScatter(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        #   cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function("mcclReduceScatter", ncclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t,
            ncclRedOp_t, ncclComm_t, cudaStream_t
        ]),

        # ncclResult_t  ncclSend(
        #   const void* sendbuff, size_t count, ncclDataType_t datatype,
        #   int dest, ncclComm_t comm, cudaStream_t stream);
        Function("mcclSend", ncclResult_t, [
            buffer_type, ctypes.c_size_t, ncclDataType_t, ctypes.c_int,
            ncclComm_t, cudaStream_t
        ]),

        # ncclResult_t  ncclRecv(
        #   void* recvbuff, size_t count, ncclDataType_t datatype,
        #   int src, ncclComm_t comm, cudaStream_t stream);
        Function("mcclRecv", ncclResult_t, [
            buffer_type, ctypes.c_size_t, ncclDataType_t, ctypes.c_int,
            ncclComm_t, cudaStream_t
        ]),

        # ncclResult_t ncclBroadcast(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, int root, ncclComm_t comm,
        #   cudaStream_t stream);
        Function("mcclBroadcast", ncclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t,
            ctypes.c_int, ncclComm_t, cudaStream_t
        ]),

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
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):

        so_file = so_file or find_mccl_library()

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
                " to point to the correct nccl library path.", so_file,
                platform.platform())
            raise e

        if so_file not in NCCLLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            for func in NCCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            NCCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = NCCLLibrary.path_to_dict_mapping[so_file]

    def ncclGetErrorString(self, result: ncclResult_t) -> str:
        return self._funcs["mcclGetErrorString"](result).decode("utf-8")

    def NCCL_CHECK(self, result: ncclResult_t) -> None:
        if result != 0:
            error_str = self.ncclGetErrorString(result)
            raise RuntimeError(f"MCCL error: {error_str}")

    def ncclGetVersion(self) -> str:
        version = ctypes.c_int()
        self.NCCL_CHECK(self._funcs["mcclGetVersion"](ctypes.byref(version)))
        version_str = str(version.value)
        # something like 21903 --> "2.19.3"
        major = version_str[0].lstrip("0")
        minor = version_str[1:3].lstrip("0")
        patch = version_str[3:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def ncclGetUniqueId(self) -> ncclUniqueId:
        unique_id = ncclUniqueId()
        self.NCCL_CHECK(self._funcs["mcclGetUniqueId"](
            ctypes.byref(unique_id)))
        return unique_id

    def unique_id_from_bytes(self, data: bytes) -> ncclUniqueId:
        if len(data) != 128:
            raise ValueError(
                f"Expected 128 bytes for mcclUniqueId, got {len(data)} bytes")
        unique_id = ncclUniqueId()
        ctypes.memmove(ctypes.addressof(unique_id.internal), data, 128)
        return unique_id

    def ncclCommInitRank(self, world_size: int, unique_id: ncclUniqueId,
                         rank: int) -> ncclComm_t:
        comm = ncclComm_t()
        self.NCCL_CHECK(self._funcs["mcclCommInitRank"](ctypes.byref(comm),
                                                        world_size, unique_id,
                                                        rank))
        return comm

    def ncclAllReduce(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, op: int, comm: ncclComm_t,
                      stream: cudaStream_t) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # and `op` should be `ncclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(self._funcs["mcclAllReduce"](sendbuff, recvbuff, count,
                                                     datatype, op, comm,
                                                     stream))
    
    def ncclReduce(self, sendbuff: buffer_type, recvbuff: buffer_type,
                   count: int, datatype: int, op: int, root: int,
                   comm: ncclComm_t, stream: cudaStream_t) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # and `op` should be `ncclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(self._funcs["mcclReduce"](sendbuff, recvbuff, count,
                                                  datatype, op, root, comm,
                                                  stream))
    
    def ncclReduceScatter(self, sendbuff: buffer_type, recvbuff: buffer_type,
                          count: int, datatype: int, op: int, comm: ncclComm_t,
                          stream: cudaStream_t) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # and `op` should be `ncclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(self._funcs["mcclReduceScatter"](sendbuff, recvbuff,
                                                         count, datatype, op,
                                                         comm, stream))

    def ncclAllGather(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, comm: ncclComm_t,
                      stream: cudaStream_t) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # which is an aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(self._funcs["mcclAllGather"](sendbuff, recvbuff, count,
                                                     datatype, comm, stream))

    def ncclSend(self, sendbuff: buffer_type, count: int, datatype: int,
                 dest: int, comm: ncclComm_t, stream: cudaStream_t) -> None:
        self.NCCL_CHECK(self._funcs["mcclSend"](sendbuff, count, datatype,
                                                dest, comm, stream))

    def ncclRecv(self, recvbuff: buffer_type, count: int, datatype: int,
                 src: int, comm: ncclComm_t, stream: cudaStream_t) -> None:
        self.NCCL_CHECK(self._funcs["mcclRecv"](recvbuff, count, datatype, src,
                                                comm, stream))

    def ncclBroadcast(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, root: int, comm: ncclComm_t,
                      stream: cudaStream_t) -> None:
        self.NCCL_CHECK(self._funcs["mcclBroadcast"](sendbuff, recvbuff, count,
                                                     datatype, root, comm,
                                                     stream))

    def ncclCommDestroy(self, comm: ncclComm_t) -> None:
        self.NCCL_CHECK(self._funcs["mcclCommDestroy"](comm))

    def ncclGroupStart(self) -> None:
        self.NCCL_CHECK(self._funcs["mcclGroupStart"]())

    def ncclGroupEnd(self) -> None:
        self.NCCL_CHECK(self._funcs["mcclGroupEnd"]())


vllm.distributed.device_communicators.pynccl_wrapper.NCCLLibrary = NCCLLibrary
