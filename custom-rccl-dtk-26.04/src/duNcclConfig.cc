/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

 #include "enqueue.h"
 #include "nccl.h"
 #include "graph/topo.h"
 
 NCCL_API(ncclResult_t, duNcclAllreduce, const void* sendbuff, void* recvbuff, size_t count,
     ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, hipStream_t stream,
     const char* proto, const char* algo, int nChannels);
 ncclResult_t duNcclAllreduce(const void* sendbuff, void* recvbuff, size_t count,
     ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, hipStream_t stream,
     const char* proto, const char* algo, int nChannels) {
   struct NvtxParamsAllReduce {
     size_t bytes;
     ncclRedOp_t op;
   };
   // Just pass the size of one message and not the total bytes sent/received.
   static constexpr nvtxPayloadSchemaEntry_t AllReduceSchema[] = {
     {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
     {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
       offsetof(NvtxParamsAllReduce, op)}
   };
   NvtxParamsAllReduce payload{count * ncclTypeSize(datatype), op};
   NVTX3_FUNC_WITH_PARAMS(AllReduce, AllReduceSchema, payload)
 
   struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
     sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
     ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS, proto, algo, nChannels};
   
   return ncclEnqueueCheck(&info);
 }

NCCL_API(ncclResult_t, duNcclAllgather, const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream,
  const char* proto, const char* algo, int nChannels);
ncclResult_t duNcclAllgather(const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream,
  const char* proto, const char* algo, int nChannels) {
  
    // Just pass the size of one message and not the total bytes sent/received.
  constexpr nvtxPayloadSchemaEntry_t AllGatherSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"}
  };
  size_t msgsize = sendcount * ncclTypeSize(datatype);
  NVTX3_FUNC_WITH_PARAMS(AllGather, AllGatherSchema, msgsize)

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS , proto, algo, nChannels};

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, duNcclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
  ncclComm_t comm, cudaStream_t stream, const char* proto, const char* algo, int nChannels);
ncclResult_t duNcclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
  ncclComm_t comm, cudaStream_t stream, const char* proto, const char* algo, int nChannels) {

  struct NvtxParamsBroadcast {
    size_t bytes;
    int root;
  };
  constexpr nvtxPayloadSchemaEntry_t BroadcastSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Root", nullptr, 0, offsetof(NvtxParamsBroadcast, root)}
  };
  NvtxParamsBroadcast payload{count * ncclTypeSize(datatype), root};
  NVTX3_FUNC_WITH_PARAMS(Broadcast, BroadcastSchema, payload)

  struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS, proto, algo, nChannels};

  return ncclEnqueueCheck(&info);
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, duNcclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
  ncclComm_t comm, cudaStream_t stream, const char* proto, const char* algo, int nChannels);
ncclResult_t duNcclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
  ncclComm_t comm, cudaStream_t stream, const char* proto, const char* algo, int nChannels) {

  return duNcclBroadcast(buff, buff, count, datatype, root, comm, stream, proto, algo, nChannels);
}

NCCL_API(ncclResult_t, duNcclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream,
  const char* proto, const char* algo, int nChannels);
ncclResult_t duNcclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream,
  const char* proto, const char* algo, int nChannels) {

  struct NvtxParamsReduceScatter {
    size_t bytes;
    ncclRedOp_t op;
  };
  constexpr nvtxPayloadSchemaEntry_t ReduceScatterSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsReduceScatter, op)}
  };
  NvtxParamsReduceScatter payload{recvcount * ncclTypeSize(datatype), op};
  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, ReduceScatterSchema, payload)

  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
    sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
    REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS, proto, algo, nChannels};

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, duNcclReduce, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream,
  const char* proto, const char* algo, int nChannels);
ncclResult_t duNcclReduce(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream,
  const char* proto, const char* algo, int nChannels) {

  struct NvtxParamsReduce {
    size_t bytes;
    int root;
    ncclRedOp_t op;
  };
  constexpr nvtxPayloadSchemaEntry_t ReduceSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Root", nullptr, 0, offsetof(NvtxParamsReduce, root)},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsReduce, op)}
  };
  NvtxParamsReduce payload{count * ncclTypeSize(datatype), root, op};
  NVTX3_FUNC_WITH_PARAMS(Reduce, ReduceSchema, payload)

  struct ncclInfo info = { ncclFuncReduce, "Reduce",
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS, proto, algo, nChannels};

  return ncclEnqueueCheck(&info);
}


struct NvtxParamsSendRecv {
  size_t bytes;
  int peer;
};
constexpr const nvtxPayloadSchemaEntry_t SendRecvSchema[] = {
  {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
  {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Peer rank", nullptr, 0, offsetof(NvtxParamsSendRecv, peer)}
};

NCCL_API(ncclResult_t, duNcclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
  ncclComm_t comm, cudaStream_t stream,const char* proto,int nChannels);
ncclResult_t duNcclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
  ncclComm_t comm, cudaStream_t stream,const char* proto,int nChannels) {
  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Send, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1, proto, NULL, nChannels};
 
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(ncclGroupEnd());
  return ret;
}

NCCL_API(ncclResult_t, duNcclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
  ncclComm_t comm, cudaStream_t stream, const char* proto, int nChannels);
ncclResult_t duNcclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
  ncclComm_t comm, cudaStream_t stream, const char* proto, int nChannels) {
  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Recv, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 , proto, NULL, nChannels};

  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(ncclGroupEnd());
  return ret;
}

NCCL_API(ncclResult_t, duNcclGather, const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream, const char* proto, int nChannels);
ncclResult_t duNcclGather(const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream, const char* proto, int nChannels) {

  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  size_t rankOffset = sendcount * ncclTypeSize(datatype);
  if (sendcount == 0) return ncclSuccess;
  int rank;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  NCCLCHECK(ncclGroupStart());
  if (rank == root) {
    for (int r=0; r<nRanks; r++) {
      NCCLCHECK(duNcclRecv(((char*)recvbuff)+r*rankOffset, sendcount, datatype, r, comm, stream, proto, nChannels));
    }
  }
  NCCLCHECK(duNcclSend(sendbuff, sendcount, datatype, root, comm, stream, proto, nChannels));
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}

NCCL_API(ncclResult_t, duNcclScatter, const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, int root,
  ncclComm_t comm, hipStream_t stream, const char* proto, int nChannels);
ncclResult_t duNcclScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, int root,
  ncclComm_t comm, hipStream_t stream, const char* proto, int nChannels) {
  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  size_t rankOffset = recvcount * ncclTypeSize(datatype);
  if (recvcount == 0) return ncclSuccess;
  int rank;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  NCCLCHECK(ncclGroupStart());
  if (rank == root) {
    for (int r=0; r<nRanks; r++) {
      NCCLCHECK(duNcclSend(((char*)sendbuff)+r*rankOffset, recvcount, datatype, r, comm, stream, proto, nChannels));
    }
  }

  NCCLCHECK(duNcclRecv(recvbuff, recvcount, datatype, root, comm, stream, proto, nChannels));
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}

NCCL_API(ncclResult_t, duNcclAllToAllv, const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
  void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
  ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream, const char* proto, int nChannels);
ncclResult_t duNcclAllToAllv(const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
  void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
  ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream, const char* proto, int nChannels) {

  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclGroupStart());
  for (int r=0; r<nRanks; r++) {
    if (sendcounts[r]) NCCLCHECK(duNcclSend(
        ((char*)sendbuff) + sdispls[r]*ncclTypeSize(datatype),
        sendcounts[r],
        datatype,
        r,
        comm,
        stream,
        proto, 
        nChannels));
    if (recvcounts[r]) NCCLCHECK(duNcclRecv(
        ((char*)recvbuff) + rdispls[r]*ncclTypeSize(datatype),
        recvcounts[r],
        datatype,
        r,
        comm,
        stream,
        proto, 
        nChannels));
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}

NCCL_API(ncclResult_t, duNcclAllToAll, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, hipStream_t stream, const char* proto, int nChannels);
ncclResult_t duNcclAllToAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, hipStream_t stream, const char* proto, int nChannels) {

  size_t rankOffset = count * ncclTypeSize(datatype);
  size_t rankAlign = rankOffset & ((~rankOffset) + 1);
  // Determine Pivot A2A support now that we know number of channels
  if (comm->topo->pivotA2AEnabled && comm->nChannels >= comm->topo->pivotA2ANumBiRings * 2 &&
      rankOffset >= 744 * 1024 && rankAlign != 4) {
    struct ncclInfo info = { ncclFuncAllToAllPivot, "AllToAllPivot",
      sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
      ALLTOALL_PIVOT_CHUNKSTEPS, ALLTOALL_PIVOT_SLICESTEPS };
    return ncclEnqueueCheck(&info);
  } else {
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    if (count == 0) return ncclSuccess;
    NCCLCHECK(ncclGroupStart());
    for (int r=0; r<nRanks; r++) {
      NCCLCHECK(duNcclSend(((char*)sendbuff)+r*rankOffset, count, datatype, r, comm, stream, proto, nChannels));
      NCCLCHECK(duNcclRecv(((char*)recvbuff)+r*rankOffset, count, datatype, r, comm, stream, proto, nChannels));
    }
    NCCLCHECK(ncclGroupEnd());
    return ncclSuccess;
  }
}
