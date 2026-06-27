/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"
#include "CallCollectiveForked.hpp"

namespace RcclUnitTesting
{
  TEST(AllGather, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32, ncclBfloat16, ncclInt8, ncclInt32};
    std::vector<int>            const numElements     = {1048576, 500, 1023, 125};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32};
    std::vector<int>            const numElements     = {1048576, 500};
#endif
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, OutOfPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclBfloat16, ncclFloat64, ncclUint8, ncclUint32, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<int>            const numElements     = {586, 1024, 1048576};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclBfloat16, ncclFloat64, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<int>            const numElements     = {586};
#endif
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, InPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {104857, 264};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, InPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt8, ncclInt64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {958};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, ManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8, ncclFloat16, ncclFloat64};
    std::vector<int>            const numElements     = {1039203, 2500, 1025};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8};
    std::vector<int>            const numElements     = {1039203, 2500};
#endif
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, ManagedMemGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllGather};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint32, ncclUint64, ncclFloat32, ncclFloat64};
    std::vector<int>            const numElements     = {1048575, 896, 163};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint32, ncclUint64};
    std::vector<int>            const numElements     = {896};
#endif
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllGather, UserBufferRegistration)
  {          
    const int nranks = 8;
    size_t count = 2048;
    std::vector<int> sendBuff(count, 0);
    std::vector<int> recvBuff(nranks*count, 0);
    std::vector<int> expected(nranks*count, 0);

    for (int i = 0; i < count; ++i){
        sendBuff[i] = i;
    }

    for(int r = 0; r < nranks; ++r)
      for (int i = 0; i < count; ++i)
        expected[r*count + i] = sendBuff[i];

    callCollectiveForked(nranks, ncclCollAllGather, sendBuff, recvBuff, expected);
  }

  TEST(AllGather, ManagedMemUserBufferRegistration)
  {          
    const int nranks = 8;
    size_t count = 2048;
    std::vector<int> sendBuff(count, 0);
    std::vector<int> recvBuff(nranks*count, 0);
    std::vector<int> expected(nranks*count, 0);
    const bool use_managed_mem = true;
    for (int i = 0; i < count; ++i){
        sendBuff[i] = i;
    }

    for(int r = 0; r < nranks; ++r)
      for (int i = 0; i < count; ++i)
        expected[r*count + i] = sendBuff[i];

    callCollectiveForked(nranks, ncclCollAllGather, sendBuff, recvBuff, expected, use_managed_mem);
  }
}
