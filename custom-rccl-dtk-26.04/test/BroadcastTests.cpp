/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(Broadcast, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollBroadcast};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat16, ncclFloat32, ncclInt8, ncclUint64, ncclInt32};
    std::vector<int>            const numElements     = {1048576, 500, 511};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat16, ncclFloat32};
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

  TEST(Broadcast, OutOfPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollBroadcast};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclBfloat16, ncclFloat64, ncclInt32, ncclUint8, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<int>            const numElements     = {586, 1024};
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

  TEST(Broadcast, InPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollBroadcast};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32, ncclFloat16, ncclFloat64};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32};
#endif
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {1};
    std::vector<int>            const numElements     = {104857, 264};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Broadcast, InPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollBroadcast};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt8, ncclInt64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {1};
    std::vector<int>            const numElements     = {958};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Broadcast, ManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollBroadcast};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8, ncclBfloat16, ncclFloat32};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8};
#endif
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1039203, 2500};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Broadcast, ManagedMemGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollBroadcast};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint32, ncclUint64, ncclInt8, ncclFloat32};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint32, ncclUint64};
#endif
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {896};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }
}
