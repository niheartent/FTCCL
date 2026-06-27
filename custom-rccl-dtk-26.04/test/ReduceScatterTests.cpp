/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(ReduceScatter, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32, ncclBfloat16, ncclInt};
    std::vector<int>            const numElements     = {393216, 384, 4096, 1025};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMax, ncclSum};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32};
    std::vector<int>            const numElements     = {393216, 384};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMax};
#endif
    std::vector<int>            const roots           = {0};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(ReduceScatter, OutOfPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat64, ncclBfloat16, ncclInt32, ncclUint8, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<int>            const numElements     = {1048576, 1248, 8427, 1643};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMax, ncclProd};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat64, ncclBfloat16, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<int>            const numElements     = {1048576};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMax};
#endif
    std::vector<int>            const roots           = {0};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(ReduceScatter, InPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32, ncclBfloat16, ncclFloat16, ncclInt64};
    std::vector<int>            const numElements     = {542357, 471, 1024, 40961};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32};
    std::vector<int>            const numElements     = {542357};
#endif
    std::vector<ncclRedOp_t>    const redOps          = {ncclProd};
    std::vector<int>            const roots           = {0, 1};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(ReduceScatter, InPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8, ncclFloat16, ncclFloat, ncclInt32};
    std::vector<int>            const numElements     = {246, 1023, 4097, 1048576};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8, ncclFloat16};
    std::vector<int>            const numElements     = {246};;
#endif
    std::vector<ncclRedOp_t>    const redOps          = {ncclMin};
    std::vector<int>            const roots           = {0};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(ReduceScatter, ManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt64, ncclUint8, ncclFloat16, ncclInt32,ncclFloat64};
    std::vector<int>            const numElements     = {1024, 4095, 127, 38547};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt64, ncclUint8};
    std::vector<int>            const numElements     = {1024};
#endif
    std::vector<ncclRedOp_t>    const redOps          = {ncclAvg};
    std::vector<int>            const roots           = {0};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(ReduceScatter, ManagedMemGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduceScatter};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint32, ncclUint64, ncclInt8, ncclFloat64, ncclBfloat16};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint32, ncclUint64};
#endif
    std::vector<ncclRedOp_t>    const redOps          = {ncclAvg};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {6485423};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }
}
