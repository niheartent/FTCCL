/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(Reduce, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduce};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32, ncclInt8, ncclUint32, ncclInt64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum, ncclAvg, ncclMax, ncclProd};
    std::vector<int>            const numElements     = {393216, 384, 1024, 1048576};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const numElements     = {393216, 384};
#endif
    std::vector<int>            const roots           = {0};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Reduce, OutOfPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduce};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat16, ncclFloat64, ncclInt8, ncclUint32, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMin, ncclAvg, ncclSum, ncclProd};
    std::vector<int>            const numElements     = {393216, 1025, 4099};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat16, ncclFloat64, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMin};
    std::vector<int>            const numElements     = {393216};
#endif
    std::vector<int>            const roots           = {0};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Reduce, InPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduce};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32, ncclInt8, ncclBfloat16, ncclFloat64, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclProd, ncclAvg, ncclSum, ncclProd};
    std::vector<int>            const numElements     = {384, 1023, 393216, 2049};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32, ncclInt8, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclProd};
    std::vector<int>            const numElements     = {384};
#endif
    std::vector<int>            const roots           = {1};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Reduce, InPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduce};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclBfloat16, ncclInt32, ncclUint8, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMax, ncclAvg, ncclSum, ncclProd};
    std::vector<int>            const numElements     = {393216, 1047, 479512};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclBfloat16, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMax};
    std::vector<int>            const numElements     = {393216};
#endif
    std::vector<int>            const roots           = {0};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Reduce, ManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduce};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint64, ncclFloat16, ncclUint8, ncclFloat32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum, ncclAvg, ncclProd};
    std::vector<int>            const numElements     = {3524082, 2500, 1047, 4068};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const numElements     = {3524082, 2500};
#endif
    std::vector<int>            const roots           = {0};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Reduce, ManagedMemGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat64, ncclBfloat16};
#ifdef FULL_STRESS_TEST
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum, ncclProd, ncclAvg};
    std::vector<int>            const numElements     = {4314, 1048576, 8191};
#else
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const numElements     = {4314};
#endif
    std::vector<int>            const roots           = {0};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }
}
