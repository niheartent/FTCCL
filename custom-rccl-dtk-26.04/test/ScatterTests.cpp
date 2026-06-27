/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  TEST(Scatter, OutOfPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollScatter};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32, ncclInt8, ncclBfloat16, ncclFloat64};
    std::vector<int>            const numElements     = {393216, 384, 1024, 4099};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32};
    std::vector<int>            const numElements     = {393216, 384};
#endif
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {1};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Scatter, OutOfPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollScatter};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat64, ncclBfloat16, ncclInt32, ncclUint8, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<int>            const numElements     = {24658, 2044, 1048577};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat64, ncclBfloat16, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<int>            const numElements     = {24658};
#endif
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {1};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Scatter, InPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollScatter};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32, ncclFloat16, ncclFloat64, ncclUint8};
    std::vector<int>            const numElements     = {1048576, 1024, 129, 4097};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32};
    std::vector<int>            const numElements     = {1048576, 1024};
#endif
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Scatter, InPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollScatter};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt8, ncclFloat16, ncclFloat64, ncclInt32};
    std::vector<int>            const numElements     = {356, 1048576, 4098};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt8, ncclFloat16};
    std::vector<int>            const numElements     = {356};
#endif
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(Scatter, ManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollScatter};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt64, ncclUint8, ncclBfloat16, ncclFloat};
    std::vector<int>            const numElements     = {948576, 1048576, 4098, 1023};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt64, ncclUint8};
    std::vector<int>            const numElements     = {948576};
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

  TEST(Scatter, ManagedMemGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollScatter};
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint32, ncclUint64, ncclInt32, ncclBfloat16};
    std::vector<int>            const numElements     = {125, 2047, 8192, 1048576};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint32, ncclUint64};
    std::vector<int>            const numElements     = {125};
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
}
