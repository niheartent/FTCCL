/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
  // Test various collectives using with non-blocking comms
  TEST(NonBlocking, SingleCalls)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t> const  funcTypes = {ncclCollBroadcast,
                                                ncclCollReduce,
                                                ncclCollAllGather,
                                                ncclCollReduceScatter,
                                                ncclCollAllReduce,
                                                ncclCollGather,
                                                ncclCollScatter};
    // int        const  numElements   = 1048576;
    bool       const  inPlace       = false;
    bool       const  useManagedMem = false;
    bool       const  useBlocking   = false;
    
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32, ncclBfloat16, ncclInt8, ncclInt32};
    std::vector<int>            const numElements     = {1048576, 500, 1023, 125};
#else
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32};
    std::vector<int>            const numElements     = {1048576};
#endif

    OptionalColArgs options;
    options.redOp = ncclSum;

    bool isCorrect = true;
    std::vector<int> sortedN = numElements;
    std::sort(sortedN.rbegin(), sortedN.rend());
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
        
      if (testing::Test::HasFailure())
      {
        isCorrect = false;
        continue;
      }

      for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
      for (int dtIdx = 0; dtIdx < dataTypes.size() && isCorrect; ++dtIdx){

      int const numProcesses = isMultiProcess ? totalRanks : 1;
      // Initialize communicators in non-blocking mode
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder), 1, 1, 1, useBlocking);

      // Loop over various collective functions
      for (auto funcType : funcTypes)
      {
        if (testBed.ev.showNames)
          INFO("%s %d-ranks Non-Blocking %s\n",
               isMultiProcess ? "MP" : "SP", totalRanks, ncclFuncNames[funcType]);

        int numInputElements;
        int numOutputElements;
        CollectiveArgs::GetNumElementsForFuncType(funcType,
                                                  sortedN[numIdx],
                                                  totalRanks,
                                                  &numInputElements,
                                                  &numOutputElements);

        testBed.SetCollectiveArgs(funcType,
                                  dataTypes[dtIdx],
                                  numInputElements,
                                  numOutputElements,
                                  options);

        testBed.AllocateMem(inPlace, useManagedMem);
        testBed.PrepareData();
        testBed.ExecuteCollectives();
        if (testing::Test::HasFailure())
        {
          isCorrect = false;
          continue;
        }
        testBed.ValidateResults(isCorrect);
        if (testing::Test::HasFailure())
        {
          isCorrect = false;
          continue;
        }
      testBed.DeallocateMem();
      }
      testBed.DestroyComms();
    }
    }

    testBed.Finalize();
  }
}
