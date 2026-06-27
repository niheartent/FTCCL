/*************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
 #include "TestBed.hpp"
 #include <dlfcn.h>
 
 namespace RcclUnitTesting
 {
  /**
   * Detect InfiniBand hardware using libibverbs API.
   * @return true if InfiniBand devices > 0, false otherwise
  */
  static bool HasInfiniBandSupport()
  {
    void* ibvhandle = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!ibvhandle) ibvhandle = dlopen("libibverbs.so", RTLD_NOW | RTLD_LOCAL);
    if (!ibvhandle) return false;
    typedef struct ibv_device** (*get_device_list_func)(int*);
    typedef void (*free_device_list_func)(struct ibv_device**);
    auto get_devices = (get_device_list_func)dlsym(ibvhandle, "ibv_get_device_list");
    auto free_devices = (free_device_list_func)dlsym(ibvhandle, "ibv_free_device_list");
    if (!get_devices || !free_devices) {
      dlclose(ibvhandle);
      return false;
    }
    int count = 0;
    struct ibv_device** devices = get_devices(&count);
    if (devices) free_devices(devices);
    dlclose(ibvhandle);
    return (count > 0);
  }
 
  TEST(SendRecv, SinglePairs)
  {
    TestBed testBed;

    // Configuration
#ifdef FULL_STRESS_TEST
    std::vector<ncclDataType_t> const& dataTypes       = {ncclInt32, ncclUint8, ncclBfloat16, ncclFloat64};
    std::vector<int>            const  numElements     = {104857600, 1048576, 53327, 1024, 511};
#else
    std::vector<ncclDataType_t> const& dataTypes       = {ncclInt32, ncclFloat16, ncclFloat64};
    std::vector<int>            const  numElements     = {1048576, 53327, 1024};
#endif
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;

    OptionalColArgs options;
    bool isCorrect = true;
    int numGpus = testBed.ev.maxGpus;
    for (int rpg=0; rpg < 2 && isCorrect; ++rpg)
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;
      int ranksPerGpu = rpg == 0 ? 1 : testBed.ev.maxRanksPerGpu;
      int totalRanks = numGpus * ranksPerGpu;
      int const numProcesses = isMultiProcess ? numGpus : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, numGpus, ranksPerGpu, gpuPriorityOrder),
                        {1,2}, //two group, second group sendrecv to self, has 2 coll
                        testBed.GetNumStreamsPerGroup(1,2),
                        2);

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
      for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
      {
        for (int recvRank = 0; recvRank  < totalRanks; ++recvRank)
        {
          options.root = recvRank;
          int groupCallId = sendRank == recvRank; //self sendrecv group has two coll
          int recvId      = sendRank == recvRank; //where recv will be second coll
          testBed.SetCollectiveArgs(ncclCollSend,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    0,
                                    groupCallId,
                                    sendRank);
          if (recvRank == 0)
          {
            //set up the collArg slot to make sure AllocateMem is called once and correctly
            testBed.SetCollectiveArgs(ncclCollSend,
                                      dataTypes[dataIdx],
                                      numElements[numIdx],
                                      numElements[numIdx],
                                      options,
                                      0,
                                      !groupCallId,
                                      sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 0, 0, sendRank);
            testBed.PrepareData(0, 0, sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 1, 0, sendRank);
            testBed.PrepareData(1, 0, sendRank);
          }

          if (testBed.ev.showNames) // Show test names
            INFO("%s Datatype: %s SendReceive test Rank %d -> Rank %d for %d Elements\n",
                 isMultiProcess ? "MP" : "SP",
                 ncclDataTypeNames[dataTypes[dataIdx]],
                 sendRank,
                 recvRank,
                 numElements[numIdx]);
          options.root = sendRank;

          testBed.SetCollectiveArgs(ncclCollRecv,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    recvId,
                                    groupCallId,
                                    recvRank);
          testBed.AllocateMem(inPlace, useManagedMem, groupCallId, recvId, recvRank);
          testBed.PrepareData(groupCallId, recvId, recvRank);
          testBed.ExecuteCollectives({sendRank, recvRank}, groupCallId);
          testBed.ValidateResults(isCorrect, groupCallId, recvId, recvRank);
          testBed.DeallocateMem(groupCallId, recvId, recvRank);
        }
        testBed.DeallocateMem(0, 0, sendRank);
        testBed.DeallocateMem(1, 0, sendRank);
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  TEST(SendRecv, UserBufferRegister)
  {
    bool enableIntranet = HasInfiniBandSupport();
    if (enableIntranet) setenv("RCCL_ENABLE_INTRANET", "1", 1);
    TestBed testBed;

    // Configuration
    std::vector<ncclDataType_t> const& dataTypes       = {ncclInt32, ncclFloat16, ncclFloat64};
    std::vector<int>            const  numElements     = {1048576, 53327, 1024};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;
    bool                        const  userRegistered  = true;

    OptionalColArgs options;
    bool isCorrect = true;
    int numGpus = testBed.ev.maxGpus;
    for (int rpg=0; rpg < 2 && isCorrect; ++rpg)
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;
      int ranksPerGpu = rpg == 0 ? 1 : testBed.ev.maxRanksPerGpu;
      int totalRanks = numGpus * ranksPerGpu;
      int const numProcesses = isMultiProcess ? numGpus : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, numGpus, ranksPerGpu, gpuPriorityOrder),
                        {1,2}, //two group, second group sendrecv to self, has 2 coll
                        testBed.GetNumStreamsPerGroup(1,2),
                        2);

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
      for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
      {
        for (int recvRank = 0; recvRank  < totalRanks; ++recvRank)
        {
          options.root = recvRank;
          int groupCallId = sendRank == recvRank;
          int recvId      = sendRank == recvRank;
          testBed.SetCollectiveArgs(ncclCollSend,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    0,
                                    groupCallId,
                                    sendRank);
          if (recvRank == 0)
          {
            testBed.SetCollectiveArgs(ncclCollSend,
                                      dataTypes[dataIdx],
                                      numElements[numIdx],
                                      numElements[numIdx],
                                      options,
                                      0,
                                      !groupCallId,
                                      sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 0, 0, sendRank, userRegistered);
            testBed.PrepareData(0, 0, sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 1, 0, sendRank, userRegistered);
            testBed.PrepareData(1, 0, sendRank);
          }

          if (testBed.ev.showNames) // Show test names
            INFO("%s Datatype: %s SendReceive test Rank %d -> Rank %d for %d Elements\n",
                 isMultiProcess ? "MP" : "SP",
                 ncclDataTypeNames[dataTypes[dataIdx]],
                 sendRank,
                 recvRank,
                 numElements[numIdx]);

          options.root = sendRank;
          testBed.SetCollectiveArgs(ncclCollRecv,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    recvId,
                                    groupCallId,
                                    recvRank);
          testBed.AllocateMem(inPlace, useManagedMem, groupCallId, recvId, recvRank, userRegistered);
          testBed.PrepareData(groupCallId, recvId, recvRank);
          testBed.ExecuteCollectives({sendRank, recvRank}, groupCallId);
          testBed.ValidateResults(isCorrect, groupCallId, recvId, recvRank);
          testBed.DeallocateMem(groupCallId, recvId, recvRank);
        }
        testBed.DeallocateMem(0, 0, sendRank);
        testBed.DeallocateMem(1, 0, sendRank);
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
    unsetenv("RCCL_ENABLE_INTRANET");
  }
}
