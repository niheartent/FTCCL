/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include "StandaloneUtils.hpp"
#include <algorithm>
#include <fstream>
#include <string>
#include <regex>
#include <sys/time.h>
#define MIN_BYTES_SIZE 1024
#define MAX_BYTES_SIZE 1024*1024*1024
#define SIZE_MULTS 16 //Each round is 16 times the size of the previous round
#define TOTAL_SIZE_TYPES 6 //There are 6 sizes in total
#define SMALL_SIZE 16*16*1024
enum ncclAlg{
  AllReduce,
  ReduceScatter,
  Broadcast,
  AllGather,
  Gather,
  Reduce,
  Scatter,
  SendRecv,
  AlltoAll,
  AlltoAllv,
  ncclAlgCount
};

std::vector<int> getCountValues() {
  std::vector<int> val;
    int value = MIN_BYTES_SIZE;
    while (value <= MAX_BYTES_SIZE) {
      val.push_back(value);
      value *= SIZE_MULTS;  
    }    
    return val;
}
enum cpuType{
  cpuIntel,
  cpuHygon,
  totalCpuTypes
};

enum dcuType{
  dcu0 = 0, Vega20 = 0,
  dcu1 = 1, Device66a1 = 1,
  dcu2 = 2, Z100SM = 2,
  dcu3 = 3, DCUK100_AI = 3,
  dcu4 = 4, DCUBW200 = 4,
  totalDcuTypes
};
const static float baseBW[totalDcuTypes*totalCpuTypes][ncclAlgCount*TOTAL_SIZE_TYPES] = {
  {0.0127,0.2597,2.0400,7.6004,9.0011,9.0971, 
   0.0162,0.2740,3.8894,14.6500,17.7845,18.1698,
   0.0201,0.2695,3.3141,11.7356,13.3372,13.5014,
   0.0205,0.2536,3.3501,14.9984,17.8531,18.1789,
   0.0156,0.2536,4.3365,18.3961,18.7311,17.8231,
   0.0192,0.2617,3.5353,11.5945,13.3860,13.5897,
   0.0173,0.2735,3.4606,15.4146,18.0301,18.4960,
   0.0178,0.2710,3.2974,12.1715,13.0758,13.1576,
   0.0148,0.2528,2.0698,13.5213,14.0318,15.7283,
   0.0135,0.1952,2.3364,11.0173,10.9396,12.4550},
  {0.0127,0.2597,2.0400,7.6004,9.0011,9.0971, 
   0.0162,0.2740,3.8894,14.6500,17.7845,18.1698,
   0.0201,0.2695,3.3141,11.7356,13.3372,13.5014,
   0.0205,0.2536,3.3501,14.9984,17.8531,18.1789,
   0.0156,0.2536,4.3365,18.3961,18.7311,17.8231,
   0.0192,0.2617,3.5353,11.5945,13.3860,13.5897,
   0.0173,0.2735,3.4606,15.4146,18.0301,18.4960,
   0.0178,0.2710,3.2974,12.1715,13.0758,13.1576,
   0.0148,0.2528,2.0698,13.5213,14.0318,15.7283,
   0.0135,0.1952,2.3364,11.0173,10.9396,12.4550},
  {0.0273,0.3934,2.0496,7.3921,8.9440,9.0818,
   0.0281,0.4869,3.6233,14.2470,17.5551,18.1096,
   0.0332,0.4995,5.4218,11.5721,13.3646,13.5547,
   0.0288,0.4583,3.7637,14.5107,18.2733,17.5226,
   0.0290,0.4564,3.7583,17.8557,18.2758,17.5065,
   0.0362,0.4589,4.9137,11.4661,13.3684,13.5799,
   0.0364,0.5769,3.4023,15.4857,17.7715,18.1762,
   0.0267,0.3581,5.2852,11.8819,13.0019,13.1001,
   0.0237,0.3753,2.2775,12.8877,14.5557,15.6320,
   0.0229,0.3689,2.3595,10.5663,10.9552,12.2129},
  {0.0273,0.3934,2.0496,7.3921,8.9440,9.0818,
   0.0281,0.4869,3.6233,14.2470,17.5551,18.1096,
   0.0276,0.4995,5.4218,11.5721,13.3646,13.5547,
   0.0288,0.4091,3.7637,14.5107,18.2733,17.5226,
   0.0290,0.4016,3.7583,17.8557,18.2758,17.5065,
   0.0312,0.4589,4.9137,11.4661,13.3684,13.5799,
   0.0251,0.4448,3.4023,15.4857,17.7715,18.1762,
   0.0267,0.3581,5.2852,11.8819,13.0019,13.1001,
   0.0237,0.3753,2.2775,12.8877,14.5557,15.6320,
   0.0229,0.3689,2.3595,10.5663,10.9552,12.2129},
  {0.0230,0.3417,2.3836,7.4091,8.9346,9.0775, 
   0.0223,0.3413,3.5666,14.2421,17.5221,18.0981,
   0.0244,0.4157,4.8916,11.7029,12.5085,13.6024,
   0.0260,0.3388,2.6156,14.5383,17.6656,18.1049,
   0.0214,0.2888,2.9687,17.7050,18.2908,17.4466,
   0.0269,0.4091,5.5539,11.5181,13.3663,13.5433,
   0.0245,0.3044,2.7245,15.1749,17.4033,17.4671,
   0.0225,0.3264,4.7619,11.6352,12.8861,12.9097,
   0.0188,0.2910,1.9153,12.3311,13.5972,15.2259,
   0.0203,0.2933,2.3585,10.5266,11.1489,12.2260},
  {0.0230,0.3417,2.3836,7.4091,8.9346,9.0775, 
   0.0223,0.3413,3.5666,14.2421,17.5221,18.0981,
   0.0244,0.4157,4.8916,11.7029,12.5085,13.6024,
   0.0260,0.3388,2.6156,14.5383,17.6656,18.1049,
   0.0214,0.2888,2.9687,17.7050,18.2908,17.4466,
   0.0269,0.4091,5.5539,11.5181,13.3663,13.5433,
   0.0245,0.3044,2.7245,15.1749,17.4033,17.4671,
   0.0225,0.3264,4.7619,11.6352,12.8861,12.9097,
   0.0188,0.2910,1.9153,12.3311,13.5972,15.2259,
   0.0203,0.2933,2.3585,10.5266,11.1489,12.2260},
  {0.0339,0.4812,5.1961,16.3616,17.9215,18.4031,
   0.0420,0.6350,7.3948,29.3411,34.1078,35.8079,
   0.0540,0.7656,9.3456,22.8635,27.3206,27.7533,
   0.0447,0.6620,7.3636,29.9486,32.8490,36.0888,
   0.0495,0.8131,13.2731,36.3143,35.4998,33.6236,
   0.0391,0.6057,7.7215,22.4174,27.3039,27.6131,
   0.0381,0.6620,9.6911,30.5040,36.3714,36.9538,
   0.0388,0.3637,4.7924,8.0838,24.9012,24.8811,
   0.0345,0.5268,7.8722,24.7817,29.7964,30.1675,
   0.0350,0.5168,7.4052,13.8198,23.6686,24.2767},
  {0.0255,0.3260,2.5929,10.7321,12.8774,13.0821,
   0.0243,0.3728,3.9961,20.7092,24.8548,25.9350,
   0.0343,0.5196,5.8777,16.6332,17.9020,19.5600,
   0.0264,0.3784,4.0642,20.8551,25.1373,26.0106,
   0.0271,0.3383,3.3587,26.7854,26.844,20.0725,
   0.0243,0.4464,4.7839,13.3655,19.2120,18.9865,
   0.0304,0.5053,3.1930,21.9811,25.3290,25.4550,
   0.0323,0.3181,5.7996,16.8887,18.6338,18.8585,
   0.0267,0.3977,2.3127,17.7238,22.0023,22.3397,
   0.0221,0.4001,2.3820,14.5931,16.1938,17.4596},
   {0.0160,0.2502,3.5171,19.9225,61.6904,65.1842,                         // BW200 intel 
   0.0158,0.2498,3.811,37.3061,116.6367,126.6723,
   0.0176,0.2793,3.7719,29.2431,59.6002,104.8252,
   0.0168,0.2815,3.7584,36.8661,117.9901,127.5526,
   0.0163,0.2655,4.1943,57.6043,156.9972,170.8926,
   0.0173,0.2768,3.7272,22.4735,70.5144,104.7928,
   0.0166,0.2098,4.2191,56.9334,156.1107,162.2189,
   0.0169,0.2679,3.4556,32.8017,48.7497,120.8869,
   0.0104,0.1675,2.6390,27.1535,108.3622,124.2317,
   0.0094,0.1644,2.6444,28.2156,100.4166,90.0559,},
   {0.0160,0.2502,3.5171,19.9225,61.6904,65.1842,                         // BW200 hygon 
   0.0158,0.2498,3.811,37.3061,116.6367,126.6723,
   0.0176,0.2793,3.7719,29.2431,59.6002,104.8252,
   0.0168,0.2815,3.7584,36.8661,117.9901,127.5526,
   0.0163,0.2655,4.1943,57.6043,156.9972,170.8926,
   0.0173,0.2768,3.7272,22.4735,70.5144,104.7928,
   0.0166,0.2098,4.2191,56.9334,156.1107,162.2189,
   0.0169,0.2679,3.4556,32.8017,48.7497,120.8869,
   0.0104,0.1675,2.6390,27.1535,108.3622,124.2317,
   0.0094,0.1644,2.6444,28.2156,100.4166,90.0559,}
};

const static float deteriorationIndicators = 0.95;
//Due to performance fluctuations, small data is processed separately: 
//Performance up to 80% of the benchmark is sufficient
const static float SmallSize_deteriorationIndicators = 0.80;

double get_time()
{
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec *1000000 + (double)tp.tv_usec);
}

std::string getCPUName() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line, name;
  while (std::getline(cpuinfo, line)) {
      if (line.find("model name") != std::string::npos || line.find("Processor") != std::string::npos) {
          size_t start = line.find(":");
          if (start != std::string::npos) {
              name = line.substr(start + 1);
              name.erase(name.begin(), name.begin() + name.find_first_not_of(" \t\r\n"));
              std::transform(name.begin(), name.end(), name.begin(), ::tolower);
              break;
          }
      }
  }
  return name;
}

std::string getDCUName() {
  hipDeviceProp_t props;
  if (hipSuccess != hipGetDeviceProperties(&props, 0)){
    printf("[ ERROR    ] Can not get device name!\n");
    return "NULL";
  }
  std::string name(props.name);
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);
  return name;
}

std::string getDCUArch() {
  hipDeviceProp_t props;
  if (hipSuccess != hipGetDeviceProperties(&props, 0)){
    printf("[ ERROR    ] Can not get device arch!\n");
    return "NULL";
  }
  return props.gcnArchName;
}

//Calculate Log2
int log2Custom(int n) {
    int result = 0;
    while (n > 1) {
        n /= 2; 
        result++;
    }
    return result;
}
float getBaseBW(ncclAlg algType,int bytes){
  std::string cpuName = getCPUName();
  int cpuType;
  if (cpuName.find("intel") != std::string::npos){
    cpuType = cpuIntel;
  }else if (cpuName.find("hygon") != std::string::npos){
    cpuType = cpuHygon;
  }else {
    printf("[ ERROR    ] Get unknown cpu name: %s\n", cpuName.c_str());
    return -1.0;
  }
  int iter=(log2Custom(bytes)-10)/4;//Calculate the number of rounds based on size   log2Custom(MIN_BYTES_SIZE)=10 log2Custom(SIZE_MULTS) = 4 
  std::string dcuName = getDCUName();
  if (std::regex_match(dcuName, std::regex(".*vega.*20.*"))){
    return baseBW[dcu0 + cpuType][algType*TOTAL_SIZE_TYPES+iter];
  }else if (std::regex_match(dcuName, std::regex(".*66a1.*"))){
    return baseBW[dcu1 * totalCpuTypes + cpuType][algType*TOTAL_SIZE_TYPES+iter];
  }else if (std::regex_match(dcuName, std::regex(".*z.*sm.*"))){
    return baseBW[dcu2 * totalCpuTypes + cpuType][algType*TOTAL_SIZE_TYPES+iter];
  }else if (std::regex_match(dcuName, std::regex(".*k.*ai.*"))){
    return baseBW[dcu3 * totalCpuTypes + cpuType][algType*TOTAL_SIZE_TYPES+iter];
  }else if (std::regex_match(dcuName, std::regex(".*bw200.*"))){
    return baseBW[dcu4 * totalCpuTypes + cpuType][algType*TOTAL_SIZE_TYPES+iter];
  }else{
    printf("[ ERROR    ] Get unknown device name: %s\n", dcuName.c_str());
    return -1.0;
  }
}

hipError_t allocResource(int numIntraRank, std::vector<ncclComm_t> &comm, float * &iputCpu, float * &oputCpu, float ** iputGpu, float **oputGpu, hipStream_t* stream, int byteSize){
  NCCLCHECK(ncclCommInitAll(comm.data(), numIntraRank, nullptr));
  for (int i = 0; i < numIntraRank; i++)
  {
    HIPCALL(hipSetDevice(i));
    HIPCALL(hipStreamCreate(&stream[i]));
  }

  // Allocate GPU memory
  for (int r = 0; r < numIntraRank; r++)
  {
    HIPCALL(hipSetDevice(r));
    HIPCALL(hipMalloc((void **)&iputGpu[r], byteSize));
    HIPCALL(hipMalloc((void **)&oputGpu[r], byteSize));
  }

  // Allocate CPU memory for input/output
  iputCpu = (float *)malloc(byteSize);
  oputCpu = (float *)malloc(byteSize);

  // Copy the input from CPU memory to GPU memory
  for (int r = 0; r < numIntraRank; r++)
  {
    HIPCALL(hipSetDevice(r));
    HIPCALL(hipMemcpy(iputGpu[r], iputCpu, byteSize, hipMemcpyHostToDevice));
  }
  return hipSuccess;
}

hipError_t releaseResource(int numIntraRank, std::vector<ncclComm_t>& comm, float *iputCpu, float *oputCpu, float **iputGpu, float **oputGpu, hipStream_t* stream){
  for (int r = 0; r < numIntraRank; r++)
  {
    HIPCALL(hipFree(oputGpu[r]));
    HIPCALL(hipFree(iputGpu[r]));
  }

  free(iputCpu);
  free(oputCpu);

  for (int r = 0; r < numIntraRank; r++)
  {
    HIPCALL(hipStreamDestroy(stream[r]));
    NCCLCHECK(ncclCommDestroy(comm[r]));
  }
  return hipSuccess;
}

  

namespace RcclUnitTesting {
  
  class Performance : public ::testing::TestWithParam<int> {};
  TEST_P(Performance, AllReduce)
  {
    int iterNum = 30;
    int byteSize = GetParam();
    std::string archName = getDCUArch();

    if (strcmp(archName.c_str(), "gfx936") != 0){
      // Set environment variables to achieve optimal performance
      if (setenv("NCCL_NCHANNELS_PER_PEER", "4", 1) != 0 || 
          setenv("NCCL_MIN_NCHANNELS", "16", 1) != 0 || 
          setenv("NCCL_MAX_NCHANNELS", "16", 1) != 0 || 
          setenv("NCCL_MIN_P2P_NCHANNELS", "4", 1) != 0 || 
          setenv("NCCL_MAX_P2P_NCHANNELS", "4", 1) != 0) {
          GTEST_SKIP() << "Failed to set environment variable!";
      }
    }

    // Check for multi-GPU
    int numIntraRank;
    HIPCALL(hipGetDeviceCount(&numIntraRank));
    if (numIntraRank < 2) {
        GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    std::vector<ncclComm_t> comm(numIntraRank);
    hipStream_t stream[numIntraRank];
    float *iputGpu[numIntraRank], *oputGpu[numIntraRank];
    float *iputCpu, *oputCpu;

    // Allocate all resources
    allocResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream, byteSize);
    
    auto commPrimTest = [&](){
        for (int iteration = 0; iteration < iterNum; iteration++)
        {
            NCCLCHECK(ncclGroupStart());
            for (int r = 0; r < numIntraRank; r++)
            {
                HIPCALL(hipSetDevice(r));
                NCCLCHECK(ncclAllReduce(iputGpu[r], oputGpu[r], byteSize / sizeof(float), ncclFloat, ncclSum, comm[r], stream[r]));
            }
            NCCLCHECK(ncclGroupEnd());
        }
        for (int r = 0; r < numIntraRank; r++)
            HIPCALL(hipStreamSynchronize(stream[r]));
    };

    // Warm-up
    commPrimTest();

    double start = get_time();
    commPrimTest();
    double costtime = get_time() - start;
  
    // Calculate bandwidth
    double algBW = byteSize / (costtime / iterNum) / 1e3;
    double baseBW = getBaseBW(AllReduce, byteSize);
    
    EXPECT_GT(baseBW, 0);

    // Retry mechanism for bandwidth expectations
    const int maxRetries = 5; // Set the maximum number of retries
    int retries = 0;
    double threshold = (byteSize <= SMALL_SIZE) ? baseBW * SmallSize_deteriorationIndicators : baseBW * deteriorationIndicators;

    while (retries < maxRetries) {
        if (algBW >= threshold) {
            break; // Success, exit the loop
        }
        printf("[ WARNING  ] %.4f < %.4f, Bandwidth test did not meet expectations. Retrying...\n",algBW,threshold);
        commPrimTest(); // Retry the test
        start = get_time();
        commPrimTest();
        costtime = get_time() - start;
        algBW = byteSize / (costtime / iterNum) / 1e3; // Recalculate bandwidth
        retries++;
    }
    printf("[ INFO     ] AllReduce [%d Bytes] real BW of %d cards: %.4f, base BW of %d cards: %.4f\n", byteSize, numIntraRank, algBW, numIntraRank, baseBW);
    EXPECT_GT(algBW, threshold); // Final expectation check

    // Release all resources
    releaseResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream);
  }

  TEST_P(Performance, ReduceScatter)
  {
    int iterNum = 30;
    int byteSize = GetParam();
    std::string archName = getDCUArch();

    if (strcmp(archName.c_str(), "gfx936") != 0){
      // Set environment variables to achieve optimal performance
      if (setenv("NCCL_NCHANNELS_PER_PEER", "4", 1) != 0 || setenv("NCCL_MIN_NCHANNELS", "16", 1) != 0 || 
          setenv("NCCL_MAX_NCHANNELS", "16", 1) != 0 || setenv("NCCL_MIN_P2P_NCHANNELS", "4", 1) != 0 || 
          setenv("NCCL_MAX_P2P_NCHANNELS", "4", 1) != 0) {
        GTEST_SKIP() << "Failed to set environment variable!";
      }
    }

    // Check for multi-gpu
    int numIntraRank;
    HIPCALL(hipGetDeviceCount(&numIntraRank));
    if (numIntraRank < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    std::vector<ncclComm_t> comm(numIntraRank);
    hipStream_t stream[numIntraRank];
    float *iputGpu[numIntraRank], *oputGpu[numIntraRank];
    float *iputCpu, *oputCpu;

    // Allocate all resources
    allocResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream, byteSize);
    
    auto commPrimTest = [&](){
      for (int iteration = 0; iteration < iterNum; iteration++)
      {
        NCCLCHECK(ncclGroupStart());
        for (int r = 0; r < numIntraRank; r++)
        {
          HIPCALL(hipSetDevice(r));
          NCCLCHECK(ncclReduceScatter(iputGpu[r], oputGpu[r], byteSize / sizeof(float) / numIntraRank, ncclFloat, ncclSum, comm[r], stream[r]));
        }
        NCCLCHECK(ncclGroupEnd());
      }
      for (int r = 0; r < numIntraRank; r++)
        HIPCALL(hipStreamSynchronize(stream[r]));
    };

    // Warm-up
    commPrimTest();

    double start = get_time();
    commPrimTest();
    double costtime = get_time() - start;
  
    // Calculate bandwidth
    double algBW = byteSize / (costtime / iterNum) / 1e3;
    double baseBW = getBaseBW(ReduceScatter, byteSize);
    
    EXPECT_GT(baseBW, 0);

    // Retry mechanism for bandwidth expectations
    const int maxRetries = 5; // Set the maximum number of retries
    int retries = 0;
    double threshold = (byteSize <= SMALL_SIZE) ? baseBW * SmallSize_deteriorationIndicators : baseBW * deteriorationIndicators;

    while (retries < maxRetries) {
        if (algBW >= threshold) {
            break; // Success, exit the loop
        }
        printf("[ WARNING  ] %.4f < %.4f, Bandwidth test did not meet expectations. Retrying...\n",algBW,threshold);
        commPrimTest(); // Retry the test
        start = get_time();
        commPrimTest();
        costtime = get_time() - start;
        algBW = byteSize / (costtime / iterNum) / 1e3; // Recalculate bandwidth
        retries++;
    }
    printf("[ INFO     ] ReduceScatter [%d Bytes] real BW of %d cards: %.4f, base BW of %d cards: %.4f\n", byteSize, numIntraRank, algBW, numIntraRank, baseBW);
    EXPECT_GT(algBW, threshold); // Final expectation check

    // Release all resources
    releaseResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream);
  }

  TEST_P(Performance, Broadcast)
  {
    int iterNum = 30;
    int byteSize = GetParam();
    std::string archName = getDCUArch();

    if (strcmp(archName.c_str(), "gfx936") != 0){
      // Set environment variables to achieve optimal performance
      if (setenv("NCCL_NCHANNELS_PER_PEER", "4", 1) != 0 || setenv("NCCL_MIN_NCHANNELS", "16", 1) != 0 || 
          setenv("NCCL_MAX_NCHANNELS", "16", 1) != 0 || setenv("NCCL_MIN_P2P_NCHANNELS", "4", 1) != 0 || 
          setenv("NCCL_MAX_P2P_NCHANNELS", "4", 1) != 0) {
        GTEST_SKIP() << "Failed to set environment variable!";
      }
    }

    // Check for multi-gpu
    int numIntraRank;
    HIPCALL(hipGetDeviceCount(&numIntraRank));
    if (numIntraRank < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    std::vector<ncclComm_t> comm(numIntraRank);
    hipStream_t stream[numIntraRank];
    float *iputGpu[numIntraRank], *oputGpu[numIntraRank];
    float *iputCpu, *oputCpu;

    // Allocate all resources
    allocResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream, byteSize);
    
    auto commPrimTest = [&](){
      for (int iteration = 0; iteration < iterNum; iteration++)
      {
        NCCLCHECK(ncclGroupStart());
        for (int r = 0; r < numIntraRank; r++)
        {
          HIPCALL(hipSetDevice(r));
          NCCLCHECK(ncclBcast(iputGpu[r], byteSize / sizeof(float), ncclFloat, 0, comm[r], stream[r]));
        }
        NCCLCHECK(ncclGroupEnd());
      }
      for (int r = 0; r < numIntraRank; r++)
        HIPCALL(hipStreamSynchronize(stream[r]));
    };

        // Warm-up
    commPrimTest();

    double start = get_time();
    commPrimTest();
    double costtime = get_time() - start;
  
    // Calculate bandwidth
    double algBW = byteSize / (costtime / iterNum) / 1e3;
    double baseBW = getBaseBW(Broadcast, byteSize);
    
    EXPECT_GT(baseBW, 0);

    // Retry mechanism for bandwidth expectations
    const int maxRetries = 5; // Set the maximum number of retries
    int retries = 0;
    double threshold = (byteSize <= SMALL_SIZE) ? baseBW * SmallSize_deteriorationIndicators : baseBW * deteriorationIndicators;

    while (retries < maxRetries) {
        if (algBW >= threshold) {
            break; // Success, exit the loop
        }
        printf("[ WARNING  ] %.4f < %.4f, Bandwidth test did not meet expectations. Retrying...\n",algBW,threshold);
        commPrimTest(); // Retry the test
        start = get_time();
        commPrimTest();
        costtime = get_time() - start;
        algBW = byteSize / (costtime / iterNum) / 1e3; // Recalculate bandwidth
        retries++;
    }
    printf("[ INFO     ] Broadcast [%d Bytes] real BW of %d cards: %.4f, base BW of %d cards: %.4f\n", byteSize, numIntraRank, algBW, numIntraRank, baseBW);
    EXPECT_GT(algBW, threshold); // Final expectation check

    // Release all resources
    releaseResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream);
  }
  TEST_P(Performance, AllGather)
  {
    int iterNum = 30;
    int byteSize = GetParam();
    std::string archName = getDCUArch();

    if (strcmp(archName.c_str(), "gfx936") != 0){
      // Set environment variables to achieve optimal performance
      if (setenv("NCCL_NCHANNELS_PER_PEER", "4", 1) != 0 || setenv("NCCL_MIN_NCHANNELS", "16", 1) != 0 || 
          setenv("NCCL_MAX_NCHANNELS", "16", 1) != 0 || setenv("NCCL_MIN_P2P_NCHANNELS", "4", 1) != 0 || 
          setenv("NCCL_MAX_P2P_NCHANNELS", "4", 1) != 0) {
        GTEST_SKIP() << "Failed to set environment variable!";
      }
    }

    // Check for multi-gpu
    int numIntraRank;
    HIPCALL(hipGetDeviceCount(&numIntraRank));
    if (numIntraRank < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    std::vector<ncclComm_t> comm(numIntraRank);
    hipStream_t stream[numIntraRank];
    float *iputGpu[numIntraRank], *oputGpu[numIntraRank];
    float *iputCpu, *oputCpu;

    // Allocate all resources
    allocResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream, byteSize);
    
    auto commPrimTest = [&](){
      for (int iteration = 0; iteration < iterNum; iteration++)
      {
        NCCLCHECK(ncclGroupStart());
        for (int r = 0; r < numIntraRank; r++)
        {
          HIPCALL(hipSetDevice(r));
          NCCLCHECK(ncclAllGather(iputGpu[r], oputGpu[r], byteSize / sizeof(float) / numIntraRank, ncclFloat, comm[r], stream[r]));
        }
        NCCLCHECK(ncclGroupEnd());
      }
      for (int r = 0; r < numIntraRank; r++)
        HIPCALL(hipStreamSynchronize(stream[r]));
    };

        // Warm-up
    commPrimTest();

    double start = get_time();
    commPrimTest();
    double costtime = get_time() - start;
  
    // Calculate bandwidth
    double algBW = byteSize / (costtime / iterNum) / 1e3;
    double baseBW = getBaseBW(AllGather, byteSize);
    
    EXPECT_GT(baseBW, 0);

    // Retry mechanism for bandwidth expectations
    const int maxRetries = 5; // Set the maximum number of retries
    int retries = 0;
    double threshold = (byteSize <= SMALL_SIZE) ? baseBW * SmallSize_deteriorationIndicators : baseBW * deteriorationIndicators;

    while (retries < maxRetries) {
        if (algBW >= threshold) {
            break; // Success, exit the loop
        }
        printf("[ WARNING  ] %.4f < %.4f, Bandwidth test did not meet expectations. Retrying...\n",algBW,threshold);
        commPrimTest(); // Retry the test
        start = get_time();
        commPrimTest();
        costtime = get_time() - start;
        algBW = byteSize / (costtime / iterNum) / 1e3; // Recalculate bandwidth
        retries++;
    }
    printf("[ INFO     ] AllGather [%d Bytes] real BW of %d cards: %.4f, base BW of %d cards: %.4f\n", byteSize, numIntraRank, algBW, numIntraRank, baseBW);
    EXPECT_GT(algBW, threshold); // Final expectation check

    // Release all resources
    releaseResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream);
  }

  TEST_P(Performance, Gather)
  {
    int iterNum = 30;
    int byteSize = GetParam();
    std::string archName = getDCUArch();

    if (strcmp(archName.c_str(), "gfx936") != 0){
      // Set environment variables to achieve optimal performance
      if (setenv("NCCL_NCHANNELS_PER_PEER", "4", 1) != 0 || setenv("NCCL_MIN_NCHANNELS", "16", 1) != 0 || 
          setenv("NCCL_MAX_NCHANNELS", "16", 1) != 0 || setenv("NCCL_MIN_P2P_NCHANNELS", "4", 1) != 0 || 
          setenv("NCCL_MAX_P2P_NCHANNELS", "4", 1) != 0) {
        GTEST_SKIP() << "Failed to set environment variable!";
      }
    }

    // Check for multi-gpu
    int numIntraRank;
    HIPCALL(hipGetDeviceCount(&numIntraRank));
    if (numIntraRank < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    std::vector<ncclComm_t> comm(numIntraRank);
    hipStream_t stream[numIntraRank];
    float *iputGpu[numIntraRank], *oputGpu[numIntraRank];
    float *iputCpu, *oputCpu;

    // Allocate all resources
    allocResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream, byteSize);
    
    auto commPrimTest = [&](){
      for (int iteration = 0; iteration < iterNum; iteration++)
      {
        NCCLCHECK(ncclGroupStart());
        for (int r = 0; r < numIntraRank; r++)
        {
          HIPCALL(hipSetDevice(r));
          NCCLCHECK(ncclGather(iputGpu[r], oputGpu[r], byteSize / sizeof(float) / numIntraRank, ncclFloat, 0, comm[r], stream[r]));
        }
        NCCLCHECK(ncclGroupEnd());
      }
      for (int r = 0; r < numIntraRank; r++)
        HIPCALL(hipStreamSynchronize(stream[r]));
    };

        // Warm-up
    commPrimTest();

    double start = get_time();
    commPrimTest();
    double costtime = get_time() - start;
  
    // Calculate bandwidth
    double algBW = byteSize / (costtime / iterNum) / 1e3;
    double baseBW = getBaseBW(Gather, byteSize);
    
    EXPECT_GT(baseBW, 0);

    // Retry mechanism for bandwidth expectations
    const int maxRetries = 5; // Set the maximum number of retries
    int retries = 0;
    double threshold = (byteSize <= SMALL_SIZE) ? baseBW * SmallSize_deteriorationIndicators : baseBW * deteriorationIndicators;

    while (retries < maxRetries) {
        if (algBW >= threshold) {
            break; // Success, exit the loop
        }
        printf("[ WARNING  ] %.4f < %.4f, Bandwidth test did not meet expectations. Retrying...\n",algBW,threshold);
        commPrimTest(); // Retry the test
        start = get_time();
        commPrimTest();
        costtime = get_time() - start;
        algBW = byteSize / (costtime / iterNum) / 1e3; // Recalculate bandwidth
        retries++;
    }
    printf("[ INFO     ] Gather [%d Bytes] real BW of %d cards: %.4f, base BW of %d cards: %.4f\n", byteSize, numIntraRank, algBW, numIntraRank, baseBW);
    EXPECT_GT(algBW, threshold); // Final expectation check

    // Release all resources
    releaseResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream);
  }

  TEST_P(Performance, Reduce)
  {
    int iterNum = 30;
    int byteSize = GetParam();
    std::string archName = getDCUArch();

    if (strcmp(archName.c_str(), "gfx936") != 0){
      // Set environment variables to achieve optimal performance
      if (setenv("NCCL_NCHANNELS_PER_PEER", "4", 1) != 0 || setenv("NCCL_MIN_NCHANNELS", "16", 1) != 0 || 
          setenv("NCCL_MAX_NCHANNELS", "16", 1) != 0 || setenv("NCCL_MIN_P2P_NCHANNELS", "4", 1) != 0 || 
          setenv("NCCL_MAX_P2P_NCHANNELS", "4", 1) != 0) {
        GTEST_SKIP() << "Failed to set environment variable!";
      }
    }

    // Check for multi-gpu
    int numIntraRank;
    HIPCALL(hipGetDeviceCount(&numIntraRank));
    if (numIntraRank < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    std::vector<ncclComm_t> comm(numIntraRank);
    hipStream_t stream[numIntraRank];
    float *iputGpu[numIntraRank], *oputGpu[numIntraRank];
    float *iputCpu, *oputCpu;

    // Allocate all resources
    allocResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream, byteSize);
    
    auto commPrimTest = [&](){
      for (int iteration = 0; iteration < iterNum; iteration++)
      {
        NCCLCHECK(ncclGroupStart());
        for (int r = 0; r < numIntraRank; r++)
        {
          HIPCALL(hipSetDevice(r));
          NCCLCHECK(ncclReduce(iputGpu[r], oputGpu[r], byteSize / sizeof(float), ncclFloat, ncclSum, 0, comm[r], stream[r]));
        }
        NCCLCHECK(ncclGroupEnd());
      }
      for (int r = 0; r < numIntraRank; r++)
        HIPCALL(hipStreamSynchronize(stream[r]));
    };

        // Warm-up
    commPrimTest();

    double start = get_time();
    commPrimTest();
    double costtime = get_time() - start;
  
    // Calculate bandwidth
    double algBW = byteSize / (costtime / iterNum) / 1e3;
    double baseBW = getBaseBW(Reduce, byteSize);
    
    EXPECT_GT(baseBW, 0);

    // Retry mechanism for bandwidth expectations
    const int maxRetries = 5; // Set the maximum number of retries
    int retries = 0;
    double threshold = (byteSize <= SMALL_SIZE) ? baseBW * SmallSize_deteriorationIndicators : baseBW * deteriorationIndicators;

    while (retries < maxRetries) {
        if (algBW >= threshold) {
            break; // Success, exit the loop
        }
        printf("[ WARNING  ] %.4f < %.4f, Bandwidth test did not meet expectations. Retrying...\n",algBW,threshold);
        commPrimTest(); // Retry the test
        start = get_time();
        commPrimTest();
        costtime = get_time() - start;
        algBW = byteSize / (costtime / iterNum) / 1e3; // Recalculate bandwidth
        retries++;
    }
    printf("[ INFO     ] Reduce [%d Bytes] real BW of %d cards: %.4f, base BW of %d cards: %.4f\n", byteSize, numIntraRank, algBW, numIntraRank, baseBW);
    EXPECT_GT(algBW, threshold); // Final expectation check

    // Release all resources
    releaseResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream);
  }

  TEST_P(Performance, Scatter)
  {
    int iterNum = 30;
    int byteSize = GetParam();
    std::string archName = getDCUArch();

    if (strcmp(archName.c_str(), "gfx936") != 0){
      // Set environment variables to achieve optimal performance
      if (setenv("NCCL_NCHANNELS_PER_PEER", "4", 1) != 0 || setenv("NCCL_MIN_NCHANNELS", "16", 1) != 0 || 
          setenv("NCCL_MAX_NCHANNELS", "16", 1) != 0 || setenv("NCCL_MIN_P2P_NCHANNELS", "4", 1) != 0 || 
          setenv("NCCL_MAX_P2P_NCHANNELS", "4", 1) != 0) {
        GTEST_SKIP() << "Failed to set environment variable!";
      }
    }

    // Check for multi-gpu
    int numIntraRank;
    HIPCALL(hipGetDeviceCount(&numIntraRank));
    if (numIntraRank < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    std::vector<ncclComm_t> comm(numIntraRank);
    hipStream_t stream[numIntraRank];
    float *iputGpu[numIntraRank], *oputGpu[numIntraRank];
    float *iputCpu, *oputCpu;

    // Allocate all resources
    allocResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream, byteSize);
    size_t Offset = byteSize / sizeof(float) / numIntraRank;
    auto commPrimTest = [&](){
      for (int iteration = 0; iteration < iterNum; iteration++)
      {
        NCCLCHECK(ncclGroupStart());
        for (int r = 0; r < numIntraRank; r++)
        {
          HIPCALL(hipSetDevice(r));
          NCCLCHECK(ncclScatter(iputGpu[r], iputGpu[r]+ Offset * r, byteSize / sizeof(float) / numIntraRank, ncclFloat, 0, comm[r], stream[r]));
        }
        NCCLCHECK(ncclGroupEnd());
      }
      for (int r = 0; r < numIntraRank; r++)
        HIPCALL(hipStreamSynchronize(stream[r]));
    };

        // Warm-up
    commPrimTest();

    double start = get_time();
    commPrimTest();
    double costtime = get_time() - start;
  
    // Calculate bandwidth
    double algBW = byteSize / (costtime / iterNum) / 1e3;
    double baseBW = getBaseBW(Scatter, byteSize);
    
    EXPECT_GT(baseBW, 0);

    // Retry mechanism for bandwidth expectations
    const int maxRetries = 5; // Set the maximum number of retries
    int retries = 0;
    double threshold = (byteSize <= SMALL_SIZE) ? baseBW * SmallSize_deteriorationIndicators : baseBW * deteriorationIndicators;

    while (retries < maxRetries) {
        if (algBW >= threshold) {
            break; // Success, exit the loop
        }
        printf("[ WARNING  ] %.4f < %.4f, Bandwidth test did not meet expectations. Retrying...\n",algBW,threshold);
        commPrimTest(); // Retry the test
        start = get_time();
        commPrimTest();
        costtime = get_time() - start;
        algBW = byteSize / (costtime / iterNum) / 1e3; // Recalculate bandwidth
        retries++;
    }
    printf("[ INFO     ] Scatter [%d Bytes] real BW of %d cards: %.4f, base BW of %d cards: %.4f\n", byteSize, numIntraRank, algBW, numIntraRank, baseBW);
    EXPECT_GT(algBW, threshold); // Final expectation check

    // Release all resources
    releaseResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream);
  }

  TEST_P(Performance, SendRecv)
  {
    int iterNum = 30;
    int byteSize = GetParam();
    std::string archName = getDCUArch();

    if (strcmp(archName.c_str(), "gfx936") != 0){
      // Set environment variables to achieve optimal performance
      if (setenv("NCCL_NCHANNELS_PER_PEER", "4", 1) != 0 || setenv("NCCL_MIN_NCHANNELS", "16", 1) != 0 || 
          setenv("NCCL_MAX_NCHANNELS", "16", 1) != 0 || setenv("NCCL_MIN_P2P_NCHANNELS", "4", 1) != 0 || 
          setenv("NCCL_MAX_P2P_NCHANNELS", "4", 1) != 0) {
        GTEST_SKIP() << "Failed to set environment variable!";
      }
    }

    // Check for multi-gpu
    int numIntraRank;
    HIPCALL(hipGetDeviceCount(&numIntraRank));
    if (numIntraRank < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    std::vector<ncclComm_t> comm(numIntraRank);
    hipStream_t stream[numIntraRank];
    float *iputGpu[numIntraRank], *oputGpu[numIntraRank];
    float *iputCpu, *oputCpu;

    // Allocate all resources
    allocResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream, byteSize);
    
    auto commPrimTest = [&](){
      for (int iteration = 0; iteration < iterNum; iteration++)
      {
        NCCLCHECK(ncclGroupStart());
        for (int r = 0; r < numIntraRank; r++)
        {
        HIPCALL(hipSetDevice(r));
        int nRanks;
        NCCLCHECK(ncclCommCount(comm[r], &nRanks));
        int rank;
        NCCLCHECK(ncclCommUserRank(comm[r], &rank));
        int recvPeer = (rank-1+nRanks) % nRanks;
        int sendPeer = (rank+1) % nRanks;

        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclSend(iputGpu[r], byteSize / sizeof(float), ncclFloat, sendPeer, comm[r], stream[r]));
        NCCLCHECK(ncclRecv(oputGpu[r], byteSize / sizeof(float), ncclFloat, recvPeer, comm[r], stream[r]));
        NCCLCHECK(ncclGroupEnd());
        }
        NCCLCHECK(ncclGroupEnd());
      }
      for (int r = 0; r < numIntraRank; r++)
        HIPCALL(hipStreamSynchronize(stream[r]));
    };

        // Warm-up
    commPrimTest();

    double start = get_time();
    commPrimTest();
    double costtime = get_time() - start;
  
    // Calculate bandwidth
    double algBW = byteSize / (costtime / iterNum) / 1e3;
    double baseBW = getBaseBW(SendRecv, byteSize);
    
    EXPECT_GT(baseBW, 0);

    // Retry mechanism for bandwidth expectations
    const int maxRetries = 5; // Set the maximum number of retries
    int retries = 0;
    double threshold = (byteSize <= SMALL_SIZE) ? baseBW * SmallSize_deteriorationIndicators : baseBW * deteriorationIndicators;

    while (retries < maxRetries) {
        if (algBW >= threshold) {
            break; // Success, exit the loop
        }
        printf("[ WARNING  ] %.4f < %.4f, Bandwidth test did not meet expectations. Retrying...\n",algBW,threshold);
        commPrimTest(); // Retry the test
        start = get_time();
        commPrimTest();
        costtime = get_time() - start;
        algBW = byteSize / (costtime / iterNum) / 1e3; // Recalculate bandwidth
        retries++;
    }
    printf("[ INFO     ] SendRecv [%d Bytes] real BW of %d cards: %.4f, base BW of %d cards: %.4f\n", byteSize, numIntraRank, algBW, numIntraRank, baseBW);
    EXPECT_GT(algBW, threshold); // Final expectation check

    // Release all resources
    releaseResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream);
  }

  TEST_P(Performance, AlltoAll)
  {
    int iterNum = 30;
    int byteSize = GetParam();
    std::string archName = getDCUArch();
    
    if (strcmp(archName.c_str(), "gfx936") != 0){
      // Set environment variables to achieve optimal performance
      if (setenv("NCCL_NCHANNELS_PER_PEER", "4", 1) != 0 || setenv("NCCL_MIN_NCHANNELS", "16", 1) != 0 || 
          setenv("NCCL_MAX_NCHANNELS", "16", 1) != 0 || setenv("NCCL_MIN_P2P_NCHANNELS", "4", 1) != 0 || 
          setenv("NCCL_MAX_P2P_NCHANNELS", "4", 1) != 0) {
        GTEST_SKIP() << "Failed to set environment variable!";
      }
    }

    // Check for multi-gpu
    int numIntraRank;
    HIPCALL(hipGetDeviceCount(&numIntraRank));
    if (numIntraRank < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    std::vector<ncclComm_t> comm(numIntraRank);
    hipStream_t stream[numIntraRank];
    float *iputGpu[numIntraRank], *oputGpu[numIntraRank];
    float *iputCpu, *oputCpu;

    // Allocate all resources
    allocResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream, byteSize);
    
    auto commPrimTest = [&](){
      for (int iteration = 0; iteration < iterNum; iteration++)
      {
        NCCLCHECK(ncclGroupStart());
        for (int r = 0; r < numIntraRank; r++)
        {
          HIPCALL(hipSetDevice(r));
          NCCLCHECK(ncclAllToAll(iputGpu[r], oputGpu[r], byteSize / sizeof(float) / numIntraRank, ncclFloat, comm[r], stream[r]));
        }
        NCCLCHECK(ncclGroupEnd());
      }
      for (int r = 0; r < numIntraRank; r++)
        HIPCALL(hipStreamSynchronize(stream[r]));
    };

        // Warm-up
    commPrimTest();

    double start = get_time();
    commPrimTest();
    double costtime = get_time() - start;
  
    // Calculate bandwidth
    double algBW = byteSize / (costtime / iterNum) / 1e3;
    double baseBW = getBaseBW(AlltoAll, byteSize);
    
    EXPECT_GT(baseBW, 0);

    // Retry mechanism for bandwidth expectations
    const int maxRetries = 5; // Set the maximum number of retries
    int retries = 0;
    double threshold = (byteSize <= SMALL_SIZE) ? baseBW * SmallSize_deteriorationIndicators : baseBW * deteriorationIndicators;

    while (retries < maxRetries) {
        if (algBW >= threshold) {
            break; // Success, exit the loop
        }
        printf("[ WARNING  ] %.4f < %.4f, Bandwidth test did not meet expectations. Retrying...\n",algBW,threshold);
        commPrimTest(); // Retry the test
        start = get_time();
        commPrimTest();
        costtime = get_time() - start;
        algBW = byteSize / (costtime / iterNum) / 1e3; // Recalculate bandwidth
        retries++;
    }
    printf("[ INFO     ] AlltoAll [%d Bytes] real BW of %d cards: %.4f, base BW of %d cards: %.4f\n", byteSize, numIntraRank, algBW, numIntraRank, baseBW);
    EXPECT_GT(algBW, threshold); // Final expectation check

    // Release all resources
    releaseResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream);
  }

  TEST_P(Performance, AlltoAllv)
  {
    int iterNum = 30;
    int byteSize = GetParam();
    std::string archName = getDCUArch();

    if (strcmp(archName.c_str(), "gfx936") != 0){
      // Set environment variables to achieve optimal performance
      if (setenv("NCCL_NCHANNELS_PER_PEER", "4", 1) != 0 || setenv("NCCL_MIN_NCHANNELS", "16", 1) != 0 || 
          setenv("NCCL_MAX_NCHANNELS", "16", 1) != 0 || setenv("NCCL_MIN_P2P_NCHANNELS", "4", 1) != 0 || 
          setenv("NCCL_MAX_P2P_NCHANNELS", "4", 1) != 0) {
        GTEST_SKIP() << "Failed to set environment variable!";
      }
    }

    // Check for multi-gpu
    int numIntraRank;
    HIPCALL(hipGetDeviceCount(&numIntraRank));
    if (numIntraRank < 2) {
      GTEST_SKIP() << "This test requires at least 2 devices.";
    }

    std::vector<ncclComm_t> comm(numIntraRank);
    hipStream_t stream[numIntraRank];
    float *iputGpu[numIntraRank], *oputGpu[numIntraRank];
    float *iputCpu, *oputCpu;

    // Allocate all resources
    allocResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream, byteSize);
    
    auto commPrimTest = [&](){
      for (int iteration = 0; iteration < iterNum; iteration++)
      {
        NCCLCHECK(ncclGroupStart());
        for (int r = 0; r < numIntraRank; r++)
        {
          int nranks;
          NCCLCHECK(ncclCommCount(comm[r], &nranks));
          int rank;
          NCCLCHECK(ncclCommUserRank(comm[r], &rank));
          #define MAX_ALLTOALLV_RANKS 256
          static size_t sendcounts[MAX_ALLTOALLV_RANKS*MAX_ALLTOALLV_RANKS], recvcounts[MAX_ALLTOALLV_RANKS*MAX_ALLTOALLV_RANKS], sdispls[MAX_ALLTOALLV_RANKS*MAX_ALLTOALLV_RANKS], rdispls[MAX_ALLTOALLV_RANKS*MAX_ALLTOALLV_RANKS];      
          size_t disp = 0;
          size_t chunksize = byteSize / sizeof(float) / numIntraRank*2/nranks;
          for (int i = 0; i < nranks; i++) {
            size_t scount = ((i+rank)%nranks)*chunksize;
            if ((i+rank)%nranks == 0)
              scount += (byteSize / sizeof(float) / numIntraRank*nranks-chunksize*(nranks-1)*nranks/2);
            sendcounts[i+rank*MAX_ALLTOALLV_RANKS] = recvcounts[i+rank*MAX_ALLTOALLV_RANKS] = scount;
            sdispls[i+rank*MAX_ALLTOALLV_RANKS] = rdispls[i+rank*MAX_ALLTOALLV_RANKS] = disp;
            disp += scount;
          }
          HIPCALL(hipSetDevice(r));
          NCCLCHECK(ncclAllToAllv(iputGpu[r], sendcounts+rank*MAX_ALLTOALLV_RANKS, sdispls+rank*MAX_ALLTOALLV_RANKS, oputGpu[r], recvcounts+rank*MAX_ALLTOALLV_RANKS, rdispls+rank*MAX_ALLTOALLV_RANKS, ncclFloat, comm[r], stream[r]));
        }
        NCCLCHECK(ncclGroupEnd());
      }
      for (int r = 0; r < numIntraRank; r++)
        HIPCALL(hipStreamSynchronize(stream[r]));
    };

        // Warm-up
    commPrimTest();

    double start = get_time();
    commPrimTest();
    double costtime = get_time() - start;
  
    // Calculate bandwidth
    double algBW = byteSize / (costtime / iterNum) / 1e3;
    double baseBW = getBaseBW(AlltoAllv, byteSize);
    
    EXPECT_GT(baseBW, 0);

    // Retry mechanism for bandwidth expectations
    const int maxRetries = 5; // Set the maximum number of retries
    int retries = 0;
    double threshold = (byteSize <= SMALL_SIZE) ? baseBW * SmallSize_deteriorationIndicators : baseBW * deteriorationIndicators;

    while (retries < maxRetries) {
        if (algBW >= threshold) {
            break; // Success, exit the loop
        }
        printf("[ WARNING  ] %.4f < %.4f, Bandwidth test did not meet expectations. Retrying...\n",algBW,threshold);
        commPrimTest(); // Retry the test
        start = get_time();
        commPrimTest();
        costtime = get_time() - start;
        algBW = byteSize / (costtime / iterNum) / 1e3; // Recalculate bandwidth
        retries++;
    }
    printf("[ INFO     ] AlltoAllv [%d Bytes] real BW of %d cards: %.4f, base BW of %d cards: %.4f\n", byteSize, numIntraRank, algBW, numIntraRank, baseBW);
    EXPECT_GT(algBW, threshold); // Final expectation check

    // Release all resources
    releaseResource(numIntraRank, comm, iputCpu, oputCpu, iputGpu, oputGpu, stream);
  }

  std::vector<int> CountValues = getCountValues();
  INSTANTIATE_TEST_SUITE_P( 
    Performance_Test, 
    Performance, 
    ::testing::ValuesIn(CountValues) 
  );
}
