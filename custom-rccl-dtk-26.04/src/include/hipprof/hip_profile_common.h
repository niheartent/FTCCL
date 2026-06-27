/**
 *  @file:   hip_profile_common.h
 *  @brief:  common define for hip_prof and profile。
 *  @author: lizhigong
 *  @data:   2021/05/11
 */
#ifndef HIP_HIP_PROFILE_COMMON_H
#define HIP_HIP_PROFILE_COMMON_H

#include <cstdint>
#include <map>
#include <bitset>
#include <cstring>

typedef uint32_t activity_kind_t;
typedef uint64_t activity_correlation_id_t;

struct hip_prof_rccl_entry {
    activity_kind_t kind;
    uint32_t cid;
    // NCCL Coll Args
    const void* sendbuff;
    void* recvbuff;
    size_t count;
    uint32_t datatype;
    uint32_t op;
    int rid; // peer for p2p operations
    // Computed later
    int algorithm;
    int protocol;
    uint8_t pattern;
    int nChannels;
    int nThreads;
    size_t nBytes;
    int chunkSize;
    int channelId;
    uint64_t elapsed;
    struct {
        activity_correlation_id_t correlation_id; /* activity ID uint64_t */
        uint64_t begin_ns; /* host begin timestamp uint64_t*/
        uint64_t end_ns; /* host end timestamp uint64_t */
    };
    union {
        struct {
            int device_id; /* device id */
            uint64_t queue_id; /* queue id */
        };
        struct {
            uint32_t process_id; /* device id */
            uint32_t thread_id; /* thread id */
        };
    };
    uint32_t ret_stat;
    const char* kernel_name;
    double BW_GBps; //(nBytes/1.0E9)/[(end_ns-begin_ns)/1.0E9]
};

typedef enum {
    STATUS_SUCCESS = 0,
    STATUS_ERROR = -1,
    GET_TIMESTAMP_FAILED = -2,
    INVALID_KIND = -3,
    INVALID_OP = -4,
    API_RETURN_ERROR = -5,
} prof_error_t;

/* rccl entry kinds */
typedef enum {
    RCCL_KIND_ID_KERNEL = 0,
    RCCL_KIND_ID_API = 1,
    RCCL_KIND_ID_NUMBER =2
} entry_kind_t;

#endif // HIP_HIP_PROFILE_COMMON_H
