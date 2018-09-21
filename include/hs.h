//
//  hs.h
//  libhs
//
//  Created by horned-sungem on 2018/4/17.
//  Copyright © 2018年 Senscape. All rights reserved.
//

#ifndef __HS_H_INCLUDED__
#define __HS_H_INCLUDED__

#ifndef __cplusplus
#define bool _Bool
#define true 1
#define false 0
#elif defined(__GNUC__) && !defined(__STRICT_ANSI__)
/* Define _Bool, bool, false, true as a GNU extension. */
#define _Bool bool
#define bool  bool
#define false false
#define true  true
#endif

#define __bool_true_false_are_defined 1

#ifdef __cplusplus
extern "C" {
#endif
    
#define HS_MAX_NAME_SIZE 28
    
    typedef enum {
        HS_OK = 0,
        HS_BUSY = -1,
        HS_ERROR = -2,
        HS_OUT_OF_MEMORY = -3,
        HS_DEVICE_NOT_FOUND = -4,
        HS_INVALID_PARAMETERS = -5,
        HS_TIMEOUT = -6,
        HS_NO_DATA = -8,
        HS_GONE = -9,
        HS_UNSUPPORTED_GRAPH_FILE = -10,
        HS_MYRIAD_ERROR = -11,
    } hsStatus;
    
    typedef enum {
        HS_LOG_LEVEL = 0,
    } hsGlobalOptions;
    
    typedef enum {
        HS_ITERATIONS = 0,
        HS_NETWORK_THROTTLE = 1,
        HS_DONT_BLOCK = 2,
        HS_TIME_TAKEN = 1000,
        HS_DEBUG_INFO = 1001,
        HS_GRAPH_ID = 1002,
    } hsGraphOptions;
    
    typedef enum {
        HS_TEMP_LIM_LOWER = 1,
        HS_TEMP_LIM_HIGHER = 2,
        HS_BACKOFF_TIME_NORMAL = 3,
        HS_BACKOFF_TIME_HIGH = 4,
        HS_BACKOFF_TIME_CRITICAL = 5,
        HS_TEMPERATURE_DEBUG = 6,
        HS_THERMAL_STATS = 1000,
        HS_OPTIMISATION_LIST = 1001,
        HS_THERMAL_THROTTLING_LEVEL = 1002,
    } hsDeviceOptions;
    
    hsStatus hsGetDeviceName(int index, char *name, unsigned int nameSize);
    hsStatus hsOpenDevice(const char *name, void **deviceHandle);
    hsStatus hsCloseDevice(void *deviceHandle);
    hsStatus hsUpdateApp(void *deviceHandle, char *path);
    hsStatus hsBootUpdateApp(char *path);
    hsStatus hsAllocateGraph(void *deviceHandle, void **graphHandle, const void *graphFile, unsigned int graphFileLength);
    hsStatus hsDeallocateGraph(void *graphHandle);
    hsStatus hsSetGlobalOption(int option, const void *data, unsigned int dataLength);
    hsStatus hsGetGlobalOption(int option, void *data, unsigned int *dataLength);
    hsStatus hsSetGraphOption(void *graphHandle, int option, const void *data, unsigned int dataLength);
    hsStatus hsGetGraphOption(void *graphHandle, int option, void *data, unsigned int *dataLength);
    hsStatus hsSetDeviceOption(void *deviceHandle, int option, const void *data, unsigned int dataLength);
    hsStatus hsGetDeviceOption(void *deviceHandle, int option, void *data, unsigned int *dataLength);
    hsStatus hsLoadTensor(void *graphHandle, const void *inputTensor, int id, unsigned int inputTensorLength, void *userParam);
    hsStatus hsGetResult(void *graphHandle, void **outputData, unsigned int *outputDataLength, void **userParam);
    hsStatus hsDeviceGetImage(void *deviceHandle, void **outputData, bool truthy);
    hsStatus hsGetImage(void *graphHandle, void **outputData, int id, void *userParam,float std_value,float mean_value, bool truthy);
    
#ifdef __cplusplus
}
#endif

#endif
