/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP8_OPENCL_H
#define	VP8_OPENCL_H

#ifdef	__cplusplus
extern "C" {
#endif

#include "vpx_config.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if HAVE_DLOPEN
#include "dynamic_cl.h"
#endif

extern char *cl_read_file(const char* file_name);
extern int cl_init();
extern void cl_destroy();

#define MAX_NUM_PLATFORMS 4

#define CL_TRIED_BUT_FAILED 1
#define CL_NOT_INITIALIZED -1
extern int cl_initialized;

extern const char *vpx_codec_lib_dir(void);

#define CL_CHECK_SUCCESS(cond,msg,alt,retCode) \
    if ( cond ){ \
        printf(msg);  \
        cl_destroy(); \
        cl_initialized = CL_TRIED_BUT_FAILED; \
        printf("CL operation failed.\n");\
        alt; \
        return retCode; \
    }

#define CL_CREATE_KERNEL(data,program,name,str_name) \
    data.name = clCreateKernel(data.program, str_name , &err); \
    CL_CHECK_SUCCESS(err != CL_SUCCESS || !data.name, \
        "Error: Failed to create compute kernel!\n", \
        ,\
        CL_TRIED_BUT_FAILED \
    );


#define CL_ENSURE_BUF_SIZE(bufRef, bufType, needSize, curSize, dataPtr, altPath) \
    if ( needSize > curSize || bufRef == NULL){ \
        if (bufRef != NULL) \
            clReleaseMemObject(bufRef); \
        if (dataPtr != NULL){ \
            bufRef = clCreateBuffer(cl_data.context, bufType, needSize, dataPtr, &err); \
            CL_CHECK_SUCCESS( \
                err != CL_SUCCESS, \
                "Error copying data to buffer! Using CPU path!\n", \
                altPath, \
            ); \
        } else {\
            bufRef = clCreateBuffer(cl_data.context, bufType, needSize, NULL, NULL);\
        } \
        CL_CHECK_SUCCESS(!bufRef, \
            "Error: Failed to allocate buffer. Using CPU path!\n", \
            altPath, \
        ); \
        curSize = needSize; \
    } else { \
        if (dataPtr != NULL){\
            err = clEnqueueWriteBuffer(cl_data.commands, bufRef, CL_FALSE, 0, \
                needSize, dataPtr, 0, NULL, NULL); \
            \
            CL_CHECK_SUCCESS( err != CL_SUCCESS, \
                "Error: Failed to write to buffer!\n", \
                altPath, \
            ); \
        }\
    }

#define CL_LOAD_PROGRAM(prog_ref, file_name, opts) \
    kernel_src = cl_read_file(file_name); \
    prog_ref = NULL; \
    if (kernel_src != NULL){ \
        printf("creating program from source file\n"); \
        prog_ref = clCreateProgramWithSource(cl_data.context, 1, &kernel_src, NULL, &err); \
        printf("Created program\n"); \
        free(kernel_src); \
    } else { \
        cl_destroy(); \
        printf("Couldn't find OpenCL source files. \nUsing software path.\n"); \
        return CL_TRIED_BUT_FAILED; \
    } \
\
    if (prog_ref == NULL) { \
        printf("Error: Couldn't create program\n"); \
        return CL_TRIED_BUT_FAILED; \
    } \
\
    if (err != CL_SUCCESS){ \
        printf("Error creating program: %d\n", err); \
    } \
\
    /* Build the program executable */ \
    printf("Building program\n"); \
    err = clBuildProgram(prog_ref, 0, NULL, opts, NULL, NULL); \
    printf("Program built\n"); \
    if (err != CL_SUCCESS) { \
        size_t len; \
        char buffer[2048]; \
\
        printf("Error: Failed to build program executable!\n"); \
        clGetProgramBuildInfo(prog_ref, cl_data.device_id, CL_PROGRAM_BUILD_LOG, sizeof (buffer), buffer, &len); \
        printf("Compile output: %s\n", buffer);\
        return CL_TRIED_BUT_FAILED; \
    } \


typedef struct VP8_COMMON_CL {
    cl_device_id device_id; // compute device id
    cl_context context; // compute context
    cl_command_queue commands; // compute command queue

    cl_program filter_program; // compute program for subpixel/bilinear filters
    cl_kernel vp8_block_variation_kernel;
    cl_kernel vp8_sixtap_predict_kernel;
    cl_kernel vp8_sixtap_predict8x4_kernel;
    cl_kernel vp8_sixtap_predict8x8_kernel;
    cl_kernel vp8_sixtap_predict16x16_kernel;

    cl_kernel vp8_bilinear_predict4x4_kernel;
    cl_kernel vp8_bilinear_predict8x4_kernel;
    cl_kernel vp8_bilinear_predict8x8_kernel;
    cl_kernel vp8_bilinear_predict16x16_kernel;

    cl_program idct_program;

    cl_kernel filter_block2d_first_pass_kernel; // compute kernel
    cl_kernel filter_block2d_second_pass_kernel; // compute kernel

    cl_kernel filter_block2d_bil_first_pass_kernel;
    cl_kernel filter_block2d_bil_second_pass_kernel;

    cl_mem srcData; //Source frame data
    size_t srcAlloc; //Amount of allocated CL memory for srcData
    cl_mem destData; //Destination data for 2nd pass.
    size_t destAlloc; //Amount of allocated CL memory for destData
} VP8_COMMON_CL;

extern VP8_COMMON_CL cl_data;

#ifdef	__cplusplus
}
#endif

#endif	/* VP8_OPENCL_H */

