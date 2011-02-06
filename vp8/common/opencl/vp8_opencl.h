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

#include "../../../vpx_config.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if HAVE_DLOPEN
#include "dynamic_cl.h"
#endif

extern char *cl_read_file(const char* file_name);
extern int cl_common_init();
extern void cl_destroy(cl_command_queue cq, int new_status);
extern int cl_load_program(cl_program *prog_ref, const char *file_name, const char *opts);

#define MAX_NUM_PLATFORMS 4
#define MAX_NUM_DEVICES 10

#define CL_TRIED_BUT_FAILED 1
#define CL_NOT_INITIALIZED -1
extern int cl_initialized;

extern const char *vpx_codec_lib_dir(void);

#define CL_FINISH(cq) \
    if (cl_initialized == CL_SUCCESS){ \
        /* Wait for kernels to finish. */ \
        /* clFinish(cl_data.commands); */ \
        clFinish(cq); \
    }

#define CL_BARRIER(cq) \
    if (cl_initialized == CL_SUCCESS){ \
        /* Insert a barrier into the command queue. */ \
        clEnqueueBarrier(cq); \
    }

#define CL_CHECK_SUCCESS(cq,cond,msg,alt,retCode) \
    if ( cond ){ \
        printf(msg);  \
        cl_destroy(cq, CL_TRIED_BUT_FAILED); \
        alt; \
        return retCode; \
    }

#define CL_CREATE_KERNEL(data,program,name,str_name) \
    data.name = clCreateKernel(data.program, str_name , &err); \
    CL_CHECK_SUCCESS(NULL, err != CL_SUCCESS || !data.name, \
        "Error: Failed to create compute kernel "#str_name"!\n", \
        ,\
        CL_TRIED_BUT_FAILED \
    );

#define CL_SET_BUF(cq, bufRef, bufSize, dataPtr, altPath) \
    { \
        err = clEnqueueWriteBuffer(cq, bufRef, CL_FALSE, 0, \
            bufSize, dataPtr, 0, NULL, NULL); \
        \
        CL_CHECK_SUCCESS(cq, err != CL_SUCCESS, \
            "Error: Failed to write to buffer!\n", \
            altPath, \
        ); \
    } \

#define CL_READ_BUF(cq, bufRef, bufSize, dstPtr) \
    err = clEnqueueReadBuffer(cq, bufRef, CL_FALSE, 0, bufSize , dstPtr, 0, NULL, NULL); \
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS, \
        "Error: Failed to read from GPU!\n",, err \
    ); \

#define CL_CREATE_BUF(cq, bufRef, bufType, bufSize, dataPtr, altPath) \
    if (dataPtr != NULL){ \
        bufRef = clCreateBuffer(cl_data.context, bufType, bufSize, dataPtr, &err); \
        CL_CHECK_SUCCESS(cq, \
            err != CL_SUCCESS, \
            "Error copying data to buffer! Using CPU path!\n", \
            altPath, \
        ); \
    } else {\
        bufRef = clCreateBuffer(cl_data.context, bufType, bufSize, NULL, NULL);\
    } \
    CL_CHECK_SUCCESS(cq, !bufRef, \
        "Error: Failed to allocate buffer. Using CPU path!\n", \
        altPath, \
    ); \

#define CL_ENSURE_BUF_SIZE(cq, bufRef, bufType, needSize, curSize, dataPtr, altPath) \
        /* CL_FINISH(cq); */\
        if ( needSize > curSize || bufRef == NULL || 1){ \
        if (bufRef != NULL) \
            clReleaseMemObject(bufRef); \
        if (dataPtr != NULL){ \
            CL_CREATE_BUF(cq, bufRef, bufType, needSize, dataPtr, altPath); \
        } else {\
            printf("Is this used\n");\
            bufRef = clCreateBuffer(cl_data.context, bufType, needSize, NULL, NULL);\
        } \
        CL_CHECK_SUCCESS(cq, !bufRef, \
            "Error: Failed to allocate buffer. Using CPU path!\n", \
            altPath, \
        ); \
        curSize = needSize; \
    } else { \
        CL_SET_BUF(cq, bufRef, needSize, dataPtr, altPath); \
    }

#define CL_RELEASE_KERNEL(kernel) \
    if (kernel) \
        clReleaseKernel(kernel); \
    kernel = NULL;

typedef struct VP8_COMMON_CL {
    cl_device_id device_id; // compute device id
    cl_context context; // compute context
    //cl_command_queue commands; // compute command queue

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
    cl_kernel vp8_memcpy_kernel;

    cl_program idct_program;
    cl_kernel vp8_short_inv_walsh4x4_1_kernel;
    cl_kernel vp8_short_inv_walsh4x4_1st_pass_kernel;
    cl_kernel vp8_short_inv_walsh4x4_2nd_pass_kernel;

    cl_kernel vp8_dc_only_idct_add_kernel;

    //Note that the following 2 kernels are encoder-only. Not used in decoder.
    cl_kernel vp8_short_idct4x4llm_1_kernel;
    cl_kernel vp8_short_idct4x4llm_kernel;

    cl_program dequant_program;
    cl_kernel vp8_dequant_dc_idct_add_kernel;
    cl_kernel vp8_dequant_idct_add_kernel;
    cl_kernel vp8_dequantize_b_kernel;

    cl_kernel filter_block2d_first_pass_kernel; // compute kernel
    cl_kernel filter_block2d_second_pass_kernel; // compute kernel

    cl_kernel filter_block2d_bil_first_pass_kernel;
    cl_kernel filter_block2d_bil_second_pass_kernel;

    cl_int cl_decode_initialized;
    cl_int cl_encode_initialized;
    
} VP8_COMMON_CL;

extern VP8_COMMON_CL cl_data;

#ifdef	__cplusplus
}
#endif

#endif	/* VP8_OPENCL_H */

