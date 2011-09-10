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

#define ENABLE_CL_IDCT_DEQUANT 0
#define ENABLE_CL_SUBPIXEL 0
#define ENABLE_CL_LOOPFILTER 1
#define TWO_PASS_SIXTAP 0

//Snow Leopard doesn't support CopyRect, Lion does.
#ifdef __APPLE__
#define MEM_COPY_KERNEL 1
#else
//Note that OpenCL 1.0 also doesn't support clEnqueueCopyBufferRect, 1.1 does.
#define MEM_COPY_KERNEL 1 //0 = clEnqueueCopyBufferRect, 1 = kernel
#endif

#define ONE_CQ_PER_MB 1 //Value of 0 is racey... still experimental.

extern int cl_common_init();
extern void cl_destroy(cl_command_queue cq, int new_status);
extern int cl_load_program(cl_program *prog_ref, const char *file_name, const char *opts);

#define MAX_NUM_PLATFORMS 4
#define MAX_NUM_DEVICES 10

#define VP8_CL_TRIED_BUT_FAILED 1
#define VP8_CL_NOT_INITIALIZED -1
extern int cl_initialized;

extern const char *vpx_codec_lib_dir(void);

#define VP8_CL_FINISH(cq) \
    if (cl_initialized == CL_SUCCESS){ \
        /* Wait for kernels to finish. */ \
        clFinish(cq); \
    }

#define VP8_CL_BARRIER(cq) \
    if (cl_initialized == CL_SUCCESS){ \
        /* Insert a barrier into the command queue. */ \
        clEnqueueBarrier(cq); \
    }

#define VP8_CL_CHECK_SUCCESS(cq,cond,msg,alt,retCode) \
    if ( cond ){ \
        fprintf(stderr, msg);  \
        cl_destroy(cq, VP8_CL_TRIED_BUT_FAILED); \
        alt; \
        return retCode; \
    }

//Determines the greatest common factor of two numbers for calculating ideal
//local work sizes and sets the local size to the GCF.
//WARNING: This will destroy the existing value of global, so use a temp copy.
#define VP8_CL_CALC_GCF(global,local,temp) \
    if ( local>global ){ \
        temp=local; local=global; global = temp; \
    } \
    while ( local != 0  ) { \
        temp=local; \
        local = global - local; \
        global = temp; \
        if ( local > global ) { \
            temp=local; local=global; global = temp; \
        } \
    } \
    local = global;


//Gets the maximum work group size supported on the device for the kernel
#define VP8_CL_CALC_LOCAL_SIZE(kernel, kernel_size) \
    err = clGetKernelWorkGroupInfo( kernel, \
  	cl_data.device_id, \
  	CL_KERNEL_WORK_GROUP_SIZE, \
  	sizeof(size_t), \
  	kernel_size, \
  	NULL);\
    VP8_CL_CHECK_SUCCESS(NULL, err != CL_SUCCESS, \
        "Error: Failed to calculate local size of kernel!\n", \
        ,\
        VP8_CL_TRIED_BUT_FAILED \
    ); \

#define VP8_CL_CREATE_KERNEL(data,program,name,str_name) \
    data.name = clCreateKernel(data.program, str_name , &err); \
    VP8_CL_CHECK_SUCCESS(NULL, err != CL_SUCCESS || !data.name, \
        "Error: Failed to create compute kernel "#str_name"!\n", \
        ,\
        VP8_CL_TRIED_BUT_FAILED \
    );

#define VP8_CL_READ_BUF(cq, bufRef, bufSize, dstPtr) \
    err = clEnqueueReadBuffer(cq, bufRef, CL_FALSE, 0, bufSize , dstPtr, 0, NULL, NULL); \
    VP8_CL_CHECK_SUCCESS( cq, err != CL_SUCCESS, \
        "Error: Failed to read from GPU!\n",, err \
    ); \

#define VP8_CL_SET_BUF(cq, bufRef, bufSize, dataPtr, altPath, retCode) \
    { \
        err = clEnqueueWriteBuffer(cq, bufRef, CL_TRUE, 0, \
            bufSize, dataPtr, 0, NULL, NULL); \
        \
        VP8_CL_CHECK_SUCCESS(cq, err != CL_SUCCESS, \
            "Error: Failed to write to buffer!\n", \
            altPath, retCode\
        ); \
    } \

//Allocates a cl_mem on the host using hopefully pinned memory, and then
//creates a host-writable pointer to the mapped memory
//Finally creates a cl_mem object which can be read/written by the device
//Still need to unmap buffer after done writing to it.
#define VP8_CL_CREATE_MAPPED_BUF(cq, pinnedBufRef, dataPtr, bufSize, altPath, retCode) \
    \
    pinnedBufRef =  clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufSize, NULL, NULL); \
    if (pinnedBufRef != NULL){ \
        dataPtr = clEnqueueMapBuffer(cq,pinnedBufRef, CL_TRUE,CL_MAP_WRITE, 0, bufSize, 0,NULL, NULL, NULL); \
    }\
    VP8_CL_CHECK_SUCCESS(cq, !pinnedBufRef && !dataPtr, \
        "Error: Failed to allocate mapped buffer. Using CPU path!\n", \
        altPath, retCode\
    ); \

#define VP8_CL_UNMAP_BUF(cq, pinnedBufRef, dataPtr, altPath, retCode) \
    err = clEnqueueUnmapMemObject(cq, pinnedBufRef, dataPtr, 0, NULL, NULL ); \
    VP8_CL_CHECK_SUCCESS(cq, err != CL_SUCCESS, \
        "Error: Failed to unmap buffer!\n", \
        altPath, retCode\
    );

#define VP8_CL_MAP_BUF(cq, bufRef, bufPtr, bufSize, altPath, retCode) \
    bufPtr = clEnqueueMapBuffer(cq, bufRef, CL_TRUE, CL_MAP_WRITE, 0, bufSize, 0, NULL, NULL, &err); \
    VP8_CL_CHECK_SUCCESS(cq, err != CL_SUCCESS, \
        "Error failed to map buffer\n", \
        printf("Err = %d\n", err), retCode\
    );


#define VP8_CL_CREATE_BUF(cq, bufRef, bufType, bufSize, dataPtr, altPath, retCode) \
    bufRef = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, bufSize, NULL, NULL); \
    if (dataPtr != NULL && bufRef != NULL){ \
        VP8_CL_SET_BUF(cq, bufRef, bufSize, dataPtr, altPath, retCode)\
    } \
    VP8_CL_CHECK_SUCCESS(cq, !bufRef, \
        "Error: Failed to allocate buffer. Using CPU path!\n", \
        altPath, retCode\
    ); \

#define VP8_CL_RELEASE_KERNEL(kernel) \
    {\
        if (kernel != NULL) \
            clReleaseKernel(kernel); \
        kernel = NULL;\
    }

typedef struct VP8_COMMON_CL {
    cl_device_id device_id; // compute device id
    cl_context context; // compute context
    //cl_command_queue commands; // compute command queue

    cl_program filter_program; // compute program for subpixel/bilinear filters
    cl_kernel vp8_sixtap_predict_kernel;
    size_t    vp8_sixtap_predict_kernel_size;
    cl_kernel vp8_sixtap_predict8x4_kernel;
    size_t    vp8_sixtap_predict8x4_kernel_size;
    cl_kernel vp8_sixtap_predict8x8_kernel;
    size_t    vp8_sixtap_predict8x8_kernel_size;
    cl_kernel vp8_sixtap_predict16x16_kernel;
    size_t    vp8_sixtap_predict16x16_kernel_size;

    cl_kernel vp8_bilinear_predict4x4_kernel;
    cl_kernel vp8_bilinear_predict8x4_kernel;
    cl_kernel vp8_bilinear_predict8x8_kernel;
    cl_kernel vp8_bilinear_predict16x16_kernel;

    cl_kernel vp8_filter_block2d_first_pass_kernel;
    size_t    vp8_filter_block2d_first_pass_kernel_size;
    cl_kernel vp8_filter_block2d_second_pass_kernel;
    size_t    vp8_filter_block2d_second_pass_kernel_size;

    cl_kernel vp8_filter_block2d_bil_first_pass_kernel;
    size_t    vp8_filter_block2d_bil_first_pass_kernel_size;
    cl_kernel vp8_filter_block2d_bil_second_pass_kernel;
    size_t    vp8_filter_block2d_bil_second_pass_kernel_size;

    cl_kernel vp8_memcpy_kernel;
    size_t    vp8_memcpy_kernel_size;
    cl_kernel vp8_memset_short_kernel;

    cl_program idct_program;
    cl_kernel vp8_short_inv_walsh4x4_1_kernel;
    cl_kernel vp8_short_inv_walsh4x4_1st_pass_kernel;
    cl_kernel vp8_short_inv_walsh4x4_2nd_pass_kernel;
    cl_kernel vp8_dc_only_idct_add_kernel;
    //Note that the following 2 kernels are encoder-only. Not used in decoder.
    cl_kernel vp8_short_idct4x4llm_1_kernel;
    cl_kernel vp8_short_idct4x4llm_kernel;

    cl_program loop_filter_program;

    cl_kernel vp8_loop_filter_all_edges_kernel;
    size_t vp8_loop_filter_all_edges_kernel_size;

    cl_kernel vp8_loop_filter_horizontal_edges_kernel;
    size_t vp8_loop_filter_horizontal_edges_kernel_size;
    cl_kernel vp8_loop_filter_vertical_edges_kernel;
    size_t vp8_loop_filter_vertical_edges_kernel_size;
    
    cl_kernel vp8_loop_filter_simple_all_edges_kernel;
    size_t vp8_loop_filter_simple_all_edges_kernel_size;

    cl_kernel vp8_loop_filter_simple_horizontal_edges_kernel;
    size_t vp8_loop_filter_simple_horizontal_edges_kernel_size;
    cl_kernel vp8_loop_filter_simple_vertical_edges_kernel;
    size_t vp8_loop_filter_simple_vertical_edges_kernel_size;
    
    cl_program dequant_program;
    cl_kernel vp8_dequant_dc_idct_add_kernel;
    cl_kernel vp8_dequant_idct_add_kernel;
    cl_kernel vp8_dequantize_b_kernel;

    cl_int cl_decode_initialized;
    cl_int cl_encode_initialized;
    
} VP8_COMMON_CL;

extern VP8_COMMON_CL cl_data;

#ifdef	__cplusplus
}
#endif

#endif	/* VP8_OPENCL_H */

