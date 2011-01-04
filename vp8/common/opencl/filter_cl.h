/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef FILTER_CL_H_
#define FILTER_CL_H_

#include "vp8_opencl.h"

#define VP8_FILTER_WEIGHT 128
#define VP8_FILTER_SHIFT  7

#define REGISTER_FILTER 1
#define CLAMP(x,min,max) if (x < min) x = min; else if ( x > max ) x = max;
#define PRE_CALC_PIXEL_STEPS 1
#define PRE_CALC_SRC_INCREMENT 1

#if PRE_CALC_PIXEL_STEPS
#define PS2 two_pixel_steps
#define PS3 three_pixel_steps
#else
#define PS2 2*(int)pixel_step
#define PS3 3*(int)pixel_step
#endif

#if REGISTER_FILTER
#define FILTER0 filter0
#define FILTER1 filter1
#define FILTER2 filter2
#define FILTER3 filter3
#define FILTER4 filter4
#define FILTER5 filter5
#else
#define FILTER0 vp8_filter[0]
#define FILTER1 vp8_filter[1]
#define FILTER2 vp8_filter[2]
#define FILTER3 vp8_filter[3]
#define FILTER4 vp8_filter[4]
#define FILTER5 vp8_filter[5]
#endif

#if PRE_CALC_SRC_INCREMENT
#define SRC_INCREMENT src_increment
#else
#define SRC_INCREMENT (src_pixels_per_line - output_width)
#endif

extern void vp8_filter_block2d
(
    unsigned char  *src_ptr,
    unsigned char  *output_ptr,
    unsigned int src_pixels_per_line,
    int output_pitch,
    const short  *HFilter,
    const short  *VFilter
);

extern void vp8_sixtap_predict8x8_c
(
    unsigned char  *src_ptr,
    int  src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    unsigned char *dst_ptr,
    int  dst_pitch
);

void vp8_sixtap_predict8x4_c
(
    unsigned char  *src_ptr,
    int  src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    unsigned char *dst_ptr,
    int  dst_pitch
);

extern void vp8_sixtap_predict16x16_c
(
    unsigned char  *src_ptr,
    int  src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    unsigned char *dst_ptr,
    int  dst_pitch
);


static const int bilinear_filters[8][2] = {
    { 128, 0},
    { 112, 16},
    { 96, 32},
    { 80, 48},
    { 64, 64},
    { 48, 80},
    { 32, 96},
    { 16, 112}
};

static const short sub_pel_filters[8][6] = {
    { 0, 0, 128, 0, 0, 0}, /* note that 1/8 pel positions are just as per alpha -0.5 bicubic */
    { 0, -6, 123, 12, -1, 0},
    { 2, -11, 108, 36, -8, 1}, /* New 1/4 pel 6 tap filter */
    { 0, -9, 93, 50, -6, 0},
    { 3, -16, 77, 77, -16, 3}, /* New 1/2 pel 6 tap filter */
    { 0, -6, 50, 93, -9, 0},
    { 1, -8, 36, 108, -11, 2}, /* New 1/4 pel 6 tap filter */
    { 0, -1, 12, 123, -6, 0},
};

#define FILTER_REF sub_pel_filters[filter_offset]
const char *compileOptions = "-DVP8_FILTER_WEIGHT=128 -DVP8_FILTER_SHIFT=7 -DFILTER_OFFSET";

const char *filter_cl_file_name = "vp8/common/opencl/filter_cl.cl";

//Copy the -2*pixel_step (and ps*3) bytes because the filter algorithm
//accesses negative indexes
#define SRC_LEN(out_width,out_height,src_px) (out_width*out_height + ((out_width*out_height-1)/out_width)*(src_px - out_width) + 5)
#define DST_LEN(dst_pitch,dst_height,dst_width) (dst_pitch*dst_height + dst_width)

#define CL_SIXTAP_PREDICT_EXEC(kernel,src_ptr,src_len, src_pixels_per_line, \
xoffset,yoffset,dst_ptr,dst_pitch,thread_count,dst_len,altPath) \
    if (cl_initialized != CL_SUCCESS){ \
        if (cl_initialized == CL_NOT_INITIALIZED){ \
            cl_initialized = cl_init_filter(); \
        } \
        if (cl_initialized != CL_SUCCESS){ \
            altPath; \
            return; \
        } \
    } \
\
    /*Make space for kernel input/output data. Initialize the buffer as well if needed. */ \
    CL_ENSURE_BUF_SIZE(cl_data.srcData, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, \
        sizeof (unsigned char) * src_len, cl_data.srcAlloc, src_ptr-2, \
    ); \
\
    CL_ENSURE_BUF_SIZE(cl_data.destData, CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR, \
        sizeof (unsigned char) * dst_len, cl_data.destAlloc, dst_ptr, \
    ); \
\
    /* Set kernel arguments */ \
    err = 0; \
    err = clSetKernelArg(kernel, 0, sizeof (cl_mem), &cl_data.srcData); \
    err |= clSetKernelArg(kernel, 1, sizeof (int), &src_pixels_per_line); \
    err |= clSetKernelArg(kernel, 2, sizeof (int), &xoffset); \
    err |= clSetKernelArg(kernel, 3, sizeof (int), &yoffset); \
    err |= clSetKernelArg(kernel, 4, sizeof (cl_mem), &cl_data.destData); \
    err |= clSetKernelArg(kernel, 5, sizeof (int), &dst_pitch); \
    CL_CHECK_SUCCESS( err != CL_SUCCESS, \
        "Error: Failed to set kernel arguments!\n", \
        altPath, \
    ); \
\
    /* Execute the kernel */ \
    err = clEnqueueNDRangeKernel(cl_data.commands, kernel, 1, NULL, &global, NULL , 0, NULL, NULL); \
    CL_CHECK_SUCCESS( err != CL_SUCCESS, \
        "Error: Failed to execute kernel!\n", \
        printf("err = %d\n",err);altPath, \
    ); \
\
    /* Read back the result data from the device */ \
    err = clEnqueueReadBuffer(cl_data.commands, cl_data.destData, CL_FALSE, 0, sizeof (unsigned char) * dst_len, dst_ptr, 0, NULL, NULL); \
    CL_CHECK_SUCCESS(err != CL_SUCCESS, \
        "Error: Failed to read output array!\n", \
        altPath, \
    );

//#end define CL_SIXTAP_PREDICT_EXEC


#endif /* FILTER_CL_H_ */
