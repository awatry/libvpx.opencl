/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include <stdlib.h>

//ACW: Remove me after debugging.
#include <stdio.h>
#include <string.h>

#include "filter_cl.h"

#ifdef FILTER_OFFSET_BUF
#define SIXTAP_FILTER_LEN 48
#else
#define SIXTAP_FILTER_LEN 6
#endif

int pass=0;

#define USE_LOCAL_SIZE 0
#if USE_LOCAL_SIZE
size_t local;
#endif

int cl_init_filter_block2d() {
    char *kernel_src;
    int err;

    //Initialize the CL context
    if (cl_init() != CL_SUCCESS)
        return CL_TRIED_BUT_FAILED;

    // Create the compute program from the file-defined source code
    kernel_src = cl_read_file(filter_cl_file_name);
    if (kernel_src != NULL){
        printf("creating program from source file\n");
        cl_data.program = clCreateProgramWithSource(cl_data.context, 1, &kernel_src, NULL, &err);
        free(kernel_src);
    } else {
        cl_destroy();
        printf("Couldn't find OpenCL source files. \nUsing software path.\n");
        return CL_TRIED_BUT_FAILED;
    }

    if (!cl_data.program) {
        printf("Error: Couldn't compile program\n");
        return CL_TRIED_BUT_FAILED;
    }

    // Build the program executable
    err = clBuildProgram(cl_data.program, 0, NULL, compileOptions, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[20480];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(cl_data.program, cl_data.device_id, CL_PROGRAM_BUILD_LOG, sizeof (buffer), &buffer, &len);
        printf("Compile output: %s\n", buffer);
        return CL_TRIED_BUT_FAILED;
    }

    // Create the compute kernel in the program we wish to run
    cl_data.filter_block2d_first_pass_kernel = clCreateKernel(cl_data.program, "vp8_filter_block2d_first_pass_kernel", &err);
    cl_data.filter_block2d_second_pass_kernel = clCreateKernel(cl_data.program, "vp8_filter_block2d_second_pass_kernel", &err);
    if (!cl_data.filter_block2d_first_pass_kernel || 
            !cl_data.filter_block2d_second_pass_kernel ||
            err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        return CL_TRIED_BUT_FAILED;
    }

#if USE_LOCAL_SIZE
    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof (local), &local, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        return CL_TRIED_BUT_FAILED;
    }
#endif

    //Filter size doesn't change. Allocate buffer once, and just replace contents
    //on each kernel execution.
#ifdef FILTER_OFFSET_BUF
    cl_data.filterData = clCreateBuffer(cl_data.context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof (short) * SIXTAP_FILTER_LEN, sub_pel_filters, &err);
#else
    cl_data.filterData = clCreateBuffer(cl_data.context, CL_MEM_READ_ONLY, sizeof (short) * SIXTAP_FILTER_LEN, NULL, NULL);
#endif

    if (!cl_data.filterData){
        printf("Error: Failed to allocate filter buffer\n");
        return CL_TRIED_BUT_FAILED;
    }

    //Initialize other memory objects to null pointers
    cl_data.srcData = NULL;
    cl_data.srcAlloc = 0;
    cl_data.destData = NULL;
    cl_data.destAlloc = 0;
    cl_data.intData = NULL;
    cl_data.intAlloc = 0;
    cl_data.intSize = 0;

    return CL_SUCCESS;
}

int vp8_filter_block2d_first_pass_cl
(
        unsigned char *src_ptr,
        unsigned int src_pixels_per_line,
        unsigned int pixel_step,
        unsigned int output_height,
        unsigned int output_width,
        int filter_offset
) {

    int err;
    size_t global;

    //Calculate size of input and output arrays
    int dest_len = output_height * output_width;

    //Copy the -2*pixel_step bytes because the filter algorithm accesses negative indexes
    int src_len = (dest_len + ((dest_len-1)/output_width)*(src_pixels_per_line - output_width) + 5 * (int)pixel_step);

    if (cl_initialized != CL_SUCCESS){
        if (cl_initialized == CL_NOT_INITIALIZED){
            cl_initialized = cl_init_filter_block2d();
        }
        if (cl_initialized != CL_SUCCESS){
            return CL_TRIED_BUT_FAILED;
        }
    }

    // Create source/intermediate buffers in device memory
    //First, make space for the output of the first pass filter
    cl_data.intSize = sizeof (int) * dest_len;
    CL_ENSURE_BUF_SIZE(cl_data.intData, CL_MEM_READ_WRITE, cl_data.intSize,
        cl_data.intAlloc, NULL,
    );

    //Make space for kernel input data. Initialize the buffer as well.
    CL_ENSURE_BUF_SIZE(cl_data.srcData, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        sizeof (unsigned char) * src_len, cl_data.srcAlloc, src_ptr-(2*(int)pixel_step),
    );

    // Set kernel arguments
    err = 0;
    err = clSetKernelArg(cl_data.filter_block2d_first_pass_kernel, 0, sizeof (cl_mem), &cl_data.srcData);
    err |= clSetKernelArg(cl_data.filter_block2d_first_pass_kernel, 1, sizeof (cl_mem), &cl_data.intData);
    err |= clSetKernelArg(cl_data.filter_block2d_first_pass_kernel, 2, sizeof (unsigned int), &src_pixels_per_line);
    err |= clSetKernelArg(cl_data.filter_block2d_first_pass_kernel, 3, sizeof (unsigned int), &pixel_step);
    err |= clSetKernelArg(cl_data.filter_block2d_first_pass_kernel, 4, sizeof (unsigned int), &output_height);
    err |= clSetKernelArg(cl_data.filter_block2d_first_pass_kernel, 5, sizeof (unsigned int), &output_width);
#ifdef FILTER_OFFSET_BUF
    err |= clSetKernelArg(cl_data.filter_block2d_first_pass_kernel, 6, sizeof (int), &filter_offset);
    err |= clSetKernelArg(cl_data.filter_block2d_first_pass_kernel, 7, sizeof (cl_mem), &cl_data.filterData);
#else
    err |= clSetKernelArg(cl_data.filter_block2d_first_pass_kernel, 6, sizeof (int), &filter_offset);
#endif
    CL_CHECK_SUCCESS( err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        ,
        CL_TRIED_BUT_FAILED
    );
    //printf("Set kernel arguments\n");

    // Execute the kernel
    global = output_width*output_height; //How many threads do we need?
#if USE_LOCAL_SIZE
    //NOTE: if local<global, global MUST be evenly divisible by local or the
    //      kernel will fail.
    printf("local=%d, global=%d\n", local, global);
    err = clEnqueueNDRangeKernel(cl_data.commands, cl_data.filter_block2d_first_pass_kernel, 1, NULL, &global, ((local<global)? &local: &global) , 0, NULL, NULL);
#else
    err = clEnqueueNDRangeKernel(cl_data.commands, cl_data.filter_block2d_first_pass_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
#endif
    CL_CHECK_SUCCESS( err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        ,
        CL_TRIED_BUT_FAILED
    );

    return CL_SUCCESS;
}

int vp8_filter_block2d_second_pass_cl
(
        int offset,
        unsigned char *output_ptr,
        int output_pitch,
        unsigned int src_pixels_per_line,
        unsigned int pixel_step,
        unsigned int output_height,
        unsigned int output_width,
        int filter_offset
        ) {

    int err; //capture CL error/return codes

    //Calculate size of output array
    int dest_len = output_width+(output_pitch*output_height);

    size_t global;

    if (cl_initialized != CL_SUCCESS){
        return CL_TRIED_BUT_FAILED;
    }

    CL_ENSURE_BUF_SIZE(cl_data.destData, CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,
        sizeof (unsigned char) * dest_len, cl_data.destAlloc, output_ptr,
    );

    // Set kernel arguments
    err = 0;
    err = clSetKernelArg(cl_data.filter_block2d_second_pass_kernel, 0, sizeof (cl_mem), &cl_data.intData);
    err |= clSetKernelArg(cl_data.filter_block2d_second_pass_kernel, 1, sizeof (int), &offset);
    err |= clSetKernelArg(cl_data.filter_block2d_second_pass_kernel, 2, sizeof (cl_mem), &cl_data.destData);
    err |= clSetKernelArg(cl_data.filter_block2d_second_pass_kernel, 3, sizeof (int), &output_pitch);
    err |= clSetKernelArg(cl_data.filter_block2d_second_pass_kernel, 4, sizeof (unsigned int), &src_pixels_per_line);
    err |= clSetKernelArg(cl_data.filter_block2d_second_pass_kernel, 5, sizeof (unsigned int), &pixel_step);
    err |= clSetKernelArg(cl_data.filter_block2d_second_pass_kernel, 6, sizeof (unsigned int), &output_height);
    err |= clSetKernelArg(cl_data.filter_block2d_second_pass_kernel, 7, sizeof (unsigned int), &output_width);
#if defined(FILTER_OFFSET_BUF)
    err |= clSetKernelArg(cl_data.filter_block2d_second_pass_kernel, 8, sizeof (int), &filter_offset);
    err |= clSetKernelArg(cl_data.filter_block2d_second_pass_kernel, 9, sizeof (cl_mem), &cl_data.filterData);
#else
    err |= clSetKernelArg(cl_data.filter_block2d_second_pass_kernel, 8, sizeof (int), &filter_offset);
#endif
    CL_CHECK_SUCCESS(err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        ,
        CL_TRIED_BUT_FAILED
    );

    // Execute the kernel
    global = output_width*output_height; //How many threads do we need?
#if USE_LOCAL_SIZE
    //NOTE: if local<global, global MUST be evenly divisible by local or the
    //      kernel will fail.
    printf("local=%d, global=%d\n", local, global);
    err = clEnqueueNDRangeKernel(cl_data.commands, cl_data.filter_block2d_second_pass_kernel, 1, NULL, &global, ((local<global)? &local: &global) , 0, NULL, NULL);
#else
    err = clEnqueueNDRangeKernel(cl_data.commands, cl_data.filter_block2d_second_pass_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
#endif
    CL_CHECK_SUCCESS(err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        ,
        CL_TRIED_BUT_FAILED
    );

    // Read back the result data from the device
    err = clEnqueueReadBuffer(cl_data.commands, cl_data.destData, CL_FALSE, 0, sizeof (unsigned char) * dest_len, output_ptr, 0, NULL, NULL);
    CL_CHECK_SUCCESS(err != CL_SUCCESS,
        "Error: Failed to read output array!\n",
        , 
        CL_TRIED_BUT_FAILED
    );

    //clEnqueueBarrier(cl_data.commands);
    //clFinish(cl_data.commands);

    return CL_SUCCESS;
}

void vp8_filter_block2d_cl
(
        unsigned char *src_ptr,
        unsigned char *output_ptr,
        unsigned int src_pixels_per_line,
        int output_pitch,
        int xoffset,
        int yoffset
        ) {
    
    int ret;

    /* First filter 1-D horizontally... */
    ret = vp8_filter_block2d_first_pass_cl(src_ptr - (2 * src_pixels_per_line), src_pixels_per_line, 1, 9, 4, xoffset);

    /* then filter vertically... */
    ret |= vp8_filter_block2d_second_pass_cl(8, output_ptr, output_pitch, 4, 4, 4, 4, yoffset);

    if (ret != CL_SUCCESS){
        vp8_filter_block2d(src_ptr, output_ptr, src_pixels_per_line,
                output_pitch, sub_pel_filters[xoffset], sub_pel_filters[yoffset]);
        return;
    }
}

void vp8_block_variation_cl
(
        unsigned char *src_ptr,
        int src_pixels_per_line,
        int *HVar,
        int *VVar
        ) {

    int i, j;
    unsigned char *Ptr = src_ptr;

    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            *HVar += abs((int) Ptr[j] - (int) Ptr[j + 1]);
            *VVar += abs((int) Ptr[j] - (int) Ptr[j + src_pixels_per_line]);
        }

        Ptr += src_pixels_per_line;
    }
}

void vp8_sixtap_predict_cl
(
        unsigned char *src_ptr,
        int src_pixels_per_line,
        int xoffset,
        int yoffset,
        unsigned char *dst_ptr,
        int dst_pitch
        ) {

    vp8_filter_block2d_cl(src_ptr, dst_ptr, src_pixels_per_line, dst_pitch, xoffset, yoffset);
}

void vp8_sixtap_predict8x8_cl
(
        unsigned char *src_ptr,
        int src_pixels_per_line,
        int xoffset,
        int yoffset,
        unsigned char *dst_ptr,
        int dst_pitch
        ) {

    int ret;

    /* First filter 1-D horizontally... */
    ret = vp8_filter_block2d_first_pass_cl(src_ptr - (2 * src_pixels_per_line), src_pixels_per_line, 1, 13, 8, xoffset);

    /* then filter vertically... */
    ret |= vp8_filter_block2d_second_pass_cl(16, dst_ptr, dst_pitch, 8, 8, 8, 8, yoffset);

    if (ret != CL_SUCCESS){
        vp8_sixtap_predict8x8_c(src_ptr, src_pixels_per_line, xoffset, yoffset,
                dst_ptr, dst_pitch);
        return;
    }
}

void vp8_sixtap_predict8x4_cl
(
        unsigned char *src_ptr,
        int src_pixels_per_line,
        int xoffset,
        int yoffset,
        unsigned char *dst_ptr,
        int dst_pitch
        ) {

    int ret;

    /* First filter 1-D horizontally... */
    ret = vp8_filter_block2d_first_pass_cl(src_ptr - (2 * src_pixels_per_line), src_pixels_per_line, 1, 9, 8, xoffset);

    /* then filter vertically... */
    ret |= vp8_filter_block2d_second_pass_cl(16, dst_ptr, dst_pitch, 8, 8, 4, 8, yoffset);

    if (ret != CL_SUCCESS){
        vp8_sixtap_predict8x4_c(src_ptr, src_pixels_per_line, xoffset, yoffset,
                dst_ptr, dst_pitch);
        return;
    }

}

void vp8_sixtap_predict16x16_cl
(
        unsigned char *src_ptr,
        int src_pixels_per_line,
        int xoffset,
        int yoffset,
        unsigned char *dst_ptr,
        int dst_pitch
        ) {
    int ret;

    /* First filter 1-D horizontally... */
    ret = vp8_filter_block2d_first_pass_cl(src_ptr - (2 * src_pixels_per_line), src_pixels_per_line, 1, 21, 16, xoffset);

    /* then filter vertically... */
    ret |= vp8_filter_block2d_second_pass_cl(32, dst_ptr, dst_pitch, 16, 16, 16, 16, yoffset);

    if (ret != CL_SUCCESS){
        vp8_sixtap_predict16x16_c(src_ptr, src_pixels_per_line, xoffset, yoffset,
                dst_ptr, dst_pitch);
        return;
    }

}

/****************************************************************************
 *
 *  ROUTINE       : filter_block2d_bil_first_pass
 *
 *  INPUTS        : UINT8  *src_ptr          : Pointer to source block.
 *                  UINT32 src_pixels_per_line : Stride of input block.
 *                  UINT32 pixel_step        : Offset between filter input samples (see notes).
 *                  UINT32 output_height     : Input block height.
 *                  UINT32 output_width      : Input block width.
 *                  INT32  *vp8_filter          : Array of 2 bi-linear filter taps.
 *
 *  OUTPUTS       : INT32 *output_ptr        : Pointer to filtered block.
 *
 *  RETURNS       : void
 *
 *  FUNCTION      : Applies a 1-D 2-tap bi-linear filter to the source block in
 *                  either horizontal or vertical direction to produce the
 *                  filtered output block. Used to implement first-pass
 *                  of 2-D separable filter.
 *
 *  SPECIAL NOTES : Produces INT32 output to retain precision for next pass.
 *                  Two filter taps should sum to VP8_FILTER_WEIGHT.
 *                  pixel_step defines whether the filter is applied
 *                  horizontally (pixel_step=1) or vertically (pixel_step=stride).
 *                  It defines the offset required to move from one input
 *                  to the next.
 *
 ****************************************************************************/
void vp8_filter_block2d_bil_first_pass_cl
(
        unsigned char *src_ptr,
        unsigned short *output_ptr,
        unsigned int src_pixels_per_line,
        int pixel_step,
        unsigned int output_height,
        unsigned int output_width,
        const int *vp8_filter
        ) {
    unsigned int i, j;

    for (i = 0; i < output_height; i++) {
        for (j = 0; j < output_width; j++) {
            /* Apply bilinear filter */
            output_ptr[j] = (((int) src_ptr[0] * vp8_filter[0]) +
                    ((int) src_ptr[pixel_step] * vp8_filter[1]) +
                    (VP8_FILTER_WEIGHT / 2)) >> VP8_FILTER_SHIFT;
            src_ptr++;
        }

        /* Next row... */
        src_ptr += src_pixels_per_line - output_width;
        output_ptr += output_width;
    }
}

/****************************************************************************
 *
 *  ROUTINE       : filter_block2d_bil_second_pass
 *
 *  INPUTS        : INT32  *src_ptr          : Pointer to source block.
 *                  UINT32 src_pixels_per_line : Stride of input block.
 *                  UINT32 pixel_step        : Offset between filter input samples (see notes).
 *                  UINT32 output_height     : Input block height.
 *                  UINT32 output_width      : Input block width.
 *                  INT32  *vp8_filter          : Array of 2 bi-linear filter taps.
 *
 *  OUTPUTS       : UINT16 *output_ptr       : Pointer to filtered block.
 *
 *  RETURNS       : void
 *
 *  FUNCTION      : Applies a 1-D 2-tap bi-linear filter to the source block in
 *                  either horizontal or vertical direction to produce the
 *                  filtered output block. Used to implement second-pass
 *                  of 2-D separable filter.
 *
 *  SPECIAL NOTES : Requires 32-bit input as produced by filter_block2d_bil_first_pass.
 *                  Two filter taps should sum to VP8_FILTER_WEIGHT.
 *                  pixel_step defines whether the filter is applied
 *                  horizontally (pixel_step=1) or vertically (pixel_step=stride).
 *                  It defines the offset required to move from one input
 *                  to the next.
 *
 ****************************************************************************/
void vp8_filter_block2d_bil_second_pass_cl
(
        unsigned short *src_ptr,
        unsigned char *output_ptr,
        int output_pitch,
        unsigned int src_pixels_per_line,
        unsigned int pixel_step,
        unsigned int output_height,
        unsigned int output_width,
        const int *vp8_filter
        ) {
    unsigned int i, j;
    int Temp;

    for (i = 0; i < output_height; i++) {
        for (j = 0; j < output_width; j++) {
            /* Apply filter */
            Temp = ((int) src_ptr[0] * vp8_filter[0]) +
                    ((int) src_ptr[pixel_step] * vp8_filter[1]) +
                    (VP8_FILTER_WEIGHT / 2);
            output_ptr[j] = (unsigned int) (Temp >> VP8_FILTER_SHIFT);
            src_ptr++;
        }

        /* Next row... */
        src_ptr += src_pixels_per_line - output_width;
        output_ptr += output_pitch;
    }
}

/****************************************************************************
 *
 *  ROUTINE       : filter_block2d_bil
 *
 *  INPUTS        : UINT8  *src_ptr          : Pointer to source block.
 *                  UINT32 src_pixels_per_line : Stride of input block.
 *                  INT32  *HFilter         : Array of 2 horizontal filter taps.
 *                  INT32  *VFilter         : Array of 2 vertical filter taps.
 *
 *  OUTPUTS       : UINT16 *output_ptr       : Pointer to filtered block.
 *
 *  RETURNS       : void
 *
 *  FUNCTION      : 2-D filters an input block by applying a 2-tap
 *                  bi-linear filter horizontally followed by a 2-tap
 *                  bi-linear filter vertically on the result.
 *
 *  SPECIAL NOTES : The largest block size can be handled here is 16x16
 *
 ****************************************************************************/
void vp8_filter_block2d_bil_cl
(
        unsigned char *src_ptr,
        unsigned char *output_ptr,
        unsigned int src_pixels_per_line,
        unsigned int dst_pitch,
        const int *HFilter,
        const int *VFilter,
        int Width,
        int Height
        ) {

    unsigned short FData[17 * 16]; /* Temp data buffer used in filtering */

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_bil_first_pass_cl(src_ptr, FData, src_pixels_per_line, 1, Height + 1, Width, HFilter);

    /* then 1-D vertically... */
    vp8_filter_block2d_bil_second_pass_cl(FData, output_ptr, dst_pitch, Width, Width, Height, Width, VFilter);
}

void vp8_bilinear_predict4x4_cl
(
        unsigned char *src_ptr,
        int src_pixels_per_line,
        int xoffset,
        int yoffset,
        unsigned char *dst_ptr,
        int dst_pitch
        ) {
    const int *HFilter;
    const int *VFilter;

    HFilter = bilinear_filters[xoffset];
    VFilter = bilinear_filters[yoffset];

    vp8_filter_block2d_bil_cl(src_ptr, dst_ptr, src_pixels_per_line, dst_pitch, HFilter, VFilter, 4, 4);

}

void vp8_bilinear_predict8x8_cl
(
        unsigned char *src_ptr,
        int src_pixels_per_line,
        int xoffset,
        int yoffset,
        unsigned char *dst_ptr,
        int dst_pitch
        ) {
    const int *HFilter;
    const int *VFilter;

    HFilter = bilinear_filters[xoffset];
    VFilter = bilinear_filters[yoffset];

    vp8_filter_block2d_bil_cl(src_ptr, dst_ptr, src_pixels_per_line, dst_pitch, HFilter, VFilter, 8, 8);

}

void vp8_bilinear_predict8x4_cl
(
        unsigned char *src_ptr,
        int src_pixels_per_line,
        int xoffset,
        int yoffset,
        unsigned char *dst_ptr,
        int dst_pitch
        ) {
    const int *HFilter;
    const int *VFilter;

    HFilter = bilinear_filters[xoffset];
    VFilter = bilinear_filters[yoffset];

    vp8_filter_block2d_bil_cl(src_ptr, dst_ptr, src_pixels_per_line, dst_pitch, HFilter, VFilter, 8, 4);

}

void vp8_bilinear_predict16x16_cl
(
        unsigned char *src_ptr,
        int src_pixels_per_line,
        int xoffset,
        int yoffset,
        unsigned char *dst_ptr,
        int dst_pitch
        ) {
    const int *HFilter;
    const int *VFilter;

    HFilter = bilinear_filters[xoffset];
    VFilter = bilinear_filters[yoffset];

    vp8_filter_block2d_bil_cl(src_ptr, dst_ptr, src_pixels_per_line, dst_pitch, HFilter, VFilter, 16, 16);
}