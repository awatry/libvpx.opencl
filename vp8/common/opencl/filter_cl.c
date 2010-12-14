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

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "filter_cl.h"

#define SIXTAP_FILTER_LEN 6
#define MAX_NUM_PLATFORMS 4
#define BLOCK_HEIGHT_WIDTH 4

int cl_initialized = 0;
int pass=0;
cl_device_id device_id; // compute device id
cl_context context; // compute context
cl_command_queue commands; // compute command queue
cl_program program; // compute program
cl_kernel kernel; // compute kernel
cl_mem srcData;
cl_mem destData;
cl_mem filterData;

#define USE_LOCAL_SIZE 0
#if USE_LOCAL_SIZE
    size_t local;
#endif


int cl_init_filter_block2d_first_pass() {
    // Connect to a compute device
    int err;
    cl_platform_id platform_ids[MAX_NUM_PLATFORMS];
    cl_uint num_found;
    err = clGetPlatformIDs(MAX_NUM_PLATFORMS, platform_ids, &num_found);

    if (err != CL_SUCCESS) {
        printf("Couldn't query platform IDs\n");
        return EXIT_FAILURE;
    }
    if (num_found == 0) {
        printf("No platforms found\n");
        return EXIT_FAILURE;
    }
    //printf("Found %d platforms\n", num_found);

    //Favor the GPU, but fall back to any other available device if necessary
    err = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
        if (err != CL_SUCCESS) {
            printf("Error: Failed to create a device group!\n");
            return EXIT_FAILURE;
        }
    }

    // Create the compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands || err != CL_SUCCESS) {
        printf("Error: Failed to create a command queue!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the header defined source code
    //printf("source: %s\n", vp8_filter_block2d_first_pass_kernel_src);
    program = clCreateProgramWithSource(context, 1, &vp8_filter_block2d_first_pass_kernel_src, NULL, &err);
    if (!program) {
        printf("Error: Couldn't compile program\n");
        return EXIT_FAILURE;
    }
    //printf("Created Program\n");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, compileOptions, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof (buffer), buffer, &len);
        printf("Compile output: %s\n", buffer);
        return EXIT_FAILURE;
    }
    //printf("Built executable\n");

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "vp8_filter_block2d_first_pass_kernel", &err);
    //kernel = clCreateKernel(program, "test_kernel", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        return EXIT_FAILURE;
    }
    //printf("Created kernel\n");

#if USE_LOCAL_SIZE
    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof (local), &local, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        return EXIT_FAILURE;
    }
    //printf("local=%d\n",local);
#endif

    //Filter size doesn't change. Allocate buffer once, and just replace contents
    //on each kernel execution.
    filterData = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof (short) * SIXTAP_FILTER_LEN, NULL, NULL);
    if (!filterData){
        printf("Error: Failed to allocate filter buffer\n");
        return EXIT_FAILURE;
    }

    cl_initialized = 1;

    return 0;
}

/**
 *
 */
void cl_destroy() {

    //printf("Free filterData\n");
    if (filterData){
        clReleaseMemObject(filterData);
        filterData = NULL;
    }

    //Release the objects that we've allocated on the GPU
    if (program)
        clReleaseProgram(program);
    if (kernel)
        clReleaseKernel(kernel);
    if (commands)
        clReleaseCommandQueue(commands);
    if (context)
        clReleaseContext(context);

    program = NULL;
    kernel = NULL;
    commands = NULL;
    context = NULL;

    cl_initialized = 0;

    return;
}

void vp8_filter_block2d_first_pass_cl
(
        unsigned char *src_ptr,
        int *output_ptr,
        unsigned int src_pixels_per_line,
        unsigned int pixel_step,
        unsigned int output_height,
        unsigned int output_width,
        const short *vp8_filter
        ) {

    int err;
#define SHOW_OUTPUT 0
#if SHOW_OUTPUT
    int j;
#endif
    size_t global;

    //Calculate size of input and output arrays
    int dest_len = output_height * output_width;

    //Copy the -2*pixel_step bytes because the filter algorithm accesses negative indexes
    int src_len = (dest_len + ((dest_len-1)/output_width)*(src_pixels_per_line - output_width) + 5 * (int)pixel_step);

    if (!cl_initialized){
        cl_init_filter_block2d_first_pass();
        if (!cl_initialized){
            vp8_filter_block2d_first_pass(src_ptr, output_ptr, src_pixels_per_line, pixel_step, output_height, output_width, vp8_filter);
            return;
        }
    }

    // Create input/output buffers in device memory
    srcData = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof (unsigned char) * src_len, NULL, NULL);
    destData = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof (int) * dest_len, NULL, NULL);

    //printf("srcData=%p\tdestData=%p\tfilterData=%p\n",srcData,destData,filterData);
    if (!srcData || !destData) {
        printf("Error: Failed to allocate device memory. Using CPU path!\n");

        //Free up whatever objects were successfully allocated
        if (srcData){
            clReleaseMemObject(srcData);
            srcData=NULL;
        }
        if (destData){
            clReleaseMemObject(destData);
            destData = NULL;
        }

        cl_destroy();
        vp8_filter_block2d_first_pass(src_ptr, output_ptr, src_pixels_per_line, pixel_step, output_height, output_width, vp8_filter);
    }
    //printf("Created buffers on device\n");

    // Copy input and filter data to device
    err = clEnqueueWriteBuffer(commands, srcData, CL_FALSE, 0,
            sizeof (unsigned char) * src_len, src_ptr-(2*(int)pixel_step), 0, NULL, NULL);

    err = clEnqueueWriteBuffer(commands, filterData, CL_FALSE, 0,
            sizeof (short) * SIXTAP_FILTER_LEN, vp8_filter, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clFinish(commands); //Wait for commands to finish so pointers are usable again.
        printf("Error: Failed to write to source array!\n");
        vp8_filter_block2d_first_pass(src_ptr, output_ptr, src_pixels_per_line, pixel_step, output_height, output_width, vp8_filter);
        return;
    }

    // Set kernel arguments
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof (cl_mem), &srcData);
    err |= clSetKernelArg(kernel, 1, sizeof (cl_mem), &destData);
    err |= clSetKernelArg(kernel, 2, sizeof (unsigned int), &src_pixels_per_line);
    err |= clSetKernelArg(kernel, 3, sizeof (unsigned int), &pixel_step);
    err |= clSetKernelArg(kernel, 4, sizeof (unsigned int), &output_height);
    err |= clSetKernelArg(kernel, 5, sizeof (unsigned int), &output_width);
    err |= clSetKernelArg(kernel, 6, sizeof (cl_mem), &filterData);
    if (err != CL_SUCCESS) {
        clFinish(commands);
        printf("Error: Failed to set kernel arguments! %d\n", err);
        vp8_filter_block2d_first_pass(src_ptr, output_ptr, src_pixels_per_line, pixel_step, output_height, output_width, vp8_filter);
        return;
    }
    //printf("Set kernel arguments\n");

    // Execute the kernel
    global = output_width*output_height; //How many threads do we need?
#if USE_LOCAL_SIZE
    //NOTE: if local<global, global MUST be evenly divisible by local or the
    //      kernel will fail.
    printf("local=%d, global=%d\n", local, global);
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, ((local<global)? &local: &global) , 0, NULL, NULL);
#else
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
#endif
    if (err) {
        clFinish(commands);
        printf("Error: Failed to execute kernel!\n");
        vp8_filter_block2d_first_pass(src_ptr, output_ptr, src_pixels_per_line, pixel_step, output_height, output_width, vp8_filter);
        return;
    }
    //printf("Kernel queued\n");

    // Read back the result data from the device
    err = clEnqueueReadBuffer(commands, destData, CL_FALSE, 0, sizeof (int) * dest_len, output_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clFinish(commands);
        printf("Error: Failed to read output array! %d\n", err);
        vp8_filter_block2d_first_pass(src_ptr, output_ptr, src_pixels_per_line, pixel_step, output_height, output_width, vp8_filter);
        return;
    }

    clFinish(commands);
    
#if SHOW_OUTPUT

    //Run C code so that we can compare output for correctness.
    int c_output[output_height*output_width];
    pass++;
    vp8_filter_block2d_first_pass(src_ptr, c_output, src_pixels_per_line, pixel_step, output_height, output_width, vp8_filter);


    for (j=0; j < dest_len; j++){
        if (output_ptr[j] != c_output[j]){
            printf("pass %d, dest_len %d, output_ptr[%d] = %d, c[%d]=%d\n", pass, dest_len, j, output_ptr[j], j, c_output[j]);
            //exit(1);
        }
    }
#endif

    // Release memory that is only used once
    clReleaseMemObject(srcData);
    clReleaseMemObject(destData);

    return;

}

void vp8_filter_block2d_second_pass_cl
(
        int *src_ptr,
        unsigned char *output_ptr,
        int output_pitch,
        unsigned int src_pixels_per_line,
        unsigned int pixel_step,
        unsigned int output_height,
        unsigned int output_width,
        const short *vp8_filter
        ) {
#define NESTED_FILTER 1
#if NESTED_FILTER
	unsigned int i, j;
#else
	unsigned int i;
	int out_offset,src_offset;
#endif

	int  Temp;

#if REGISTER_FILTER
    short filter0 = vp8_filter[0];
    short filter1 = vp8_filter[1];
    short filter2 = vp8_filter[2];
    short filter3 = vp8_filter[3];
    short filter4 = vp8_filter[4];
    short filter5 = vp8_filter[5];
#endif

#if PRE_CALC_PIXEL_STEPS
    int two_pixel_steps = ((int)pixel_step) << 1;
    int three_pixel_steps = two_pixel_steps + (int)pixel_step;
#endif

#if PRE_CALC_SRC_INCREMENT
    unsigned int src_increment = src_pixels_per_line - output_width;
#endif
#if NESTED_FILTER
    for (i = 0; i < output_height; i++)
    {
        for (j = 0; j < output_width; j++)
        {
            /* Apply filter */
            Temp = ((int)src_ptr[-1*PS2] * FILTER0) +
                   ((int)src_ptr[-1*(int)pixel_step] * FILTER1) +
                   ((int)src_ptr[0]                  * FILTER2) +
                   ((int)src_ptr[pixel_step]         * FILTER3) +
                   ((int)src_ptr[PS2]       * FILTER4) +
                   ((int)src_ptr[PS3]       * FILTER5) +
                   (VP8_FILTER_WEIGHT >> 1);   /* Rounding */
#else
	for (i = 0; i < output_height * output_width; i++){
            src_offset = out_offset = i/output_width;
            src_offset = i + (src_offset * SRC_INCREMENT);
            out_offset = i%output_width + (out_offset * output_pitch);
            /* Apply filter */
            Temp = ((int)src_ptr[src_offset - PS2] * FILTER0) +
               ((int)src_ptr[src_offset -(int)pixel_step] * FILTER1) +
               ((int)src_ptr[src_offset]                  * FILTER2) +
               ((int)src_ptr[src_offset + pixel_step]         * FILTER3) +
               ((int)src_ptr[src_offset + PS2]       * FILTER4) +
               ((int)src_ptr[src_offset + PS3]       * FILTER5) +
               (VP8_FILTER_WEIGHT >> 1);   /* Rounding */
#endif
            /* Normalize back to 0-255 */
            Temp = Temp >> VP8_FILTER_SHIFT;
            CLAMP(Temp, 0, 255);

#if NESTED_FILTER
            output_ptr[j] = (unsigned char)Temp;
            src_ptr++;
        }

        /* Start next row */
        src_ptr    += src_pixels_per_line - output_width;
        output_ptr += output_pitch;
#else
        output_ptr[out_offset] = (unsigned char)Temp;
#endif
    }
}

void vp8_filter_block2d_cl
(
        unsigned char *src_ptr,
        unsigned char *output_ptr,
        unsigned int src_pixels_per_line,
        int output_pitch,
        const short *HFilter,
        const short *VFilter
        ) {
    int FData[9 * 4]; /* Temp data buffer used in filtering */

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_first_pass_cl(src_ptr - (2 * src_pixels_per_line), FData, src_pixels_per_line, 1, 9, 4, HFilter);

    /* then filter vertically... */
    vp8_filter_block2d_second_pass_cl(FData + 8, output_ptr, output_pitch, 4, 4, 4, 4, VFilter);
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
    const short *HFilter;
    const short *VFilter;

    HFilter = sub_pel_filters[xoffset]; /* 6 tap */
    VFilter = sub_pel_filters[yoffset]; /* 6 tap */

    vp8_filter_block2d_cl(src_ptr, dst_ptr, src_pixels_per_line, dst_pitch, HFilter, VFilter);
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
    const short *HFilter;
    const short *VFilter;
    int FData[13 * 16]; /* Temp data buffer used in filtering */

    HFilter = sub_pel_filters[xoffset]; /* 6 tap */
    VFilter = sub_pel_filters[yoffset]; /* 6 tap */

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_first_pass_cl(src_ptr - (2 * src_pixels_per_line), FData, src_pixels_per_line, 1, 13, 8, HFilter);


    /* then filter vertically... */
    vp8_filter_block2d_second_pass_cl(FData + 16, dst_ptr, dst_pitch, 8, 8, 8, 8, VFilter);

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
    const short *HFilter;
    const short *VFilter;
    int FData[13 * 16]; /* Temp data buffer used in filtering */

    HFilter = sub_pel_filters[xoffset]; /* 6 tap */
    VFilter = sub_pel_filters[yoffset]; /* 6 tap */

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_first_pass_cl(src_ptr - (2 * src_pixels_per_line), FData, src_pixels_per_line, 1, 9, 8, HFilter);


    /* then filter vertically... */
    vp8_filter_block2d_second_pass_cl(FData + 16, dst_ptr, dst_pitch, 8, 8, 4, 8, VFilter);

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
    const short *HFilter;
    const short *VFilter;
    int FData[21 * 24]; /* Temp data buffer used in filtering */


    HFilter = sub_pel_filters[xoffset]; /* 6 tap */
    VFilter = sub_pel_filters[yoffset]; /* 6 tap */

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_first_pass_cl(src_ptr - (2 * src_pixels_per_line), FData, src_pixels_per_line, 1, 21, 16, HFilter);

    /* then filter vertically... */
    vp8_filter_block2d_second_pass_cl(FData + 32, dst_ptr, dst_pitch, 16, 16, 16, 16, VFilter);

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
