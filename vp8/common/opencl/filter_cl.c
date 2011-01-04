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

#define SIXTAP_FILTER_LEN 6


int pass=0;

int cl_init_filter() {
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
    CL_CREATE_KERNEL(cl_data,vp8_sixtap_predict_kernel,"vp8_sixtap_predict_kernel");
    CL_CREATE_KERNEL(cl_data,vp8_block_variation_kernel,"vp8_block_variation_kernel");
    CL_CREATE_KERNEL(cl_data,vp8_sixtap_predict8x8_kernel,"vp8_sixtap_predict8x8_kernel");
    CL_CREATE_KERNEL(cl_data,vp8_sixtap_predict8x4_kernel,"vp8_sixtap_predict8x4_kernel");
    CL_CREATE_KERNEL(cl_data,vp8_sixtap_predict16x16_kernel,"vp8_sixtap_predict16x16_kernel");

    //Initialize other memory objects to null pointers
    cl_data.srcData = NULL;
    cl_data.srcAlloc = 0;
    cl_data.destData = NULL;
    cl_data.destAlloc = 0;

    return CL_SUCCESS;
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

    int err;
    size_t global = 36; //9*4

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,4,4);

    //int output1_width=4,output1_height=9;
    //int src_len = SRC_LEN(output1_width,output1_height,src_pixels_per_line);
    int src_len = SRC_LEN(4,9,src_pixels_per_line);

    CL_SIXTAP_PREDICT_EXEC(cl_data.vp8_sixtap_predict_kernel,src_ptr,src_len,
            src_pixels_per_line, xoffset,yoffset,dst_ptr,dst_pitch,global,
            dst_len,
            vp8_filter_block2d(src_ptr, dst_ptr, src_pixels_per_line,
                dst_pitch, sub_pel_filters[xoffset], sub_pel_filters[yoffset])
    );
    
    return;
}

void vp8_filter_block2d_cl
(
        unsigned char *src_ptr,
        unsigned char *output_ptr,
        unsigned int src_pixels_per_line,
        int output_pitch,
        int xoffset,
        int yoffset
)
{
    vp8_sixtap_predict_cl(src_ptr,src_pixels_per_line,xoffset,yoffset,output_ptr,output_pitch);
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

    int err;
    size_t global = 104; //13*8

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,8,8);

    //int output1_width=8,output1_height=13;
    //int src_len = SRC_LEN(output1_width,output1_height,src_pixels_per_line);
    int src_len = SRC_LEN(8,13,src_pixels_per_line);

    CL_SIXTAP_PREDICT_EXEC(cl_data.vp8_sixtap_predict8x8_kernel,src_ptr,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_filter_block2d(src_ptr, dst_ptr, src_pixels_per_line,
                dst_pitch, sub_pel_filters[xoffset], sub_pel_filters[yoffset])
    );

    return;
}

void vp8_sixtap_predict8x4_cl
(
        unsigned char *src_ptr,
        int src_pixels_per_line,
        int xoffset,
        int yoffset,
        unsigned char *dst_ptr,
        int dst_pitch
)
{

    int err;
    size_t global = 72; //9*8

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,4,8);

    //int output1_width=8,output1_height=9;
    //int src_len = SRC_LEN(output1_width,output1_height,src_pixels_per_line);
    int src_len = SRC_LEN(8,9,src_pixels_per_line);

    CL_SIXTAP_PREDICT_EXEC(cl_data.vp8_sixtap_predict8x4_kernel,src_ptr,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_filter_block2d(src_ptr, dst_ptr, src_pixels_per_line,
                dst_pitch, sub_pel_filters[xoffset], sub_pel_filters[yoffset])
    );

    return;

}

void vp8_sixtap_predict16x16_cl
(
        unsigned char *src_ptr,
        int src_pixels_per_line,
        int xoffset,
        int yoffset,
        unsigned char *dst_ptr,
        int dst_pitch
)
{

    int err;
    size_t global = 336; //21*16

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,16,16);

    //int output1_width=16,output1_height=21;
    //int src_len = SRC_LEN(output1_width,output1_height,src_pixels_per_line);
    int src_len = SRC_LEN(16,21,src_pixels_per_line);

    CL_SIXTAP_PREDICT_EXEC(cl_data.vp8_sixtap_predict16x16_kernel,src_ptr,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_filter_block2d(src_ptr, dst_ptr, src_pixels_per_line,
                dst_pitch, sub_pel_filters[xoffset], sub_pel_filters[yoffset])
    );

    return;

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