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

#define CLAMP(x,min,max) if (x < min) x = min; else if ( x > max ) x = max;
#define STRINGIFY(x) #x

const char *compileOptions = "-DVP8_FILTER_WEIGHT=128 -DVP8_FILTER_SHIFT=7";

const char *vp8_filter_block2d_first_pass_kernel_src = STRINGIFY(
__kernel void vp8_filter_block2d_first_pass_kernel(
    __global unsigned char *src_ptr,
    __global int *output_ptr,
    unsigned int src_pixels_per_line,
    unsigned int pixel_step,
    unsigned int output_height,
    unsigned int output_width,
    __global short *vp8_filter)
{
    uint i = get_global_id(0);

    int src_offset;
    int Temp;
    int PS2 = 2*(int)pixel_step;
    int PS3 = 3*(int)pixel_step;

    if (i < (output_width*output_height)){
        src_offset = i + (i/output_width * (src_pixels_per_line - output_width)) + PS2;

        Temp = ((int)*(src_ptr+src_offset - PS2)      * vp8_filter[0]) +
           ((int)*(src_ptr+src_offset - (int)pixel_step) * vp8_filter[1]) +
           ((int)*(src_ptr+src_offset)                * vp8_filter[2]) +
           ((int)*(src_ptr+src_offset + pixel_step)   * vp8_filter[3]) +
           ((int)*(src_ptr+src_offset + PS2)          * vp8_filter[4]) +
           ((int)*(src_ptr+src_offset + PS3)          * vp8_filter[5]) +
           (VP8_FILTER_WEIGHT >> 1);      /* Rounding */

        /* Normalize back to 0-255 */
        Temp = Temp >> VP8_FILTER_SHIFT;
        if (Temp < 0)
            Temp = 0;
        else if ( Temp > 255 )
            Temp = 255;

        output_ptr[i] = Temp;
    }
}
);

const char *test_kernel_src="\
__kernel void test_kernel(__global    int *src_ptr,\
                             __global   int *output_ptr,\
                             unsigned int max)\
{\
    uint tid = get_global_id(0);\
    if (tid < max)\
        output_ptr[tid] = -src_ptr[tid];\
        \
    return;\
}";

#endif /* FILTER_CL_H_ */
