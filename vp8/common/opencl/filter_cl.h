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

const char *vp8_filter_block2d_first_pass_kernel_src="__kernel void vp8_filter_block2d_first_pass_kernel(\
    __global unsigned char *src_ptr, \
    __global int *output_ptr, \
    unsigned int src_pixels_per_line, \
    unsigned int pixel_step, \
    unsigned int output_height, \
    unsigned int output_width, \
    __global short *vp8_filter) \
{\
    uint tid = get_global_id(0);\
\
    int src_offset;\
\
    if (tid < (output_width*output_height)){\
        src_offset = tid + (tid/output_width * (src_pixels_per_line - output_width));\
        int Temp = \
           ((int)*(src_ptr+src_offset - 2*(int)pixel_step) * vp8_filter[0]) +\
           ((int)*(src_ptr+src_offset - (int)pixel_step) * vp8_filter[1]) +\
           ((int)src_ptr[src_offset]                * vp8_filter[2]) +\
           ((int)src_ptr[src_offset + pixel_step]       * vp8_filter[3]) +\
           ((int)src_ptr[src_offset + 2*(int)pixel_step]              * vp8_filter[4]) +\
           ((int)src_ptr[src_offset + 3*(int)pixel_step]              * vp8_filter[5]) +\
           (128 >> 1);\
\
        Temp = Temp >> 7;\
\
        if (Temp < 0)\
            Temp = 0;\
        else if (Temp > 255)\
            Temp = 255;\
\
        output_ptr[tid] = Temp;\
    }\
}";


//for (i = 0; i < output_height*output_width; i++){
//	src_offset = i + (i/output_width * SRC_INCREMENT);
//        Temp = ((int)*(src_ptr+src_offset - PS2)         * FILTER0) +
//           ((int)*(src_ptr+src_offset - (int)pixel_step) * FILTER1) +
//           ((int)*(src_ptr+src_offset)                * FILTER2) +
//           ((int)*(src_ptr+src_offset + pixel_step)       * FILTER3) +
//           ((int)*(src_ptr+src_offset + PS2)              * FILTER4) +
//           ((int)*(src_ptr+src_offset + PS3)              * FILTER5) +
//           (VP8_FILTER_WEIGHT >> 1);      /* Rounding */

        /* Normalize back to 0-255 */
//        Temp = Temp >> VP8_FILTER_SHIFT;
//        CLAMP(Temp, 0, 255);
        //printf("Clamped Temp=%d\n",Temp);

//        output_ptr[i] = Temp;


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
