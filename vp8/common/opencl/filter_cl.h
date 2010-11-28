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

#define vp8_filter_block2d_first_pass_kernel_src "__kernel void vp8_filter_block2d_first_pass_kernel(__global    unsigned char *src_ptr,\
                             __global   int *output_ptr,\
                             unsigned int src_pixels_per_line,\
                             unsigned int pixel_step,\
                             unsigned int output_height,\
                             unsigned int output_width,\
                             __global   const short *vp8_filter)\
{\
    uint tid = get_global_id(0);\
\
	int out_offset,src_offset;\
    int PS2 = 2*(int)pixel_step;\
\
	out_offset = src_offset = tid/output_width;\
	out_offset = (tid - out_offset*output_width) + (out_offset * output_width);\
	src_offset = tid + (src_offset * (src_pixels_per_line - output_width));\
	int Temp = ((int)src_ptr[src_offset - PS2]         * vp8_filter[0]) +\
           ((int)src_ptr[src_offset - (int)pixel_step] * vp8_filter[1]) +\
           ((int)src_ptr[src_offset]                * vp8_filter[2]) +\
           ((int)src_ptr[src_offset + pixel_step]       * vp8_filter[3]) +\
           ((int)src_ptr[src_offset + PS2]              * vp8_filter[4]) +\
           ((int)src_ptr[src_offset + 3*(int)pixel_step]              * vp8_filter[5]) +\
           (VP8_FILTER_WEIGHT >> 1);\
\
\
    Temp = Temp >> VP8_FILTER_SHIFT;\
	\
    if (Temp < 0)\
		Temp = 0;\
	else if (Temp > 255)\
		Temp = 255;\
\
    output_ptr[out_offset] = Temp;\
}"

#endif /* FILTER_CL_H_ */
