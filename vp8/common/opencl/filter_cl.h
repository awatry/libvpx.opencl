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


//#define FILTER_OFFSET_BUF //Filter offset using a CL buffer and int offset
#ifndef FILTER_OFFSET_BUF
#define FILTER_OFFSET //Filter data stored as CL constant memory
#ifdef FILTER_OFFSET
#define FILTER_REF sub_pel_filters[filter_offset]
const char *compileOptions = "-DVP8_FILTER_WEIGHT=128 -DVP8_FILTER_SHIFT=7 -DFILTER_OFFSET";
#else
const char *compileOptions = "-DVP8_FILTER_WEIGHT=128 -DVP8_FILTER_SHIFT=7";
#define FILTER_REF vp8_filter
#endif
#else
#define FILTER_REF sub_pel_filters[filter_offset]
const char *compileOptions = "-DVP8_FILTER_WEIGHT=128 -DVP8_FILTER_SHIFT=7 -DFILTER_OFFSET_BUF";
#endif

const char *filter_cl_file_name = "vp8/common/opencl/filter_cl.cl";

#endif /* FILTER_CL_H_ */
