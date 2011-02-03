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
#include "../blockd.h"

#define SIXTAP_FILTER_LEN 6

const char *filterCompileOptions = "-Ivp8/common/opencl -DVP8_FILTER_WEIGHT=128 -DVP8_FILTER_SHIFT=7 -DFILTER_OFFSET";
const char *filter_cl_file_name = "vp8/common/opencl/filter_cl.cl";


int pass=0;

int cl_init_filter() {
    int err;

    // Create the filter compute program from the file-defined source code
    if ( cl_load_program(&cl_data.filter_program, filter_cl_file_name,
            filterCompileOptions) != CL_SUCCESS )
        return CL_TRIED_BUT_FAILED;

    // Create the compute kernel in the program we wish to run
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_sixtap_predict_kernel,"vp8_sixtap_predict_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_block_variation_kernel,"vp8_block_variation_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_sixtap_predict8x8_kernel,"vp8_sixtap_predict8x8_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_sixtap_predict8x4_kernel,"vp8_sixtap_predict8x4_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_sixtap_predict16x16_kernel,"vp8_sixtap_predict16x16_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_bilinear_predict4x4_kernel,"vp8_bilinear_predict4x4_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_bilinear_predict8x4_kernel,"vp8_bilinear_predict8x4_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_bilinear_predict8x8_kernel,"vp8_bilinear_predict8x8_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_bilinear_predict16x16_kernel,"vp8_bilinear_predict16x16_kernel");

    return CL_SUCCESS;
}

void vp8_sixtap_predict_cl
(
    MACROBLOCKD *x,
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
    int src_len = SIXTAP_SRC_LEN(4,9,src_pixels_per_line);

    if (cl_initialized != CL_SUCCESS){
        vp8_sixtap_predict_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
        return;
    }

    CL_SIXTAP_PREDICT_EXEC(x->cl_commands, cl_data.vp8_sixtap_predict_kernel,(src_ptr-2*src_pixels_per_line),src_len,
            src_pixels_per_line, xoffset,yoffset,dst_ptr,dst_pitch,global,
            dst_len,
            vp8_sixtap_predict_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );

    return;
}

void vp8_sixtap_predict8x8_cl
(
    MACROBLOCKD *x,
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
    int src_len = SIXTAP_SRC_LEN(8,13,src_pixels_per_line);

    if (cl_initialized != CL_SUCCESS){
        vp8_sixtap_predict8x8_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
        return;
    }

    CL_SIXTAP_PREDICT_EXEC(x->cl_commands, cl_data.vp8_sixtap_predict8x8_kernel,(src_ptr-2*src_pixels_per_line),src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_sixtap_predict8x8_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );

    return;
}

void vp8_sixtap_predict8x4_cl
(
    MACROBLOCKD *x,
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
    int src_len = SIXTAP_SRC_LEN(8,9,src_pixels_per_line);

    if (cl_initialized != CL_SUCCESS){
        vp8_sixtap_predict8x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
        return;
    }

    CL_SIXTAP_PREDICT_EXEC(x->cl_commands, cl_data.vp8_sixtap_predict8x4_kernel,(src_ptr-2*src_pixels_per_line),src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_sixtap_predict8x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );

    return;
}

void vp8_sixtap_predict16x16_cl
(
        MACROBLOCKD *x,
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
    int src_len = SIXTAP_SRC_LEN(16,21,src_pixels_per_line);

    if (cl_initialized != CL_SUCCESS){
        vp8_sixtap_predict16x16_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
        return;
    }

    CL_SIXTAP_PREDICT_EXEC(x->cl_commands, cl_data.vp8_sixtap_predict16x16_kernel,(src_ptr-2*src_pixels_per_line),src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_sixtap_predict16x16_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );

    return;

}

void vp8_bilinear_predict4x4_cl
(
    MACROBLOCKD *x,
    unsigned char *src_ptr,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    unsigned char *dst_ptr,
    int dst_pitch
)
{

#define CL_BILINEAR 1
#if CL_BILINEAR
    int err;

    //global is the max of width*height for 1st and 2nd pass filters
    size_t global = 20; //5*4

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,4,4);

    //int output1_width=16,output1_height=21;
    //int src_len = SRC_LEN(output1_width,output1_height,src_pixels_per_line);
    int src_len = BIL_SRC_LEN(4,5,src_pixels_per_line);

    if (cl_initialized != CL_SUCCESS){
        vp8_bilinear_predict4x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
        return;
    }

    CL_BILINEAR_EXEC(x->cl_commands, cl_data.vp8_bilinear_predict4x4_kernel,src_ptr,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_bilinear_predict4x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );
#else
    vp8_bilinear_predict4x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
#endif

}

void vp8_bilinear_predict8x8_cl
(
    MACROBLOCKD *x,
    unsigned char *src_ptr,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    unsigned char *dst_ptr,
    int dst_pitch
) {

#if CL_BILINEAR
    int err;
    
    //global is the max of width*height for 1st and 2nd pass filters
    size_t global = 72; //9*8

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,8,8);

    //int output1_width=16,output1_height=21;
    //int src_len = SRC_LEN(output1_width,output1_height,src_pixels_per_line);
    int src_len = BIL_SRC_LEN(8,9,src_pixels_per_line);

    if (cl_initialized != CL_SUCCESS){
        vp8_bilinear_predict8x8_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
        return;
    }
    
    CL_BILINEAR_EXEC(x->cl_commands, cl_data.vp8_bilinear_predict8x8_kernel,src_ptr,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_bilinear_predict8x8_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );
#else
    vp8_bilinear_predict8x8_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
#endif
}

void vp8_bilinear_predict8x4_cl
(
    MACROBLOCKD *x,
    unsigned char *src_ptr,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    unsigned char *dst_ptr,
    int dst_pitch
) {

#if CL_BILINEAR
    int err;

    //global is the max of width*height for 1st and 2nd pass filters
    size_t global = 9*4;

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,8,4);

    int src_len = BIL_SRC_LEN(4,9,src_pixels_per_line);

    if (cl_initialized != CL_SUCCESS){
        vp8_bilinear_predict8x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
        return;
    }

    CL_BILINEAR_EXEC(x->cl_commands, cl_data.vp8_bilinear_predict8x4_kernel,src_ptr,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_bilinear_predict8x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );
#else
    vp8_bilinear_predict8x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
#endif
}

void vp8_bilinear_predict16x16_cl
(
    MACROBLOCKD *x,
    unsigned char *src_ptr,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    unsigned char *dst_ptr,
    int dst_pitch
) {

#if CL_BILINEAR
    int err;

    //global is the max of width*height for 1st and 2nd pass filters
    size_t global = 17*16;

    //Element counts of output/input data
    int dst_len = DST_LEN(dst_pitch,16,16);
    int src_len = BIL_SRC_LEN(16,17,src_pixels_per_line);

    if (cl_initialized != CL_SUCCESS){
        vp8_bilinear_predict16x16_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
        return;
    }

    CL_BILINEAR_EXEC(x->cl_commands, cl_data.vp8_bilinear_predict16x16_kernel,src_ptr,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_bilinear_predict16x16_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );
#else
    vp8_bilinear_predict16x16_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
#endif
}
