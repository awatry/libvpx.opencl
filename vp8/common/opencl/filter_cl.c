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

void cl_destroy_filter(){

    if (cl_data.filter_program)
        clReleaseProgram(cl_data.filter_program);

    CL_RELEASE_KERNEL(cl_data.vp8_block_variation_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_sixtap_predict_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_sixtap_predict8x8_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_sixtap_predict8x4_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_sixtap_predict16x16_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_bilinear_predict4x4_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_bilinear_predict8x4_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_bilinear_predict8x8_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_bilinear_predict16x16_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_memcpy_kernel);

    CL_RELEASE_KERNEL(cl_data.vp8_filter_block2d_first_pass_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_filter_block2d_second_pass_kernel);

    cl_data.filter_program = NULL;
}

int cl_init_filter() {
    int err;

    // Create the filter compute program from the file-defined source code
    if ( cl_load_program(&cl_data.filter_program, filter_cl_file_name,
            filterCompileOptions) != CL_SUCCESS )
        return CL_TRIED_BUT_FAILED;

    // Create the compute kernel in the program we wish to run
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_filter_block2d_first_pass_kernel,"vp8_filter_block2d_first_pass_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_filter_block2d_second_pass_kernel,"vp8_filter_block2d_second_pass_kernel");
    

    CL_CREATE_KERNEL(cl_data,filter_program,vp8_block_variation_kernel,"vp8_block_variation_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_sixtap_predict_kernel,"vp8_sixtap_predict_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_sixtap_predict8x8_kernel,"vp8_sixtap_predict8x8_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_sixtap_predict8x4_kernel,"vp8_sixtap_predict8x4_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_sixtap_predict16x16_kernel,"vp8_sixtap_predict16x16_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_bilinear_predict4x4_kernel,"vp8_bilinear_predict4x4_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_bilinear_predict8x4_kernel,"vp8_bilinear_predict8x4_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_bilinear_predict8x8_kernel,"vp8_bilinear_predict8x8_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_bilinear_predict16x16_kernel,"vp8_bilinear_predict16x16_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_memcpy_kernel,"vp8_memcpy_kernel");

    return CL_SUCCESS;
}

void vp8_filter_block2d_first_pass_cl(
    cl_command_queue cq,
    unsigned char *src_base,
    int src_offset,
    cl_mem int_mem,
    unsigned int src_pixels_per_line,
    unsigned int pixel_step,
    unsigned int output_height,
    unsigned int output_width,
    int filter_offset
)
{
    cl_mem src_mem;
    int err;
    //int dst_len = DST_LEN(dst_pitch,4,4);

    //int output1_width=4,output1_height=9;
    //int src_len = SRC_LEN(output1_width,output1_height,src_pixels_per_line);
    int src_len = SIXTAP_SRC_LEN(4,9,src_pixels_per_line);

    CL_CREATE_BUF( cq, src_mem, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, \
        sizeof (unsigned char) * src_len, src_base+src_offset-2, \
    ); \

    clReleaseMemObject(src_mem);
    
}

void vp8_filter_block2d_second_pass_cl
(
    cl_command_queue cq,
    cl_mem int_mem,
    int int_offset,
    unsigned char *output_base,
    int output_offset,
    int output_pitch,
    unsigned int src_pixels_per_line,
    unsigned int pixel_step,
    unsigned int output_height,
    unsigned int output_width,
    int filter_offset
)
{

}

void vp8_sixtap_predict4x4_cl
(
    cl_command_queue cq,
    unsigned char *src_base,
    int src_offset,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    unsigned char *dst_base,
    int dst_offset,
    int dst_pitch
) {

    int tmp_offset = 0;
    unsigned char *src_ptr = src_base + src_offset;
    unsigned char *dst_ptr = dst_base + dst_offset;
    
    int err;
    size_t global = 36; //9*4

    cl_mem src_mem;
    cl_mem dst_mem;

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,4,4);

    //int output1_width=4,output1_height=9;
    //int src_len = SRC_LEN(output1_width,output1_height,src_pixels_per_line);
    int src_len = SIXTAP_SRC_LEN(4,9,src_pixels_per_line);

    CL_SIXTAP_PREDICT_EXEC(cq, src_mem, dst_mem ,cl_data.vp8_sixtap_predict_kernel,(src_ptr-2*src_pixels_per_line),tmp_offset, src_len,
            src_pixels_per_line, xoffset,yoffset,dst_ptr,tmp_offset,dst_pitch,global,
            dst_len,
            vp8_sixtap_predict_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );

    return;
}

void vp8_sixtap_predict8x8_cl
(
    cl_command_queue cq,
    unsigned char *src_base,
    int src_offset,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    unsigned char *dst_base,
    int dst_offset,
    int dst_pitch
) {
    int tmp_offset = 0;
    unsigned char *src_ptr = src_base + src_offset;
    unsigned char *dst_ptr = dst_base + dst_offset;

    int err;
    size_t global = 104; //13*8
    cl_mem src_mem;
    cl_mem dst_mem;
    //Size of output data
    int dst_len = DST_LEN(dst_pitch,8,8);

    //int output1_width=8,output1_height=13;
    //int src_len = SRC_LEN(output1_width,output1_height,src_pixels_per_line);
    int src_len = SIXTAP_SRC_LEN(8,13,src_pixels_per_line);

    CL_SIXTAP_PREDICT_EXEC(cq, src_mem, dst_mem ,cl_data.vp8_sixtap_predict8x8_kernel,(src_ptr-2*src_pixels_per_line),tmp_offset,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,tmp_offset,dst_pitch,global,dst_len,
            vp8_sixtap_predict8x8_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );

    return;
}

void vp8_sixtap_predict8x4_cl
(
    cl_command_queue cq,
    unsigned char *src_base,
    int src_offset,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    unsigned char *dst_base,
    int dst_offset,
    int dst_pitch
) {

    int tmp_offset = 0;
    unsigned char *src_ptr = src_base + src_offset;
    unsigned char *dst_ptr = dst_base + dst_offset;

    int err;
    size_t global = 72; //9*8
    cl_mem src_mem;
    cl_mem dst_mem;
    //Size of output data
    int dst_len = DST_LEN(dst_pitch,4,8);

    //int output1_width=8,output1_height=9;
    //int src_len = SRC_LEN(output1_width,output1_height,src_pixels_per_line);
    int src_len = SIXTAP_SRC_LEN(8,9,src_pixels_per_line);

    CL_SIXTAP_PREDICT_EXEC(cq, src_mem, dst_mem, cl_data.vp8_sixtap_predict8x4_kernel,(src_ptr-2*src_pixels_per_line),tmp_offset,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,tmp_offset,dst_pitch,global,dst_len,
            vp8_sixtap_predict8x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );

    return;
}

void vp8_sixtap_predict16x16_cl
(
    cl_command_queue cq,
    unsigned char *src_base,
    int src_offset,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    unsigned char *dst_base,
    int dst_offset,
    int dst_pitch
) {

    int tmp_offset = 0;
    unsigned char *src_ptr = src_base + src_offset;
    unsigned char *dst_ptr = dst_base + dst_offset;

    int err;
    size_t global = 336; //21*16
    cl_mem src_mem;
    cl_mem dst_mem;
    //Size of output data
    int dst_len = DST_LEN(dst_pitch,16,16);

    //int output1_width=16,output1_height=21;
    //int src_len = SRC_LEN(output1_width,output1_height,src_pixels_per_line);
    int src_len = SIXTAP_SRC_LEN(16,21,src_pixels_per_line);

    CL_SIXTAP_PREDICT_EXEC(cq, src_mem, dst_mem, cl_data.vp8_sixtap_predict16x16_kernel,(src_ptr-2*src_pixels_per_line),tmp_offset,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,tmp_offset,dst_pitch,global,dst_len,
            vp8_sixtap_predict16x16_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );

    return;

}

void vp8_bilinear_predict4x4_cl
(
    cl_command_queue cq,
    unsigned char *src_base,
    int src_offset,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    unsigned char *dst_base,
    int dst_offset,
    int dst_pitch
) {

    int tmp_offset = 0;
    unsigned char *src_ptr = src_base + src_offset;
    unsigned char *dst_ptr = dst_base + dst_offset;

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

    cl_mem src_mem, dst_mem, int_mem;

    CL_BILINEAR_EXEC(cq, src_mem, dst_mem, int_mem, cl_data.vp8_bilinear_predict4x4_kernel,src_ptr,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_bilinear_predict4x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );
#else
    vp8_bilinear_predict4x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
#endif

}

void vp8_bilinear_predict8x8_cl
(
    cl_command_queue cq,
    unsigned char *src_base,
    int src_offset,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    unsigned char *dst_base,
    int dst_offset,
    int dst_pitch
) {

    int tmp_offset = 0;
    unsigned char *src_ptr = src_base + src_offset;
    unsigned char *dst_ptr = dst_base + dst_offset;

#if CL_BILINEAR
    int err;
    
    //global is the max of width*height for 1st and 2nd pass filters
    size_t global = 72; //9*8

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,8,8);

    //int output1_width=16,output1_height=21;
    //int src_len = SRC_LEN(output1_width,output1_height,src_pixels_per_line);
    int src_len = BIL_SRC_LEN(8,9,src_pixels_per_line);

    cl_mem src_mem, dst_mem, int_mem;

    CL_BILINEAR_EXEC(cq, src_mem, dst_mem, int_mem, cl_data.vp8_bilinear_predict8x8_kernel,src_ptr,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_bilinear_predict8x8_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );
#else
    vp8_bilinear_predict8x8_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
#endif
}

void vp8_bilinear_predict8x4_cl
(
    cl_command_queue cq,
    unsigned char *src_base,
    int src_offset,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    unsigned char *dst_base,
    int dst_offset,
    int dst_pitch
) {

    int tmp_offset = 0;
    unsigned char *src_ptr = src_base + src_offset;
    unsigned char *dst_ptr = dst_base + dst_offset;

#if CL_BILINEAR
    int err;

    //global is the max of width*height for 1st and 2nd pass filters
    size_t global = 9*4;

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,8,4);

    int src_len = BIL_SRC_LEN(4,9,src_pixels_per_line);

    cl_mem src_mem, dst_mem, int_mem;

    CL_BILINEAR_EXEC(cq, src_mem, dst_mem, int_mem, cl_data.vp8_bilinear_predict8x4_kernel,src_ptr,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_bilinear_predict8x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );
#else
    vp8_bilinear_predict8x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
#endif
}

void vp8_bilinear_predict16x16_cl
(
    cl_command_queue cq,
    unsigned char *src_base,
    int src_offset,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    unsigned char *dst_base,
    int dst_offset,
    int dst_pitch
) {

    int tmp_offset = 0;
    unsigned char *src_ptr = src_base + src_offset;
    unsigned char *dst_ptr = dst_base + dst_offset;

#if CL_BILINEAR
    int err;

    //global is the max of width*height for 1st and 2nd pass filters
    size_t global = 17*16;

    //Element counts of output/input data
    int dst_len = DST_LEN(dst_pitch,16,16);
    int src_len = BIL_SRC_LEN(16,17,src_pixels_per_line);

    cl_mem src_mem, dst_mem, int_mem;

    CL_BILINEAR_EXEC(cq, src_mem, dst_mem, int_mem, cl_data.vp8_bilinear_predict16x16_kernel,src_ptr,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_bilinear_predict16x16_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );
#else
    vp8_bilinear_predict16x16_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
#endif
}
