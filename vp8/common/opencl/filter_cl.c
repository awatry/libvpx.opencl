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

    //Don't allow re-initialization without proper teardown first.
    if (cl_data.filter_program != NULL)
        return CL_SUCCESS;

    //Initialize the CL context
    if (cl_init() != CL_SUCCESS)
        return CL_TRIED_BUT_FAILED;

    // Create the filter compute program from the file-defined source code
    CL_LOAD_PROGRAM(cl_data.filter_program, filter_cl_file_name, filterCompileOptions);

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

    //cl_data.filter_block2d_first_pass_kernel = clCreateKernel(cl_data.filter_program, "vp8_filter_block2d_first_pass_kernel", &err);
    //cl_data.filter_block2d_second_pass_kernel = clCreateKernel(cl_data.filter_program, "vp8_filter_block2d_second_pass_kernel", &err);
    //if (!cl_data.filter_block2d_first_pass_kernel ||
    //        !cl_data.filter_block2d_second_pass_kernel ||
    //        err != CL_SUCCESS) {
    //    printf("Error: Failed to create compute kernel!\n");
    //    return CL_TRIED_BUT_FAILED;
    //}

    //Initialize other memory objects to null pointers
    cl_data.srcData = NULL;
    cl_data.srcAlloc = 0;
    cl_data.destData = NULL;
    cl_data.destAlloc = 0;

    return CL_SUCCESS;
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
    int src_len = SIXTAP_SRC_LEN(4,9,src_pixels_per_line);

    //size_t i;
    //unsigned char c_output[src_len];

    //printf("4x4: src_ptr = %p, src_len = %d, dst_ptr = %p, dst_len = %d\n",src_ptr,src_len,dst_ptr,dst_len);
    //memcpy(c_output,dst_ptr,dst_len);
    //vp8_sixtap_predict_c(src_ptr,src_pixels_per_line,xoffset,yoffset,c_output,dst_pitch);

    CL_SIXTAP_PREDICT_EXEC(cl_data.vp8_sixtap_predict_kernel,(src_ptr-2*src_pixels_per_line),src_len,
            src_pixels_per_line, xoffset,yoffset,dst_ptr,dst_pitch,global,
            dst_len,
            vp8_sixtap_predict_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );
    //clFinish(cl_data.commands);

    //for (i=0; i < dst_len; i++){
    //    if (c_output[i] != dst_ptr[i]){
    //        printf("c_output[%d] (%d) != dst_ptr[%d] (%d)\n",i,c_output[i],i,dst_ptr[i]);
    //        exit(1);
    //    }
    //}

    return;
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
    int src_len = SIXTAP_SRC_LEN(8,13,src_pixels_per_line);

    //size_t i;
    //unsigned char c_output[src_len];

    //printf("8x8: src_ptr = %p, src_len = %d, dst_ptr = %p, dst_len = %d\n",src_ptr,src_len,dst_ptr,dst_len);
    //memcpy(c_output,dst_ptr,dst_len);
    //vp8_sixtap_predict8x8_c(src_ptr,src_pixels_per_line,xoffset,yoffset,c_output,dst_pitch);

    CL_SIXTAP_PREDICT_EXEC(cl_data.vp8_sixtap_predict8x8_kernel,(src_ptr-2*src_pixels_per_line),src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_sixtap_predict8x8_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );
    //clFinish(cl_data.commands);

    //for (i=0; i < dst_len; i++){
    //    if (c_output[i] != dst_ptr[i]){
    //        printf("c_output[%d] (%d) != dst_ptr[%d] (%d)\n",i,c_output[i],i,dst_ptr[i]);
    //        exit(1);
    //    }
    //}

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
    int src_len = SIXTAP_SRC_LEN(8,9,src_pixels_per_line);

    //size_t i;
    //unsigned char c_output[src_len];

    //printf("8x4: src_ptr = %p, src_len = %d, dst_ptr = %p, dst_len = %d\n",src_ptr,src_len,dst_ptr,dst_len);
    //for (i=0; i < src_len; i++){
    //    printf("initial src[%d] = %d\n",i,src_ptr[i]);
    //}
    //for (i = 0; i < dst_len; i++){
    //    printf("initial dst_ptr[%d] = %d\n",i,dst_ptr[i]);
    //}
    //memcpy(c_output,dst_ptr,dst_len);
    //vp8_sixtap_predict8x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,c_output,dst_pitch);

    CL_SIXTAP_PREDICT_EXEC(cl_data.vp8_sixtap_predict8x4_kernel,(src_ptr-2*src_pixels_per_line),src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_sixtap_predict8x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );

    //clFinish(cl_data.commands);

    //for (i=0; i < dst_len; i++){
    //    if (c_output[i] != dst_ptr[i]){
    //        printf("c_output[%d] (%d) != dst_ptr[%d] (%d)\n",i,c_output[i],i,dst_ptr[i]);
    //        exit(1);
    //    }
    //}

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
    int src_len = SIXTAP_SRC_LEN(16,21,src_pixels_per_line);

    //size_t i;
    //unsigned char c_output[src_len];

    //printf("16x16: src_ptr = %p, src_len = %d, dst_ptr = %p, dst_len = %d\n",src_ptr,src_len,dst_ptr,dst_len);
    //memcpy(c_output,dst_ptr,dst_len);
    //vp8_sixtap_predict16x16_c(src_ptr,src_pixels_per_line,xoffset,yoffset,c_output,dst_pitch);

    CL_SIXTAP_PREDICT_EXEC(cl_data.vp8_sixtap_predict16x16_kernel,(src_ptr-2*src_pixels_per_line),src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_sixtap_predict16x16_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );
    //clFinish(cl_data.commands);

    //for (i=0; i < dst_len; i++){
    //    if (c_output[i] != dst_ptr[i]){
    //        printf("c_output[%d] (%d) != dst_ptr[%d] (%d)\n",i,c_output[i],i,dst_ptr[i]);
    //        exit(1);
    //    }
    //}

    return;

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

    size_t i;
    unsigned char c_output[src_len];

    printf("bilinear 4x4: src_ptr = %p, src_len = %d, dst_ptr = %p, dst_len = %d\n",src_ptr,src_len,dst_ptr,dst_len);
    memcpy(c_output,dst_ptr,dst_len);
    //vp8_bilinear_predict4x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,c_output,dst_pitch);

    CL_BILINEAR_EXEC(cl_data.vp8_bilinear_predict4x4_kernel,src_ptr,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_bilinear_predict4x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );

    clFinish(cl_data.commands);

    for (i=0; i < dst_len; i++){
        if (c_output[i] != dst_ptr[i]){
            printf("c_output[%d] (%d) != dst_ptr[%d] (%d)\n",i,c_output[i],i,dst_ptr[i]);
            //exit(1);
        }
    }
#else
    vp8_bilinear_predict4x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
#endif

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

#if CL_BILINEAR
    int err;
    
    //global is the max of width*height for 1st and 2nd pass filters
    size_t global = 72; //9*8

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,8,8);

    //int output1_width=16,output1_height=21;
    //int src_len = SRC_LEN(output1_width,output1_height,src_pixels_per_line);
    int src_len = BIL_SRC_LEN(8,9,src_pixels_per_line);

    size_t i;
    unsigned char c_output[src_len];

    printf("bilinear 8x8: src_ptr = %p, src_len = %d, dst_ptr = %p, dst_len = %d\n",src_ptr,src_len,dst_ptr,dst_len);
    memcpy(c_output,dst_ptr,dst_len);
    vp8_bilinear_predict8x8_c(src_ptr,src_pixels_per_line,xoffset,yoffset,c_output,dst_pitch);

    CL_BILINEAR_EXEC(cl_data.vp8_bilinear_predict8x8_kernel,src_ptr,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_bilinear_predict8x8_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );

    clFinish(cl_data.commands);

    for (i=0; i < dst_len; i++){
        if (c_output[i] != dst_ptr[i]){
            printf("c_output[%d] (%d) != dst_ptr[%d] (%d)\n",i,c_output[i],i,dst_ptr[i]);
            //exit(1);
        }
    }

#else
    vp8_bilinear_predict8x8_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
#endif
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

#if CL_BILINEAR
    int err;

    //global is the max of width*height for 1st and 2nd pass filters
    size_t global = 9*4;

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,8,4);

    int src_len = BIL_SRC_LEN(4,9,src_pixels_per_line);

    size_t i;
    unsigned char c_output[src_len];

    printf("bilinear 8x4: src_ptr = %p, src_len = %d, dst_ptr = %p, dst_len = %d\n",src_ptr,src_len,dst_ptr,dst_len);
    memcpy(c_output,dst_ptr,dst_len);
    vp8_bilinear_predict8x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,c_output,dst_pitch);

    CL_BILINEAR_EXEC(cl_data.vp8_bilinear_predict8x4_kernel,src_ptr,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_bilinear_predict8x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );

    clFinish(cl_data.commands);

/*
    for (i=0; i < dst_len; i++){
        if (c_output[i] != dst_ptr[i]){
            printf("c_output[%d] (%d) != dst_ptr[%d] (%d)\n",i,c_output[i],i,dst_ptr[i]);
            exit(1);
        }
    }
*/

#else
    vp8_bilinear_predict8x4_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
#endif
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

#if CL_BILINEAR
    int err;

    //global is the max of width*height for 1st and 2nd pass filters
    size_t global = 17*16;

    //Element counts of output/input data
    int dst_len = DST_LEN(dst_pitch,16,16);
    int src_len = BIL_SRC_LEN(16,17,src_pixels_per_line);

    size_t i;
    unsigned char c_output[src_len];

    printf("initial src_ptr = %p\n",src_ptr);

/*
    for (i = 0; i < dst_len; i++){
        printf("input[%d] = %d\n", i, src_ptr[i]);
    }
*/

    printf("bilinear 16x16: src_ptr = %p, src_len = %d, dst_ptr = %p, dst_len = %d, pitch = %d\n",src_ptr,src_len,dst_ptr,dst_len,dst_pitch);
    memcpy(c_output,dst_ptr,dst_len*sizeof(unsigned char));
    printf("memcpy done\n");
    vp8_bilinear_predict16x16_c(src_ptr,src_pixels_per_line,xoffset,yoffset,c_output,dst_pitch);
    printf("C version complete\n");

    CL_BILINEAR_EXEC(cl_data.vp8_bilinear_predict16x16_kernel,src_ptr,src_len,
            src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch,global,dst_len,
            vp8_bilinear_predict16x16_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch)
    );

    clFinish(cl_data.commands);

    printf("16x16 compare\n");
    printf("dst_ptr[%d] = %d\n", dst_len-1,dst_ptr[dst_len-1]);
    printf("c_output[%d] = %d\n", dst_len-1,c_output[dst_len-1]);


    for (i=0; i < dst_len; i++){
        if (c_output[i] != dst_ptr[i]){
            printf("c_output[%d] (%d) != dst_ptr[%d] (%d)\n",i,c_output[i],i,dst_ptr[i]);
            //exit(1);
        }
    }
    printf("16x16 compare completed\n");
    //exit(1);

#else
    vp8_bilinear_predict16x16_c(src_ptr,src_pixels_per_line,xoffset,yoffset,dst_ptr,dst_pitch);
#endif
}
