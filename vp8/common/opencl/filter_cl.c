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

int vp8_filter_block2d_first_pass_cl
(
    cl_command_queue cq,
    unsigned char *src_base,
    int src_offset,
    unsigned int src_pixels_per_line,
    unsigned int pixel_step,
    unsigned int output_height,
    unsigned int output_width,
    int filter_offset,
    cl_mem int_mem
) {

    int err;
    size_t global;

    cl_mem src_mem;

    //Calculate size of input and output arrays
    int dest_len = output_height * output_width;

    //Copy the -2*pixel_step bytes because the filter algorithm accesses negative indexes
    src_offset += (2*(int)pixel_step);
    int src_len = src_offset+(dest_len + ((dest_len-1)/output_width)*(src_pixels_per_line - output_width) + 3 * (int)pixel_step);

    //Make space for kernel input data. Initialize the buffer as well.
    CL_CREATE_BUF( cq, src_mem, ,
        sizeof (unsigned char) * src_len, src_base-(2*(int)pixel_step),
    );

    // Set kernel arguments
    err = 0;
    err = clSetKernelArg(cl_data.vp8_filter_block2d_first_pass_kernel, 0, sizeof (cl_mem), &src_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_first_pass_kernel, 1, sizeof (int), &src_offset);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_first_pass_kernel, 2, sizeof (cl_mem), &int_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_first_pass_kernel, 3, sizeof (unsigned int), &src_pixels_per_line);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_first_pass_kernel, 4, sizeof (unsigned int), &pixel_step);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_first_pass_kernel, 5, sizeof (unsigned int), &output_height);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_first_pass_kernel, 6, sizeof (unsigned int), &output_width);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_first_pass_kernel, 7, sizeof (int), &filter_offset);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        ,
        CL_TRIED_BUT_FAILED
    );

    // Execute the kernel
    global = output_width*output_height; //How many threads do we need?
#if USE_LOCAL_SIZE
    //NOTE: if local<global, global MUST be evenly divisible by local or the
    // kernel will fail.
    printf("local=%d, global=%d\n", local, global);
    err = clEnqueueNDRangeKernel(cq, cl_data.vp8_filter_block2d_first_pass_kernel, 1, NULL, &global, ((local<global)? &local: &global) , 0, NULL, NULL);
#else
    err = clEnqueueNDRangeKernel(cq, cl_data.vp8_filter_block2d_first_pass_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
#endif
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        ,
        CL_TRIED_BUT_FAILED
    );

    clReleaseMemObject(src_mem);

    return CL_SUCCESS;
}

int vp8_filter_block2d_second_pass_cl
(
    cl_command_queue cq,
    cl_mem int_mem,
    int offset,
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

    int err; //capture CL error/return codes

    //Calculate size of output array
    cl_mem dest_mem;
    int dest_len = output_offset + output_width+(output_pitch*output_height);

    size_t global;

    CL_CREATE_BUF(cq, dest_mem,,sizeof (unsigned char) * dest_len, output_base,);

    // Set kernel arguments
    err = 0;
    err = clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 0, sizeof (cl_mem), &int_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 1, sizeof (int), &offset);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 2, sizeof (cl_mem), &dest_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 3, sizeof (int), &output_offset);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 4, sizeof (int), &output_pitch);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 5, sizeof (unsigned int), &src_pixels_per_line);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 6, sizeof (unsigned int), &pixel_step);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 7, sizeof (unsigned int), &output_height);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 8, sizeof (unsigned int), &output_width);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 9, sizeof (int), &filter_offset);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        ,
        CL_TRIED_BUT_FAILED
    );

    // Execute the kernel
    global = output_width*output_height; //How many threads do we need?
#if USE_LOCAL_SIZE
    //NOTE: if local<global, global MUST be evenly divisible by local or the
    // kernel will fail.
    printf("local=%d, global=%d\n", local, global);
    err = clEnqueueNDRangeKernel(cq, cl_data.vp8_filter_block2d_second_pass_kernel, 1, NULL, &global, ((local<global)? &local: &global) , 0, NULL, NULL);
#else
    err = clEnqueueNDRangeKernel(cq, cl_data.vp8_filter_block2d_second_pass_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
#endif
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        ,
        CL_TRIED_BUT_FAILED
    );

    // Read back the result data from the device
    err = clEnqueueReadBuffer(cq, dest_mem, CL_FALSE, 0, sizeof (unsigned char) * dest_len, output_base, 0, NULL, NULL);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to read output array!\n",
        ,
        CL_TRIED_BUT_FAILED
    );

    clReleaseMemObject(dest_mem);
    
    return CL_SUCCESS;
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
)
{

    int ret;
    int err;

    CL_FINISH(cq);
    vp8_sixtap_predict_c(src_base+src_offset, src_pixels_per_line,xoffset,yoffset,dst_base+dst_offset, dst_pitch);
    return;

    cl_mem int_mem;
    CL_CREATE_BUF(cq, int_mem,, sizeof(cl_int)*13*16,NULL,);

    //The return of the C comparison...
    int i;
    int dest_len = dst_offset + 4+(dst_pitch*4);
    unsigned char c_output[dest_len];

    vp8_sixtap_predict_c( src_base+src_offset,
        src_pixels_per_line,
        xoffset,
        yoffset,
        c_output,
        dst_pitch
    );

    /* First filter 1-D horizontally... */
    ret = vp8_filter_block2d_first_pass_cl(cq, src_base, src_offset, src_pixels_per_line, 1, 9, 4, xoffset, int_mem);

    /* then filter vertically... */
    ret = vp8_filter_block2d_second_pass_cl(cq, int_mem, 8, dst_base, dst_offset, dst_pitch, 4, 4, 4, 4, yoffset);

    CL_FINISH(cq);

    for(i = 0; i < dest_len; i++){
        printf("i=%d, expected %d, got %d\n", i, c_output[i], dst_base[dst_offset+i]);
    }

    clReleaseMemObject(int_mem);
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
)
{

    int ret;
    int err;

    CL_FINISH(cq);
    vp8_sixtap_predict8x8_c(src_base+src_offset, src_pixels_per_line,xoffset,yoffset,dst_base+dst_offset, dst_pitch);
    return;


    cl_mem int_mem;
    CL_CREATE_BUF(cq, int_mem,, sizeof(cl_int)*13*16,NULL,);

    /* First filter 1-D horizontally... */
    ret = vp8_filter_block2d_first_pass_cl(cq, src_base, src_offset, src_pixels_per_line, 1, 13, 8, xoffset, int_mem);

    /* then filter vertically... */
    ret = vp8_filter_block2d_second_pass_cl(cq, int_mem, 16, dst_base, dst_offset, dst_pitch, 8, 8, 8, 8, yoffset);

    clReleaseMemObject(int_mem);
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
)
{

    int ret;
    int err;

    CL_FINISH(cq);
    vp8_sixtap_predict8x4_c(src_base+src_offset, src_pixels_per_line,xoffset,yoffset,dst_base+dst_offset, dst_pitch);
    return;

    cl_mem int_mem;
    CL_CREATE_BUF(cq, int_mem,, sizeof(cl_int)*13*16,NULL,);

    /* First filter 1-D horizontally... */
    ret = vp8_filter_block2d_first_pass_cl(cq, src_base, src_offset, src_pixels_per_line, 1, 9, 8, xoffset, int_mem);

    /* then filter vertically... */
    ret = vp8_filter_block2d_second_pass_cl(cq, int_mem, 16, dst_base, dst_offset, dst_pitch, 8, 8, 4, 8, yoffset);

    clReleaseMemObject(int_mem);
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
)
{
    int ret;
    int err;

    CL_FINISH(cq);
    vp8_sixtap_predict16x16_c(src_base+src_offset, src_pixels_per_line,xoffset,yoffset,dst_base+dst_offset, dst_pitch);
    return;

    cl_mem int_mem;
    CL_CREATE_BUF(cq, int_mem,, sizeof(cl_int)*13*16,NULL,);

    /* First filter 1-D horizontally... */
    ret = vp8_filter_block2d_first_pass_cl(cq, src_base, src_offset, src_pixels_per_line, 1, 21, 16, xoffset, int_mem);

    /* then filter vertically... */
    ret = vp8_filter_block2d_second_pass_cl(cq, int_mem, 32, dst_base, dst_offset, dst_pitch, 16, 16, 16, 16, yoffset);

    clReleaseMemObject(int_mem);

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
