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

    //CL_RELEASE_KERNEL(cl_data.vp8_block_variation_kernel);
    //CL_RELEASE_KERNEL(cl_data.vp8_sixtap_predict_kernel);
    //CL_RELEASE_KERNEL(cl_data.vp8_sixtap_predict8x8_kernel);
    //CL_RELEASE_KERNEL(cl_data.vp8_sixtap_predict8x4_kernel);
    //CL_RELEASE_KERNEL(cl_data.vp8_sixtap_predict16x16_kernel);
    //CL_RELEASE_KERNEL(cl_data.vp8_bilinear_predict4x4_kernel);
    //CL_RELEASE_KERNEL(cl_data.vp8_bilinear_predict8x4_kernel);
    //CL_RELEASE_KERNEL(cl_data.vp8_bilinear_predict8x8_kernel);
    //CL_RELEASE_KERNEL(cl_data.vp8_bilinear_predict16x16_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_memcpy_kernel);

    CL_RELEASE_KERNEL(cl_data.vp8_filter_block2d_first_pass_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_filter_block2d_second_pass_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_filter_block2d_bil_first_pass_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_filter_block2d_bil_second_pass_kernel);

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
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_filter_block2d_bil_first_pass_kernel,"vp8_filter_block2d_bil_first_pass_kernel");
    CL_CREATE_KERNEL(cl_data,filter_program,vp8_filter_block2d_bil_second_pass_kernel,"vp8_filter_block2d_bil_second_pass_kernel");

    //CL_CREATE_KERNEL(cl_data,filter_program,vp8_bilinear_predict4x4_kernel,"vp8_bilinear_predict4x4_kernel");
    //CL_CREATE_KERNEL(cl_data,filter_program,vp8_bilinear_predict8x4_kernel,"vp8_bilinear_predict8x4_kernel");
    //CL_CREATE_KERNEL(cl_data,filter_program,vp8_bilinear_predict8x8_kernel,"vp8_bilinear_predict8x8_kernel");
    //CL_CREATE_KERNEL(cl_data,filter_program,vp8_bilinear_predict16x16_kernel,"vp8_bilinear_predict16x16_kernel");

    CL_CREATE_KERNEL(cl_data,filter_program,vp8_memcpy_kernel,"vp8_memcpy_kernel");

    return CL_SUCCESS;
}

void vp8_filter_block2d_first_pass_cl(
    cl_command_queue cq,
    cl_mem src_mem,
    int src_offset,
    cl_mem int_mem,
    unsigned int src_pixels_per_line,
    unsigned int int_height,
    unsigned int int_width,
    int xoffset
){
    int err;
    size_t global = int_width*int_height;

    err =  clSetKernelArg(cl_data.vp8_filter_block2d_first_pass_kernel, 0, sizeof (cl_mem), &src_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_first_pass_kernel, 1, sizeof (int), &src_offset);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_first_pass_kernel, 2, sizeof (cl_mem), &int_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_first_pass_kernel, 3, sizeof (cl_uint), &src_pixels_per_line);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_first_pass_kernel, 4, sizeof (cl_uint), &int_height);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_first_pass_kernel, 5, sizeof (cl_int), &int_width);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_first_pass_kernel, 6, sizeof (int), &xoffset);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        ,
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel( cq, cl_data.vp8_filter_block2d_first_pass_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);,
    );
}

void vp8_filter_block2d_second_pass_cl(
    cl_command_queue cq,
    cl_mem int_mem,
    int int_offset,
    cl_mem dst_mem,
    int dst_offset,
    int dst_pitch,
    unsigned int output_height,
    unsigned int output_width,
    int yoffset
){
    int err;
    size_t global = output_width*output_height;

    /* Set kernel arguments */
    err =  clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 0, sizeof (cl_mem), &int_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 1, sizeof (int), &int_offset);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 2, sizeof (cl_mem), &dst_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 3, sizeof (int), &dst_offset);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 4, sizeof (int), &dst_pitch);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 5, sizeof (int), &output_width);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 6, sizeof (int), &output_width);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 7, sizeof (int), &output_height);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 8, sizeof (int), &output_width);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_second_pass_kernel, 9, sizeof (int), &yoffset);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        ,
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel( cq, cl_data.vp8_filter_block2d_second_pass_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);,
    );
}

void vp8_sixtap_run_cl(
    cl_command_queue cq,
    cl_mem src_mem,
    cl_mem dst_mem,
    unsigned char *src_base,
    int src_offset,
    size_t src_len,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    unsigned char *dst_ptr,
    int dst_offset,
    int dst_pitch,
    size_t dst_len,
    unsigned int FData_height,
    unsigned int FData_width,
    unsigned int output_height,
    unsigned int output_width,
    int int_offset
)
{
    int err;
    cl_mem int_mem;

/*Make space for kernel input/output data. Initialize the buffer as well if needed. */
    CL_CREATE_BUF( cq, src_mem,, sizeof (unsigned char) * src_len, src_base-2, );
    CL_CREATE_BUF( cq, dst_mem,, sizeof (unsigned char) * dst_len, dst_ptr, );
    CL_CREATE_BUF( cq, int_mem,, sizeof(cl_int)*13*21, NULL, );

    vp8_filter_block2d_first_pass_cl(
        cq, src_mem, src_offset, int_mem, src_pixels_per_line,
        FData_height, FData_width, xoffset
    );

    vp8_filter_block2d_second_pass_cl(cq,int_mem,int_offset,dst_mem,dst_offset,dst_pitch,
            output_height,output_width,yoffset);

    /* Read back the result data from the device */
    err = clEnqueueReadBuffer(cq, dst_mem, CL_FALSE, 0, sizeof (unsigned char) * dst_len, dst_ptr, 0, NULL, NULL);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to read output array!\n",
        ,
    );

    clReleaseMemObject(src_mem);
    clReleaseMemObject(dst_mem);
    clReleaseMemObject(int_mem);
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

    int output_width=4, output_height=4, FData_height=9, FData_width=4;
    int int_offset = 8;
    int tmp_offset = 0;
    unsigned char *src_ptr = src_base + src_offset;
    unsigned char *dst_ptr = dst_base + dst_offset;
    
    cl_mem src_mem = NULL;
    cl_mem dst_mem = NULL;

    //Size of output to transfer
    int dst_len = DST_LEN(dst_pitch,output_height,output_width);
    int src_len = SIXTAP_SRC_LEN(FData_width,FData_height,src_pixels_per_line);

    vp8_sixtap_run_cl(cq, src_mem, dst_mem,
            (src_ptr-2*src_pixels_per_line),tmp_offset, src_len,
            src_pixels_per_line, xoffset,yoffset,dst_ptr,tmp_offset,
            dst_pitch,dst_len,FData_height,FData_width,output_height,
            output_width,int_offset
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
    int output_width=8, output_height=8, FData_height=13, FData_width=8;
    int int_offset = 16;
    int tmp_offset = 0;
    unsigned char *src_ptr = src_base + src_offset;
    unsigned char *dst_ptr = dst_base + dst_offset;

    cl_mem src_mem = NULL;
    cl_mem dst_mem = NULL;

    //Size of output to transfer
    int dst_len = DST_LEN(dst_pitch,output_height,output_width);
    int src_len = SIXTAP_SRC_LEN(FData_width,FData_height,src_pixels_per_line);

    vp8_sixtap_run_cl(cq, src_mem, dst_mem,
            (src_ptr-2*src_pixels_per_line),tmp_offset, src_len,
            src_pixels_per_line, xoffset,yoffset,dst_ptr,tmp_offset,
            dst_pitch,dst_len,FData_height,FData_width,output_height,
            output_width,int_offset
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

    int output_width=8, output_height=4, FData_height=9, FData_width=8;
    int int_offset = 16;
    int tmp_offset = 0;
    unsigned char *src_ptr = src_base + src_offset;
    unsigned char *dst_ptr = dst_base + dst_offset;

    cl_mem src_mem = NULL;
    cl_mem dst_mem = NULL;

    //Size of output to transfer
    int dst_len = DST_LEN(dst_pitch,output_height,output_width);
    int src_len = SIXTAP_SRC_LEN(FData_width,FData_height,src_pixels_per_line);

    vp8_sixtap_run_cl(cq, src_mem, dst_mem,
            (src_ptr-2*src_pixels_per_line),tmp_offset, src_len,
            src_pixels_per_line, xoffset,yoffset,dst_ptr,tmp_offset,
            dst_pitch,dst_len,FData_height,FData_width,output_height,
            output_width,int_offset
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

    int output_width=16, output_height=16, FData_height=21, FData_width=16;
    int int_offset = 32;
    int tmp_offset = 0;
    unsigned char *src_ptr = src_base + src_offset;
    unsigned char *dst_ptr = dst_base + dst_offset;

    cl_mem src_mem = NULL;
    cl_mem dst_mem = NULL;

    //Size of output to transfer
    int dst_len = DST_LEN(dst_pitch,output_height,output_width);
    int src_len = SIXTAP_SRC_LEN(FData_width,FData_height,src_pixels_per_line);

    vp8_sixtap_run_cl(cq, src_mem, dst_mem,
            (src_ptr-2*src_pixels_per_line),tmp_offset, src_len,
            src_pixels_per_line, xoffset,yoffset,dst_ptr,tmp_offset,
            dst_pitch,dst_len,FData_height,FData_width,output_height,
            output_width,int_offset
    );

    return;

}



void vp8_filter_block2d_bil_first_pass_cl(
    cl_command_queue cq,
    unsigned char *src_base,
    int src_offset,
    cl_mem int_mem,
    int src_pixels_per_line,
    int height,
    int width,
    int xoffset
)
{
    int err;
    size_t global = width*height;

    cl_mem src_mem = NULL;
    int src_len = BIL_SRC_LEN(width,height,src_pixels_per_line);

    /*Make space for kernel input/output data. Initialize the buffer as well if needed. */
    CL_CREATE_BUF(cq, src_mem, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        sizeof (unsigned char) * src_len, src_base+src_offset,
    );
    src_offset = 0; //Set to zero as long as src_mem starts at base+offset

    err =  clSetKernelArg(cl_data.vp8_filter_block2d_bil_first_pass_kernel, 0, sizeof (cl_mem), &src_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_bil_first_pass_kernel, 1, sizeof (int), &src_offset);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_bil_first_pass_kernel, 2, sizeof (cl_mem), &int_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_bil_first_pass_kernel, 3, sizeof (int), &src_pixels_per_line);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_bil_first_pass_kernel, 4, sizeof (int), &height);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_bil_first_pass_kernel, 5, sizeof (int), &width);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_bil_first_pass_kernel, 6, sizeof (int), &xoffset);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        ,
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel( cq, cl_data.vp8_filter_block2d_bil_first_pass_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);,
    );

    clReleaseMemObject(src_mem);
}


void vp8_filter_block2d_bil_second_pass_cl(
    cl_command_queue cq,
    cl_mem int_mem,
    cl_mem dst_mem,
    int dst_offset,
    int dst_pitch,
    int height,
    int width,
    int yoffset
)
{
    int err;
    size_t global = width*height;

    err =  clSetKernelArg(cl_data.vp8_filter_block2d_bil_second_pass_kernel, 0, sizeof (cl_mem), &int_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_bil_second_pass_kernel, 1, sizeof (cl_mem), &dst_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_bil_second_pass_kernel, 2, sizeof (int), &dst_offset);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_bil_second_pass_kernel, 3, sizeof (int), &dst_pitch);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_bil_second_pass_kernel, 4, sizeof (int), &height);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_bil_second_pass_kernel, 5, sizeof (int), &width);
    err |= clSetKernelArg(cl_data.vp8_filter_block2d_bil_second_pass_kernel, 6, sizeof (int), &yoffset);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        ,
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel( cq, cl_data.vp8_filter_block2d_bil_second_pass_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);,
    );

}



void vp8_bilinear_run_cl(
    cl_command_queue cq,
        cl_mem src_mem,
        cl_mem dst_mem,
        cl_mem int_mem,
        cl_kernel kernel,
        unsigned char *src_base,
        int src_offset,
        int src_len,
        int src_pixels_per_line,
        int xoffset,
        int yoffset,
        int dst_offset,
        int dst_pitch,
        size_t global
)
{
    int err;

    /*Make space for kernel input/output data. Initialize the buffer as well if needed. */
    CL_CREATE_BUF(cq, src_mem, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        sizeof (unsigned char) * src_len, src_base+src_offset,
    );

    //Remove this after untangling base_pre/base_dst stuff in mbpitch.c
    src_offset = 0;

    /* Set kernel arguments */
    err =  clSetKernelArg(kernel, 0, sizeof (cl_mem), &src_mem);
    err |= clSetKernelArg(kernel, 1, sizeof (int), &src_offset);
    err |= clSetKernelArg(kernel, 2, sizeof (int), &src_pixels_per_line);
    err |= clSetKernelArg(kernel, 3, sizeof (int), &xoffset);
    err |= clSetKernelArg(kernel, 4, sizeof (int), &yoffset);
    err |= clSetKernelArg(kernel, 5, sizeof (cl_mem), &dst_mem);
    err |= clSetKernelArg(kernel, 6, sizeof (int), &dst_offset);
    err |= clSetKernelArg(kernel, 7, sizeof (int), &dst_pitch);
    err |= clSetKernelArg(kernel, 8, sizeof (cl_mem), &int_mem);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        ,
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(cq, kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %dn",err);,
    );

    clReleaseMemObject(src_mem);
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

    int err;

    const int height = 4, width = 4;

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,height,width);

    cl_mem src_mem = NULL;
    cl_mem int_mem = NULL;
    cl_mem dst_mem = NULL;

    CL_CREATE_BUF(cq, dst_mem, CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,
        sizeof (unsigned char) * dst_len + dst_offset, dst_base,
    );

    CL_CREATE_BUF(cq, int_mem,CL_MEM_READ_WRITE,
        sizeof(cl_int)*17*16, NULL,);
    
    /* First filter 1-D horizontally... */
    vp8_filter_block2d_bil_first_pass_cl(cq, src_base, src_offset, int_mem, src_pixels_per_line, height + 1, width, xoffset);

    /* then 1-D vertically... */
    vp8_filter_block2d_bil_second_pass_cl(cq, int_mem, dst_mem, dst_offset, dst_pitch, height, width, yoffset);

    /* Read back the result data from the device */
    err = clEnqueueReadBuffer(cq, dst_mem, CL_FALSE, 0, sizeof (unsigned char) * dst_len + dst_offset, dst_base, 0, NULL, NULL);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to read output array!\n",
        ,
    );

    clReleaseMemObject(int_mem);
    clReleaseMemObject(dst_mem);
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

    int err;

    const int height = 8, width = 8;

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,height,width);

    cl_mem src_mem = NULL;
    cl_mem int_mem = NULL;
    cl_mem dst_mem = NULL;

    CL_CREATE_BUF(cq, dst_mem, CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,
        sizeof (unsigned char) * dst_len + dst_offset, dst_base,
    );

    CL_CREATE_BUF(cq, int_mem,CL_MEM_READ_WRITE,
        sizeof(cl_int)*17*16, NULL,);

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_bil_first_pass_cl(cq, src_base, src_offset, int_mem, src_pixels_per_line, height + 1, width, xoffset);

    /* then 1-D vertically... */
    vp8_filter_block2d_bil_second_pass_cl(cq, int_mem, dst_mem, dst_offset, dst_pitch, height, width, yoffset);

    /* Read back the result data from the device */
    err = clEnqueueReadBuffer(cq, dst_mem, CL_FALSE, 0, sizeof (unsigned char) * dst_len + dst_offset, dst_base, 0, NULL, NULL);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to read output array!\n",
        ,
    );

    clReleaseMemObject(int_mem);
    clReleaseMemObject(dst_mem);
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

    int err;

    const int height = 4, width = 8;

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,height,width);

    cl_mem src_mem = NULL;
    cl_mem int_mem = NULL;
    cl_mem dst_mem = NULL;

    CL_CREATE_BUF(cq, dst_mem, CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,
        sizeof (unsigned char) * dst_len + dst_offset, dst_base,
    );

    CL_CREATE_BUF(cq, int_mem,CL_MEM_READ_WRITE,
        sizeof(cl_int)*17*16, NULL,);

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_bil_first_pass_cl(cq, src_base, src_offset, int_mem, src_pixels_per_line, height + 1, width, xoffset);

    /* then 1-D vertically... */
    vp8_filter_block2d_bil_second_pass_cl(cq, int_mem, dst_mem, dst_offset, dst_pitch, height, width, yoffset);

    /* Read back the result data from the device */
    err = clEnqueueReadBuffer(cq, dst_mem, CL_FALSE, 0, sizeof (unsigned char) * dst_len + dst_offset, dst_base, 0, NULL, NULL);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to read output array!\n",
        ,
    );

    clReleaseMemObject(int_mem);
    clReleaseMemObject(dst_mem);
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

    int err;

    const int height = 16, width = 16;

    //Size of output data
    int dst_len = DST_LEN(dst_pitch,height,width);

    cl_mem src_mem = NULL;
    cl_mem int_mem = NULL;
    cl_mem dst_mem = NULL;

    CL_CREATE_BUF(cq, dst_mem, CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,
        sizeof (unsigned char) * dst_len + dst_offset, dst_base,
    );

    CL_CREATE_BUF(cq, int_mem,CL_MEM_READ_WRITE,
        sizeof(cl_int)*17*16, NULL,);

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_bil_first_pass_cl(cq, src_base, src_offset, int_mem, src_pixels_per_line, height + 1, width, xoffset);

    /* then 1-D vertically... */
    vp8_filter_block2d_bil_second_pass_cl(cq, int_mem, dst_mem, dst_offset, dst_pitch, height, width, yoffset);

    /* Read back the result data from the device */
    err = clEnqueueReadBuffer(cq, dst_mem, CL_FALSE, 0, sizeof (unsigned char) * dst_len + dst_offset, dst_base, 0, NULL, NULL);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to read output array!\n",
        ,
    );

    clReleaseMemObject(int_mem);
    clReleaseMemObject(dst_mem);
}
