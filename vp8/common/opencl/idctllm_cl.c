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

#include "idct_cl.h"
#include "idctllm_cl.h"

int cl_init_idct() {
    int err;

    // Create the filter compute program from the file-defined source code
    if (cl_load_program(&cl_data.idct_program, idctllm_cl_file_name,
            idctCompileOptions) != CL_SUCCESS)
        return CL_TRIED_BUT_FAILED;

    // Create the compute kernel in the program we wish to run
    CL_CREATE_KERNEL(cl_data,idct_program,vp8_short_inv_walsh4x4_1_kernel,"vp8_short_inv_walsh4x4_1_kernel");
    CL_CREATE_KERNEL(cl_data,idct_program,vp8_short_inv_walsh4x4_1st_pass_kernel,"vp8_short_inv_walsh4x4_1st_pass_kernel");
    CL_CREATE_KERNEL(cl_data,idct_program,vp8_short_inv_walsh4x4_2nd_pass_kernel,"vp8_short_inv_walsh4x4_2nd_pass_kernel");
    CL_CREATE_KERNEL(cl_data,idct_program,vp8_dc_only_idct_add_kernel,"vp8_dc_only_idct_add_kernel");

    ////idct4x4llm kernels are only useful for the encoder
    //CL_CREATE_KERNEL(cl_data,idct_program,vp8_short_idct4x4llm_1_kernel,"vp8_short_idct4x4llm_1_kernel");
    //CL_CREATE_KERNEL(cl_data,idct_program,vp8_short_idct4x4llm_kernel,"vp8_short_idct4x4llm_kernel");

    return CL_SUCCESS;
}

#define max(x,y) (x > y ? x: y)
//#define NO_CL

/* Only useful for encoder... Untested... */
void vp8_short_idct4x4llm_cl(BLOCKD *b, int pitch)
{
    int err;

    short *input = b->dqcoeff_base + b->dqcoeff_offset;
    short *output = &b->diff_base[b->diff_offset];

    //1 instance for now. This should be split into 2-pass * 4 thread.
    size_t global = 1;

    if (cl_initialized != CL_SUCCESS){
        vp8_short_idct4x4llm_c(input,output,pitch);
        return;
    }

    CL_ENSURE_BUF_SIZE(b->cl_commands, cl_data.srcData, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
            sizeof(short)*16, cl_data.srcAlloc, input,
            vp8_short_idct4x4llm_c(input,output,pitch)
    );

    CL_ENSURE_BUF_SIZE(b->cl_commands, cl_data.destData,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
            sizeof(short)*(4+(pitch/2)*3), cl_data.destAlloc, output,
            vp8_short_idct4x4llm_c(input,output,pitch)
    );

    //Set arguments and run kernel
    err = 0;
    err = clSetKernelArg(cl_data.vp8_short_idct4x4llm_kernel, 0, sizeof (cl_mem), &cl_data.srcData);
    err |= clSetKernelArg(cl_data.vp8_short_idct4x4llm_kernel, 1, sizeof (cl_mem), &cl_data.destData);
    err |= clSetKernelArg(cl_data.vp8_short_idct4x4llm_kernel, 2, sizeof (int), &pitch);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        vp8_short_idct4x4llm_c(input,output,pitch),
    );
    
    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(b->cl_commands, cl_data.vp8_short_idct4x4llm_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);
        vp8_short_idct4x4llm_c(input,output,pitch),
    );

    /* Read back the result data from the device */
    err = clEnqueueReadBuffer(b->cl_commands, cl_data.destData, CL_FALSE, 0, sizeof(short)*(4+pitch/2*3), output, 0, NULL, NULL);
    CL_CHECK_SUCCESS(b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read output array!\n",
        vp8_short_idct4x4llm_c(input,output,pitch),
    );

    return;
}

/* Only useful for encoder... Untested... */
void vp8_short_idct4x4llm_1_cl(BLOCKD *b, int pitch)
{
    int err;
    size_t global = 4;

    short *input = b->dqcoeff_base + b->dqcoeff_offset;
    short *output = &b->diff_base[b->diff_offset];

    if (cl_initialized != CL_SUCCESS){
        vp8_short_idct4x4llm_1_c(input,output,pitch);
        return;
    }

    printf("vp8_short_idct4x4llm_1_cl\n");

    CL_ENSURE_BUF_SIZE(b->cl_commands, cl_data.srcData, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
            sizeof(short), cl_data.srcAlloc, input,
            vp8_short_idct4x4llm_1_c(input,output,pitch)
    );

    CL_ENSURE_BUF_SIZE(b->cl_commands, cl_data.destData,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
            sizeof(short)*(4+(pitch/2)*3), cl_data.destAlloc, output,
            vp8_short_idct4x4llm_1_c(input,output,pitch)
    );

    //Set arguments and run kernel
    err = 0;
    err = clSetKernelArg(cl_data.vp8_short_idct4x4llm_1_kernel, 0, sizeof (cl_mem), &cl_data.srcData);
    err |= clSetKernelArg(cl_data.vp8_short_idct4x4llm_1_kernel, 1, sizeof (cl_mem), &cl_data.destData);
    err |= clSetKernelArg(cl_data.vp8_short_idct4x4llm_1_kernel, 2, sizeof (int), &pitch);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        vp8_short_idct4x4llm_1_c(input,output,pitch),
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(b->cl_commands, cl_data.vp8_short_idct4x4llm_1_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);
        vp8_short_idct4x4llm_1_c(input,output,pitch),
    );

    /* Read back the result data from the device */
    err = clEnqueueReadBuffer(b->cl_commands, cl_data.destData, CL_FALSE, 0, sizeof(short)*(4+pitch/2*3), output, 0, NULL, NULL);
    CL_CHECK_SUCCESS(b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read output array!\n",
        vp8_short_idct4x4llm_1_c(input,output,pitch),
    );

    return;

}

void vp8_dc_only_idct_add_cl(BLOCKD *b, cl_bool use_diff, int diff_offset, int qcoeff_offset, int pred_offset, unsigned char *dst_ptr, int pitch, int stride)
{
    
    int err;
    size_t global = 16;
    unsigned char *pred_ptr = b->predictor_base + pred_offset;

    short input_dc;
    if (use_diff == CL_TRUE){
        input_dc = b->diff_base[diff_offset];
    } else {
        input_dc = b->qcoeff_base[qcoeff_offset] * b->dequant[0];
    }

    if (cl_initialized != CL_SUCCESS){
        vp8_dc_only_idct_add_c(input_dc, pred_ptr, dst_ptr, pitch, stride);
        return;
    }

    CL_ENSURE_BUF_SIZE(b->cl_commands, cl_data.srcData, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
            sizeof(unsigned char)*(4*pitch+4), cl_data.srcAlloc, pred_ptr,
            vp8_dc_only_idct_add_c(input_dc, pred_ptr, dst_ptr, pitch, stride)
    );

    CL_ENSURE_BUF_SIZE(b->cl_commands, cl_data.destData,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
            sizeof(unsigned char) * ( 4 * stride + 4), cl_data.destAlloc, dst_ptr,
            vp8_dc_only_idct_add_c(input_dc, pred_ptr, dst_ptr, pitch, stride)
    );

    //Set arguments and run kernel
    err = 0;
    err = clSetKernelArg(cl_data.vp8_dc_only_idct_add_kernel, 0, sizeof(short), &input_dc);
    err |= clSetKernelArg(cl_data.vp8_dc_only_idct_add_kernel, 1, sizeof (cl_mem), &cl_data.srcData);
    err |= clSetKernelArg(cl_data.vp8_dc_only_idct_add_kernel, 2, sizeof (cl_mem), &cl_data.destData);
    err |= clSetKernelArg(cl_data.vp8_dc_only_idct_add_kernel, 3, sizeof (int), &pitch);
    err |= clSetKernelArg(cl_data.vp8_dc_only_idct_add_kernel, 4, sizeof (int), &stride);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        vp8_dc_only_idct_add_c(input_dc, pred_ptr, dst_ptr, pitch, stride),
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(b->cl_commands, cl_data.vp8_dc_only_idct_add_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);
        vp8_dc_only_idct_add_c(input_dc, pred_ptr, dst_ptr, pitch, stride),
    );

    /* Read back the result data from the device */
    err = clEnqueueReadBuffer(b->cl_commands, cl_data.destData, CL_FALSE, 0,
            sizeof(unsigned char) * ( 4 * stride + 4), dst_ptr, 0, NULL, NULL);
    CL_CHECK_SUCCESS(b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read output array!\n",
        vp8_dc_only_idct_add_c(input_dc, pred_ptr, dst_ptr, pitch, stride),
    );

    return;
}

void vp8_short_inv_walsh4x4_cl(BLOCKD *b)
{
    int err;
    size_t global = 4;

    if (cl_initialized != CL_SUCCESS){
        vp8_short_inv_walsh4x4_c(b->dqcoeff_base+b->dqcoeff_offset,&b->diff_base[b->diff_offset]);
        return;
    }

    //Set arguments and run kernel
    err = 0;
    err = clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_1st_pass_kernel, 0, sizeof (cl_mem), &b->cl_dqcoeff_mem);
    err |= clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_1st_pass_kernel, 1, sizeof(int), &b->dqcoeff_offset);
    err |= clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_1st_pass_kernel, 2, sizeof (cl_mem), &b->cl_diff_mem);
    err |= clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_1st_pass_kernel, 3, sizeof(int), &b->diff_offset);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        vp8_short_inv_walsh4x4_c(b->dqcoeff_base+b->dqcoeff_offset, &b->diff_base[b->diff_offset]),
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(b->cl_commands, cl_data.vp8_short_inv_walsh4x4_1st_pass_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);
        vp8_short_inv_walsh4x4_c(b->dqcoeff_base+b->dqcoeff_offset, &b->diff_base[b->diff_offset]),
    );

    //Second pass
    //Set arguments and run kernel
    err = 0;
    err = clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_2nd_pass_kernel, 0, sizeof (cl_mem), &b->cl_diff_mem);
    err |= clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_2nd_pass_kernel, 1, sizeof(int), &b->diff_offset);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        vp8_short_inv_walsh4x4_c(b->dqcoeff_base+b->dqcoeff_offset, &b->diff_base[b->diff_offset]),
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(b->cl_commands, cl_data.vp8_short_inv_walsh4x4_2nd_pass_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);
        vp8_short_inv_walsh4x4_c(b->dqcoeff_base+b->dqcoeff_offset, &b->diff_base[b->diff_offset]),
    );

    return;
}

void vp8_short_inv_walsh4x4_1_cl(BLOCKD *b)
{
    
    int err;
    size_t global = 4;

    if (cl_initialized != CL_SUCCESS){
        vp8_short_inv_walsh4x4_1_c(b->dqcoeff_base + b->dqcoeff_offset,
            &b->diff_base[b->diff_offset]);
        return;
    }

    //Set arguments and run kernel
    err = 0;
    err = clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_1_kernel, 0, sizeof (cl_mem), &b->cl_dqcoeff_mem);
    err |= clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_1_kernel, 1, sizeof (int), &b->dqcoeff_offset);
    err |= clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_1_kernel, 2, sizeof (cl_mem), &b->cl_diff_mem);
    err |= clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_1_kernel, 3, sizeof (int), &b->diff_offset);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        vp8_short_inv_walsh4x4_1_c(b->dqcoeff_base + b->dqcoeff_offset,
            &b->diff_base[b->diff_offset]),
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(b->cl_commands, cl_data.vp8_short_inv_walsh4x4_1_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);
        vp8_short_inv_walsh4x4_1_c(b->dqcoeff_base + b->dqcoeff_offset,
                &b->diff_base[b->diff_offset]),
    );

    return;
}
