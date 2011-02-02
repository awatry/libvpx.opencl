/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

//ACW: Remove me after debugging.
#include <stdio.h>
#include <string.h>

#include "vp8/common/blockd.h"
#include "dequantize_cl.h"

const char *dequantCompileOptions = "";
const char *dequant_cl_file_name = "vp8/decoder/opencl/dequantize_cl.cl";

extern void vp8_short_idct4x4llm_cl(short *input, short *output, int pitch) ;

void cl_memset_short(short *s, int c, size_t n) {
    for (n /= sizeof(short); n > 0; --n)
        *s++ = c;
}

int cl_destroy_dequant(){
    printf("Freeing dequant decoder resources\n");

    CL_RELEASE_KERNEL(cl_data.vp8_dequant_dc_idct_add_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_dequant_idct_add_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_dequantize_b_kernel);

    if (cl_data.dequant_program)
        clReleaseProgram(cl_data.dequant_program);
    cl_data.dequant_program = NULL;

    return CL_SUCCESS;
}

int cl_init_dequant() {
    int err;

    //printf("Initializing dequant program/kernels\n");

    // Create the compute program from the file-defined source code
    if (cl_load_program(&cl_data.dequant_program, dequant_cl_file_name,
            dequantCompileOptions) != CL_SUCCESS)
        return CL_TRIED_BUT_FAILED;

    // Create the compute kernels in the program we wish to run
    CL_CREATE_KERNEL(cl_data,dequant_program,vp8_dequant_dc_idct_add_kernel,"vp8_dequant_dc_idct_add_kernel");
    CL_CREATE_KERNEL(cl_data,dequant_program,vp8_dequant_idct_add_kernel,"vp8_dequant_idct_add_kernel");
    CL_CREATE_KERNEL(cl_data,dequant_program,vp8_dequantize_b_kernel,"vp8_dequantize_b_kernel");

    //printf("Created dequant kernels\n");

    return CL_SUCCESS;
}

void vp8_dequantize_b_cl(BLOCKD *d)
{
    int i,err;
    size_t global = 1;

    if (cl_initialized != CL_SUCCESS){
        vp8_dequantize_b_c(d);
        return;
    }

     //Initialize memory
    CL_SET_BUF(d->cl_commands, d->cl_dqcoeff_mem, sizeof(cl_short)*400, d->dqcoeff_base,
        vp8_dequantize_b_c(d)
    );

    CL_SET_BUF(d->cl_commands, d->cl_qcoeff_mem, sizeof(cl_short)*400, d->qcoeff_base,
        vp8_dequantize_b_c(d)
    );

    //Already set in decodframe.c when initializing block
    CL_SET_BUF(d->cl_commands, d->cl_dequant_mem, sizeof(cl_short)*16, d->dequant,
        vp8_dequantize_b_c(d)
    );

    /* Set kernel arguments */
    err = 0;
    err = clSetKernelArg(cl_data.vp8_dequantize_b_kernel, 0, sizeof (cl_mem), &d->cl_dqcoeff_mem);
    err |= clSetKernelArg(cl_data.vp8_dequantize_b_kernel, 1, sizeof (cl_int), &d->dqcoeff_offset);
    err |= clSetKernelArg(cl_data.vp8_dequantize_b_kernel, 2, sizeof (cl_mem), &d->cl_qcoeff_mem);
    err |= clSetKernelArg(cl_data.vp8_dequantize_b_kernel, 3, sizeof (cl_int), &d->qcoeff_offset);
    err |= clSetKernelArg(cl_data.vp8_dequantize_b_kernel, 4, sizeof (cl_mem), &d->cl_dequant_mem);
    CL_CHECK_SUCCESS( d->cl_commands, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        vp8_dequantize_b_c(d),
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel( d->cl_commands, cl_data.vp8_dequantize_b_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( d->cl_commands, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);\
        vp8_dequantize_b_c(d),
    );

    /* Read back the result data from the device */
    err = clEnqueueReadBuffer(d->cl_commands, d->cl_dqcoeff_mem, CL_FALSE, 0, sizeof(cl_short)*400, d->dqcoeff_base, 0, NULL, NULL); \
    CL_CHECK_SUCCESS( d->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read output array!\n",
        vp8_dequantize_b_c(d),
    );

    clFinish(d->cl_commands);

}

void vp8_dequant_idct_add_cl(BLOCKD *b, unsigned char *dest_base,int dest_offset, int q_offset, int pred_offset, int pitch, int stride, vp8_dequant_idct_add_fn_t idct_add)
{
    short *qcoeff = b->qcoeff_base+q_offset;
    int err;
    int i;
    size_t global = 1, cur_size, dest_size;
    cl_mem dest_mem = NULL;

    //This should be set by callers
    //pred_offset += b->predictor_offset;

    if (cl_initialized != CL_SUCCESS){
        idct_add(qcoeff, b->dequant,  b->predictor_base + pred_offset,
            dest_base + dest_offset, pitch, stride);
        return;
    }

    /* NOTE: Eventually, all of these buffers need to be initialized outside of
     *       this function.
     */

    printf("vp8_dequant_idct_add_cl\n");

    //Not needed probably
    CL_FINISH(b->cl_commands);

    //Initialize memory
    CL_SET_BUF(b->cl_commands, b->cl_qcoeff_mem, sizeof(cl_short)*400, b->qcoeff_base,
        idct_add(qcoeff, b->dequant,  b->predictor_base + pred_offset,
            dest_base + dest_offset, pitch, stride)
    );

    //Don't think this is necessary
    CL_SET_BUF(b->cl_commands, b->cl_dequant_mem, sizeof(cl_short)*16 ,b->dequant,
        idct_add(qcoeff, b->dequant,  b->predictor_base + pred_offset,
            dest_base + dest_offset, pitch, stride)
    );

    CL_SET_BUF(b->cl_commands, b->cl_predictor_mem, sizeof(cl_uchar)*384, b->predictor_base,
        idct_add(qcoeff, b->dequant,  b->predictor_base + pred_offset,
            dest_base + dest_offset, pitch, stride)
    );

    //Dest size calculation stolen from memory allocation function for planes.
    dest_size = sizeof(cl_uchar)*(4*stride + dest_offset + 4);
    cur_size = 0;
    CL_ENSURE_BUF_SIZE(b->cl_commands, dest_mem, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
            dest_size, cur_size, dest_base,
            idct_add(qcoeff, b->dequant,  b->predictor_base + pred_offset,
            dest_base + dest_offset, pitch, stride)
    );
    
    /* Set kernel arguments */
    err = 0;
    err = clSetKernelArg(cl_data.vp8_dequant_idct_add_kernel, 0, sizeof (cl_mem), &b->cl_qcoeff_mem);
    err |= clSetKernelArg(cl_data.vp8_dequant_idct_add_kernel, 1, sizeof (int), &q_offset);
    err |= clSetKernelArg(cl_data.vp8_dequant_idct_add_kernel, 2, sizeof (cl_mem), &b->cl_dequant_mem);
    err |= clSetKernelArg(cl_data.vp8_dequant_idct_add_kernel, 3, sizeof (cl_mem), &b->cl_predictor_mem);
    err |= clSetKernelArg(cl_data.vp8_dequant_idct_add_kernel, 4, sizeof (int), &pred_offset);
    err |= clSetKernelArg(cl_data.vp8_dequant_idct_add_kernel, 5, sizeof (cl_mem), &dest_mem);
    err |= clSetKernelArg(cl_data.vp8_dequant_idct_add_kernel, 6, sizeof (int), &dest_offset);
    err |= clSetKernelArg(cl_data.vp8_dequant_idct_add_kernel, 7, sizeof (int), &pitch);
    err |= clSetKernelArg(cl_data.vp8_dequant_idct_add_kernel, 8, sizeof (int), &stride);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        idct_add(qcoeff, b->dequant,  b->predictor_base + pred_offset,
            dest_base + dest_offset, pitch, stride),
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel( b->cl_commands, cl_data.vp8_dequant_idct_add_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);\
        idct_add(qcoeff, b->dequant,  b->predictor_base + pred_offset,
            dest_base + dest_offset, pitch, stride),
    );

    /* Read back the result data from the device */
    err = clEnqueueReadBuffer(b->cl_commands, dest_mem, CL_FALSE, 0, dest_size, dest_base, 0, NULL, NULL);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read output array!\n",
        idct_add(qcoeff, b->dequant,  b->predictor_base + pred_offset,
            dest_base + dest_offset, pitch, stride),
    );

    //And remember to copy back qcoeff (modified by the memset)
    err = clEnqueueReadBuffer(b->cl_commands, b->cl_qcoeff_mem, CL_FALSE, 0, sizeof(short)*400, b->qcoeff_base, 0, NULL, NULL);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read from GPU!\n",
        idct_add(qcoeff, b->dequant,  b->predictor_base + pred_offset,
            dest_base + dest_offset, pitch, stride),
    );

    CL_FINISH(b->cl_commands);

    //CL Spec says this can be freed without clFinish first
    clReleaseMemObject(dest_mem); 
    dest_mem = NULL;

    return;
}

//Can modify arguments. Only called from vp8_dequant_dc_idct_add_y_block_cl.
void vp8_dequant_dc_idct_add_cl(short *input, short *dq, unsigned char *pred,
                               unsigned char *dest, int pitch, int stride,
                               int Dc)
{
    int i;
    short output[16];
    short *diff_ptr = output;
    int r, c;

    printf("vp8_dequant_dc_idct_add_cl\n");

    input[0] = (short)Dc;

    for (i = 1; i < 16; i++)
    {
        input[i] = dq[i] * input[i];
    }

    /* the idct halves ( >> 1) the pitch */
    vp8_short_idct4x4llm_cl(input, output, 4 << 1);
    CL_FINISH(cl_data.commands); //Need to fix idct4x4llm for Mblock-level CQs
    
    cl_memset_short(input, 0, 32);

    for (r = 0; r < 4; r++)
    {
        for (c = 0; c < 4; c++)
        {
            int a = diff_ptr[c] + pred[c];

            if (a < 0)
                a = 0;

            if (a > 255)
                a = 255;

            dest[c] = (unsigned char) a;
        }

        dest += stride;
        diff_ptr += 4;
        pred += pitch;
    }
}
