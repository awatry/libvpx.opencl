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

#include "idctllm_cl.h"

static const int cospi8sqrt2minus1 = 20091;
static const int sinpi8sqrt2      = 35468;
static const int rounding = 0;

int cl_init_idct() {
    int err;

    // Create the filter compute program from the file-defined source code
    if (cl_load_program(&cl_data.idct_program, idctllm_cl_file_name,
            idctCompileOptions) != CL_SUCCESS)
        return CL_TRIED_BUT_FAILED;

    // Create the compute kernel in the program we wish to run
    CL_CREATE_KERNEL(cl_data,idct_program,vp8_short_inv_walsh4x4_1_kernel,"vp8_short_inv_walsh4x4_1_kernel");
    CL_CREATE_KERNEL(cl_data,idct_program,vp8_short_inv_walsh4x4_kernel,"vp8_short_inv_walsh4x4_kernel");
    CL_CREATE_KERNEL(cl_data,idct_program,vp8_dc_only_idct_add_kernel,"vp8_dc_only_idct_add_kernel");
    CL_CREATE_KERNEL(cl_data,idct_program,vp8_short_idct4x4llm_1_kernel,"vp8_short_idct4x4llm_1_kernel");
    CL_CREATE_KERNEL(cl_data,idct_program,vp8_short_idct4x4llm_kernel,"vp8_short_idct4x4llm_kernel");

    return CL_SUCCESS;
}

#define max(x,y) (x > y ? x: y)

void vp8_short_idct4x4llm_cl(short *input, short *output, int pitch)
{
    int err;
    size_t global = 1; //1 instance for now

    if (cl_initialized != CL_SUCCESS){
        vp8_short_idct4x4llm_c(input,output,pitch);
        return;
    }

    CL_ENSURE_BUF_SIZE(cl_data.srcData, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
            sizeof(short)*16, cl_data.srcAlloc, input,
            vp8_short_idct4x4llm_c(input,output,pitch)
    );

    CL_ENSURE_BUF_SIZE(cl_data.destData,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
            sizeof(short)*(4+(pitch/2)*3), cl_data.destAlloc, output,
            vp8_short_idct4x4llm_c(input,output,pitch)
    );

    //Set arguments and run kernel
    err = 0;
    err = clSetKernelArg(cl_data.vp8_short_idct4x4llm_kernel, 0, sizeof (cl_mem), &cl_data.srcData);
    err |= clSetKernelArg(cl_data.vp8_short_idct4x4llm_kernel, 1, sizeof (cl_mem), &cl_data.destData);
    err |= clSetKernelArg(cl_data.vp8_short_idct4x4llm_kernel, 2, sizeof (int), &pitch);
    CL_CHECK_SUCCESS( err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        vp8_short_idct4x4llm_c(input,output,pitch),
    );
    
    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(cl_data.commands, cl_data.vp8_short_idct4x4llm_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);
        vp8_short_idct4x4llm_c(input,output,pitch),
    );

    /* Read back the result data from the device */
    err = clEnqueueReadBuffer(cl_data.commands, cl_data.destData, CL_FALSE, 0, sizeof(short)*(4+pitch/2*3), output, 0, NULL, NULL);
    CL_CHECK_SUCCESS(err != CL_SUCCESS,
        "Error: Failed to read output array!\n",
        vp8_short_idct4x4llm_c(input,output,pitch),
    );

    clFinish(cl_data.commands);
    printf("Ran 4x4 IDCT kernel\n");

    //vp8_short_idct4x4llm_c(input,output,pitch);
    return;
}

void vp8_short_idct4x4llm_1_cl(short *input, short *output, int pitch)
{
    int i;
    int a1;
    short *op = output;
    int shortpitch = pitch >> 1;
    a1 = ((input[0] + 4) >> 3);

    printf("4x4llm_1_cl\n");

    for (i = 0; i < 4; i++)
    {
        op[0] = a1;
        op[1] = a1;
        op[2] = a1;
        op[3] = a1;
        op += shortpitch;
    }
}

void vp8_dc_only_idct_add_cl(short input_dc, unsigned char *pred_ptr, unsigned char *dst_ptr, int pitch, int stride)
{
    int a1 = ((input_dc + 4) >> 3);
    int r, c;

    printf("dc_only_idct_add_cl\n");
    
    for (r = 0; r < 4; r++)
    {
        for (c = 0; c < 4; c++)
        {
            int a = a1 + pred_ptr[c] ;

            if (a < 0)
                a = 0;

            if (a > 255)
                a = 255;

            dst_ptr[c] = (unsigned char) a ;
        }

        dst_ptr += stride;
        pred_ptr += pitch;
    }

}

void vp8_short_inv_walsh4x4_cl(short *input, short *output)
{
    int err;
    size_t global = 16;

    if (cl_initialized != CL_SUCCESS){
        vp8_short_inv_walsh4x4_c(input,output);
        return;
    }
    
    CL_ENSURE_BUF_SIZE(cl_data.srcData,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
            sizeof(cl_short)*16, cl_data.srcAlloc, input,
            vp8_short_inv_walsh4x4_c(input, output)
    );

    CL_ENSURE_BUF_SIZE(cl_data.destData,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
            sizeof(cl_short)*16, cl_data.destAlloc, output,
            vp8_short_inv_walsh4x4_c(input, output)
    );

    //Set arguments and run kernel
    err = 0;
    err = clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_kernel, 0, sizeof (cl_mem), &cl_data.srcData);
    err |= clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_kernel, 1, sizeof (cl_mem), &cl_data.destData);
    CL_CHECK_SUCCESS( err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        vp8_short_inv_walsh4x4_c(input, output),
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(cl_data.commands, cl_data.vp8_short_inv_walsh4x4_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);
        vp8_short_inv_walsh4x4_c(input, output),
    );

    /* Read back the result data from the device */
    err = clEnqueueReadBuffer(cl_data.commands, cl_data.destData, CL_FALSE, 0, sizeof(cl_short)*16, output, 0, NULL, NULL);
    CL_CHECK_SUCCESS(err != CL_SUCCESS,
        "Error: Failed to read output array!\n",
        vp8_short_inv_walsh4x4_c(input, output),
    );
    
    clFinish(cl_data.commands);
}

void vp8_short_inv_walsh4x4_1_cl(short *input, short *output)
{
    
    int err;
    size_t global = 16;

    if (cl_initialized != CL_SUCCESS){
        vp8_short_inv_walsh4x4_1_c(input,output);
        return;
    }

/*
    CL_ENSURE_BUF_SIZE(cl_data.srcData,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
            sizeof(short), cl_data.srcAlloc, input,
            vp8_short_inv_walsh4x4_1_c(input,output)
    );
*/

    CL_ENSURE_BUF_SIZE(cl_data.destData,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
            sizeof(short)*16, cl_data.destAlloc, output,
            vp8_short_inv_walsh4x4_1_c(input,output)
    );

    //Set arguments and run kernel
    err = 0;
//    err = clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_1_kernel, 0, sizeof (cl_mem), &cl_data.srcData);
    err = clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_1_kernel, 0, sizeof (cl_short), &input[0]);
    err |= clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_1_kernel, 1, sizeof (cl_mem), &cl_data.destData);
    CL_CHECK_SUCCESS( err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        vp8_short_inv_walsh4x4_1_c(input,output),
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(cl_data.commands, cl_data.vp8_short_inv_walsh4x4_1_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);
        vp8_short_inv_walsh4x4_1_c(input,output),
    );

    /* Read back the result data from the device */
    err = clEnqueueReadBuffer(cl_data.commands, cl_data.destData, CL_FALSE, 0, sizeof(short)*16, output, 0, NULL, NULL);
    CL_CHECK_SUCCESS(err != CL_SUCCESS,
        "Error: Failed to read output array!\n",
        vp8_short_inv_walsh4x4_1_c(input,output),
    );
    clFinish(cl_data.commands);
}
