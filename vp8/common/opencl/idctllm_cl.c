/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
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
#include "blockd_cl.h"

void cl_destroy_idct(){

    if (cl_data.idct_program)
        clReleaseProgram(cl_data.idct_program);

    cl_data.idct_program = NULL;
    
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_short_inv_walsh4x4_1_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_short_inv_walsh4x4_1st_pass_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_short_inv_walsh4x4_2nd_pass_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_dc_only_idct_add_kernel);
    //VP8_CL_RELEASE_KERNEL(cl_data.vp8_short_idct4x4llm_1_kernel);
    //VP8_CL_RELEASE_KERNEL(cl_data.vp8_short_idct4x4llm_kernel);

}

int cl_init_idct() {
    int err;

    // Create the filter compute program from the file-defined source code
    if (cl_load_program(&cl_data.idct_program, idctllm_cl_file_name,
            idctCompileOptions) != CL_SUCCESS)
        return VP8_CL_TRIED_BUT_FAILED;

    // Create the compute kernel in the program we wish to run
    VP8_CL_CREATE_KERNEL(cl_data,idct_program,vp8_short_inv_walsh4x4_1_kernel,"vp8_short_inv_walsh4x4_1_kernel");
    VP8_CL_CREATE_KERNEL(cl_data,idct_program,vp8_short_inv_walsh4x4_1st_pass_kernel,"vp8_short_inv_walsh4x4_1st_pass_kernel");
    VP8_CL_CREATE_KERNEL(cl_data,idct_program,vp8_short_inv_walsh4x4_2nd_pass_kernel,"vp8_short_inv_walsh4x4_2nd_pass_kernel");
    VP8_CL_CREATE_KERNEL(cl_data,idct_program,vp8_dc_only_idct_add_kernel,"vp8_dc_only_idct_add_kernel");

    ////idct4x4llm kernels are only useful for the encoder
    //VP8_CL_CREATE_KERNEL(cl_data,idct_program,vp8_short_idct4x4llm_1_kernel,"vp8_short_idct4x4llm_1_kernel");
    //VP8_CL_CREATE_KERNEL(cl_data,idct_program,vp8_short_idct4x4llm_kernel,"vp8_short_idct4x4llm_kernel");

    return CL_SUCCESS;
}

#define max(x,y) (x > y ? x: y)
//#define NO_CL

void vp8_dc_only_idct_add_cl(BLOCKD *b, cl_int use_diff, int diff_offset, 
        int qcoeff_offset, int pred_offset,
        unsigned char *dst_base, cl_mem dst_mem, int dst_offset, size_t dest_size,
        int pitch, int stride
)
{
    
    int err;
    size_t global = 16;

    int free_mem = 0;
    //cl_mem dest_mem = NULL;

    if (dst_mem == NULL){
        VP8_CL_CREATE_BUF(b->cl_commands, dst_mem,,
                dest_size, dst_base,,
        );
        free_mem = 1;
    }

    //Set arguments and run kernel
	printf("I'm going to fail because I still believe in the existence of blockd.diff_mem\n");
    err =  clSetKernelArg(cl_data.vp8_dc_only_idct_add_kernel, 0, sizeof (cl_mem), &b->cl_predictor_mem);
    err |= clSetKernelArg(cl_data.vp8_dc_only_idct_add_kernel, 1, sizeof (int), &pred_offset);
    err |= clSetKernelArg(cl_data.vp8_dc_only_idct_add_kernel, 2, sizeof (cl_mem), &dst_mem);
    err |= clSetKernelArg(cl_data.vp8_dc_only_idct_add_kernel, 3, sizeof (int), &dst_offset);
    err |= clSetKernelArg(cl_data.vp8_dc_only_idct_add_kernel, 4, sizeof (int), &pitch);
    err |= clSetKernelArg(cl_data.vp8_dc_only_idct_add_kernel, 5, sizeof (int), &stride);
    err |= clSetKernelArg(cl_data.vp8_dc_only_idct_add_kernel, 6, sizeof (cl_mem), &b->cl_qcoeff_mem);
    err |= clSetKernelArg(cl_data.vp8_dc_only_idct_add_kernel, 7, sizeof (int), &qcoeff_offset);
    err |= clSetKernelArg(cl_data.vp8_dc_only_idct_add_kernel, 8, sizeof (cl_mem), &b->cl_dequant_mem);
    VP8_CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",,
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(b->cl_commands, cl_data.vp8_dc_only_idct_add_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    VP8_CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);,
    );


    if (free_mem == 1){
    /* Read back the result data from the device */
        err = clEnqueueReadBuffer(b->cl_commands, dst_mem, CL_FALSE, 0,
                dest_size, dst_base, 0, NULL, NULL);

        VP8_CL_CHECK_SUCCESS(b->cl_commands, err != CL_SUCCESS,
            "Error: Failed to read output array!\n",,
        );

        clReleaseMemObject(dst_mem);
    }

    return;
}

void vp8_short_inv_walsh4x4_cl(BLOCKD *b)
{
    int err;
    size_t global = 4;

    if (cl_initialized != CL_SUCCESS){
        vp8_short_inv_walsh4x4_c(&b->dqcoeff_base[b->dqcoeff_offset],
                    b->dqcoeff_base);
        return;
    }

    //Set arguments and run kernel
    err = 0;
	printf("I'm going to fail now... blockd.diff_mem doesn't exist anymore... rewrite this kernel");
    err = clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_1st_pass_kernel, 0, sizeof (cl_mem), &b->cl_dqcoeff_mem);
    err |= clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_1st_pass_kernel, 1, sizeof(int), &b->dqcoeff_offset);
    VP8_CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
                vp8_short_inv_walsh4x4_c(&b->dqcoeff_base[b->dqcoeff_offset],
                    b->dqcoeff_base),
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(b->cl_commands, cl_data.vp8_short_inv_walsh4x4_1st_pass_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    VP8_CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);
		vp8_short_inv_walsh4x4_c(&b->dqcoeff_base[b->dqcoeff_offset],
				b->dqcoeff_base),
    );

    //Second pass
    //Set arguments and run kernel
    err = 0;
	printf("Does the 2 pass kernel even make sense anymore?");
    VP8_CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
		vp8_short_inv_walsh4x4_c(&b->dqcoeff_base[b->dqcoeff_offset],
			b->dqcoeff_base),
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(b->cl_commands, cl_data.vp8_short_inv_walsh4x4_2nd_pass_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    VP8_CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);
		vp8_short_inv_walsh4x4_c(&b->dqcoeff_base[b->dqcoeff_offset],
			b->dqcoeff_base),
    );

    return;
}

void vp8_short_inv_walsh4x4_1_cl(BLOCKD *b)
{
    
    int err;
    size_t global = 4;

    if (cl_initialized != CL_SUCCESS){
        vp8_short_inv_walsh4x4_1_c(b->dqcoeff_base + b->dqcoeff_offset,
            b->dqcoeff_base);
        return;
    }

    //Set arguments and run kernel
    err = 0;
    err = clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_1_kernel, 0, sizeof (cl_mem), &b->cl_dqcoeff_mem);
    err |= clSetKernelArg(cl_data.vp8_short_inv_walsh4x4_1_kernel, 1, sizeof (int), &b->dqcoeff_offset);
	printf("I'm going to fail now... blockd.diff_mem doesn't exist anymore... rewrite this kernel");
	VP8_CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        vp8_short_inv_walsh4x4_1_c(&b->dqcoeff_base[b->dqcoeff_offset],
			b->dqcoeff_base),
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(b->cl_commands, cl_data.vp8_short_inv_walsh4x4_1_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    VP8_CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);
        vp8_short_inv_walsh4x4_1_c(&b->dqcoeff_base[b->dqcoeff_offset],
			b->dqcoeff_base),
    );

    return;
}
