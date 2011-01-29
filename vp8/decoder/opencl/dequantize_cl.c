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

#include "dequantize.h"
#include "dequantize_cl.h"

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
    int i;
    short *DQ  = d->dqcoeff_base + d->dqcoeff_offset;
    short *Q   = d->qcoeff_base + d->qcoeff_offset;
    short *DQC = d->dequant;

    printf("vp8_dequantize_b_cl\n");

    for (i = 0; i < 16; i++)
    {
        DQ[i] = Q[i] * DQC[i];
    }
}

void vp8_dequant_idct_add_cl(BLOCKD *b, short *input_base, int input_offset, short *dq, unsigned char *pred,
                            unsigned char *dest, int pitch, int stride)
{
    short output[16];
    short *diff_ptr = output;
    short *input = input_base + input_offset;
    int r, c;
    int i;

    printf("vp8_dequant_idct_add_cl\n");

    //set arguments

    //run kernel

    for (i = 0; i < 16; i++)
    {
        input[i] = dq[i] * input[i];
    }

    /* the idct halves ( >> 1) the pitch */
    vp8_short_idct4x4llm_c(input, output, 4 << 1);

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
