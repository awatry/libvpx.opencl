/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include "invtrans.h"

#if CONFIG_OPENCL
#include <stdio.h>
#include "opencl/vp8_opencl.h"
#endif


static void recon_dcblock(MACROBLOCKD *x)
{
    BLOCKD *b = &x->block[24];
    int i;

    for (i = 0; i < 16; i++)
    {
        *(x->block[i].dqcoeff_base+x->block[i].dqcoeff_offset) = b->diff_base[b->diff_offset+i];
    }

}

#if CONFIG_OPENCL
static void recon_dcblock_cl(MACROBLOCKD *x){
    size_t global = 16;
    int err;

    if (cl_initialized != CL_SUCCESS){
        recon_dcblock(x);
        return;
    }

    //Set kernel arguments
    err = 0;
    err = clSetKernelArg(cl_data.recon_dcblock_kernel, 0, sizeof (cl_mem), &x->cl_dqcoeff_mem);
    err |= clSetKernelArg(cl_data.recon_dcblock_kernel, 1, sizeof (cl_mem), &x->cl_diff_mem);
    err |= clSetKernelArg(cl_data.recon_dcblock_kernel, 2, sizeof (int), &x->block[24].diff_offset);
    CL_CHECK_SUCCESS( x->cl_commands, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        recon_dcblock(x),
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(x->cl_commands, cl_data.recon_dcblock_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( x->cl_commands, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);
        recon_dcblock(x),
    );

    //Finish here until the kernel tests out fine.
    CL_FINISH(x->cl_commands);
}
#endif


void vp8_inverse_transform_b(const vp8_idct_rtcd_vtable_t *rtcd, BLOCKD *b, int pitch)
{
#if CONFIG_OPENCL
    if (b->eob > 1)
        vp8_short_idct4x4llm_cl(b, b->dqcoeff_base + b->dqcoeff_offset, &b->diff_base[b->diff_offset], pitch);
    else
        vp8_short_idct4x4llm_1_cl(b, b->dqcoeff_base + b->dqcoeff_offset, &b->diff_base[b->diff_offset], pitch);

    //Move this to after the loops in the callers. It shouldn't affect correctness.
    CL_FINISH(b->cl_commands);
#else
    if (b->eob > 1)
        IDCT_INVOKE(rtcd, idct16)(b->dqcoeff_base + b->dqcoeff_offset, &b->diff_base[b->diff_offset], pitch);
    else
        IDCT_INVOKE(rtcd, idct1)(b->dqcoeff_base + b->dqcoeff_offset, &b->diff_base[b->diff_offset], pitch);
#endif
}


void vp8_inverse_transform_mby(const vp8_idct_rtcd_vtable_t *rtcd, MACROBLOCKD *x)
{
    int i;

    /* do 2nd order transform on the dc block */
#if CONFIG_OPENCL
    int err;
    CL_SET_BUF(x->cl_commands, x->cl_dqcoeff_mem, sizeof(cl_short)*400, x->block[24].dqcoeff_base,
            IDCT_INVOKE(rtcd, iwalsh16)(x->block[24].dqcoeff_base + x->block[23].dqcoeff_offset, &x->block[24].diff_base[x->block[24].diff_offset]));

    vp8_short_inv_walsh4x4_cl(&x->block[24], x->block[23].dqcoeff_offset,
            x->block[24].dqcoeff_base + x->block[23].dqcoeff_offset,
            &x->block[24].diff_base[x->block[24].diff_offset]);

    CL_FINISH(x->cl_commands);
#else
    IDCT_INVOKE(rtcd, iwalsh16)(x->block[24].dqcoeff_base + x->block[23].dqcoeff_offset, &x->block[24].diff_base[x->block[24].diff_offset]);
#endif

    recon_dcblock(x);

    for (i = 0; i < 16; i++)
    {
        vp8_inverse_transform_b(rtcd, &x->block[i], 32);
    }

#if CONFIG_OPENCL
    CL_FINISH(x->cl_commands);
#endif

}
void vp8_inverse_transform_mbuv(const vp8_idct_rtcd_vtable_t *rtcd, MACROBLOCKD *x)
{
    int i;

    for (i = 16; i < 24; i++)
    {
        vp8_inverse_transform_b(rtcd, &x->block[i], 16);
    }

#if CONFIG_OPENCL
    CL_FINISH(x->cl_commands);
#endif

}


void vp8_inverse_transform_mb(const vp8_idct_rtcd_vtable_t *rtcd, MACROBLOCKD *x)
{
    int i;

    if (x->mode_info_context->mbmi.mode != B_PRED &&
        x->mode_info_context->mbmi.mode != SPLITMV)
    {
        /* do 2nd order transform on the dc block */
        BLOCKD b = x->block[24];

#if CONFIG_OPENCL
        vp8_short_inv_walsh4x4_cl( &b, b.dqcoeff_offset,
                b.dqcoeff_base + b.dqcoeff_offset,
                &b.diff_base[b.diff_offset]);
        CL_FINISH(x->cl_commands);
#else
        IDCT_INVOKE(rtcd, iwalsh16)(b.dqcoeff_base+b.dqcoeff_offset, &b.diff_base[b.diff_offset]);
#endif

        recon_dcblock(x);
    }

    for (i = 0; i < 16; i++)
    {
        vp8_inverse_transform_b(rtcd, &x->block[i], 32);
    }


    for (i = 16; i < 24; i++)
    {
        vp8_inverse_transform_b(rtcd, &x->block[i], 16);
    }

#if CONFIG_OPENCL
    CL_FINISH(x->cl_commands);
#endif

}
