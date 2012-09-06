/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include "../onyxd_int.h"
#include "vp8/common/header.h"
#include "vp8/common/reconintra.h"
#include "vp8/common/reconintra4x4.h"
#include "vp8/common/recon.h"
#include "vp8/common/reconinter.h"

#include <assert.h>
#include <stdio.h>

#include "vpx_config.h"
#if CONFIG_OPENCL
#include "vp8/common/opencl/vp8_opencl.h"
#include "vp8/common/opencl/blockd_cl.h"
#include "vp8/common/opencl/reconinter_cl.h"
#include "vp8/common/opencl/dequantize_cl.h"
#endif

#define PROFILE_OUTPUT 0

//Implemented in ../decodframe.c
extern void mb_init_dequantizer(VP8D_COMP *pbi, MACROBLOCKD *xd);

void mb_init_dequantizer_cl(MACROBLOCKD *xd){
    int i, err;
    //Set up per-block dequant CL memory. Eventually, might be able to set up
    //one large buffer containing the entire large dequant buffer.
    if (cl_initialized == CL_SUCCESS){
        for (i=0; i < 25; i++){

#if 1 //Initialize CL memory on allocation?
            VP8_CL_CREATE_BUF(xd->cl_commands, xd->block[i].cl_dequant_mem,
                ,
                16*sizeof(cl_short),
                xd->block[i].dequant,,
            );
#else
            VP8_CL_CREATE_BUF(xd->cl_commands, xd->block[i].cl_dequant_mem,
                ,
                16*sizeof(cl_short),
                NULL,,
            );
#endif
        }
    }
}

#if CONFIG_RUNTIME_CPU_DETECT
#define RTCD_VTABLE(x) (&(pbi)->common.rtcd.x)
#else
#define RTCD_VTABLE(x) NULL
#endif

/* skip_recon_mb() is Modified: Instead of writing the result to predictor buffer and then copying it
 *  to dst buffer, we can write the result directly to dst buffer. This eliminates unnecessary copy.
 */
static void skip_recon_mb_cl(VP8D_COMP *pbi, MACROBLOCKD *xd)
{
    if (xd->frame_type == KEY_FRAME  ||  xd->mode_info_context->mbmi.ref_frame == INTRA_FRAME)
    {

        vp8_build_intra_predictors_mbuv_s(xd);
        RECON_INVOKE(&pbi->common.rtcd.recon,
                     build_intra_predictors_mby_s)(xd);

    }
    else
    {
#if ENABLE_CL_SUBPIXEL
        if (cl_initialized == CL_SUCCESS)
        {
            vp8_build_inter_predictors_mb_s_cl(xd);
        } else
#endif
        {
            vp8_build_inter16x16_predictors_mb(xd, xd->dst.y_buffer,
                                           xd->dst.u_buffer, xd->dst.v_buffer,
                                           xd->dst.y_stride, xd->dst.uv_stride);
        }
        VP8_CL_FINISH(xd->cl_commands);
#if !ONE_CQ_PER_MB
        VP8_CL_FINISH(xd->block[0].cl_commands);
        VP8_CL_FINISH(xd->block[16].cl_commands);
        VP8_CL_FINISH(xd->block[20].cl_commands);
#endif
    }
}

void vp8_decode_macroblock_cl(VP8D_COMP *pbi, MACROBLOCKD *xd, int eobtotal)
{
    int i;
    int throw_residual = 0;
    MB_PREDICTION_MODE mode;
	
	mode = xd->mode_info_context->mbmi.mode;

    if (eobtotal == 0 && mode != B_PRED && mode != SPLITMV &&
            !vp8dx_bool_error(xd->current_bc))
    {
        /* Special case:  Force the loopfilter to skip when eobtotal and
         * mb_skip_coeff are zero.
         * */
        xd->mode_info_context->mbmi.mb_skip_coeff = 1;

        skip_recon_mb_cl(pbi, xd);
        return;
    }

    if (xd->segmentation_enabled)
        mb_init_dequantizer(pbi, xd);

    /* do prediction */
    if (xd->mode_info_context->mbmi.ref_frame == INTRA_FRAME)
    {
        RECON_INVOKE(&pbi->common.rtcd.recon, build_intra_predictors_mbuv_s)(xd);

        if (mode != B_PRED)
        {
            RECON_INVOKE(&pbi->common.rtcd.recon,
                         build_intra_predictors_mby_s)(xd);
        } else {
            vp8_intra_prediction_down_copy(xd);
        }
    }
    else
    {
#if ENABLE_CL_SUBPIXEL
        vp8_build_inter_predictors_mb_cl(xd);
#else
        vp8_build_inter_predictors_mb(xd);
#endif

#if (1 || !ENABLE_CL_IDCT_DEQUANT)
        //Wait for inter-predict if dequant/IDCT is being done on the CPU
        VP8_CL_FINISH(xd->cl_commands);
#endif
    }
    /* When we have independent partitions we can apply residual even
     * though other partitions within the frame are corrupt.
     */
    throw_residual = (!pbi->independent_partitions &&
                      pbi->frame_corrupt_residual);
    throw_residual = (throw_residual || vp8dx_bool_error(xd->current_bc));

#if CONFIG_ERROR_CONCEALMENT
    if (pbi->ec_active &&
        (mb_idx >= pbi->mvs_corrupt_from_mb || throw_residual))
    {
        /* MB with corrupt residuals or corrupt mode/motion vectors.
         * Better to use the predictor as reconstruction.
         */
        pbi->frame_corrupt_residual = 1;
        vpx_memset(xd->qcoeff, 0, sizeof(xd->qcoeff));
        vp8_conceal_corrupt_mb(xd);
        return;
    }
#endif

    /* dequantization and idct */
    if (mode == B_PRED)
    {
        for (i = 0; i < 16; i++)
        {
            BLOCKD *b = &xd->block[i];
            short *qcoeff = b->qcoeff_base + b->qcoeff_offset;
            int b_mode = xd->mode_info_context->bmi[i].as_mode;

            RECON_INVOKE(RTCD_VTABLE(recon), intra4x4_predict)
                          ( *(b->base_dst) + b->dst, b->dst_stride, b_mode,
                            *(b->base_dst) + b->dst, b->dst_stride );

            if (xd->eobs[i] )
            {
                if (xd->eobs[i] > 1)
                {
                    DEQUANT_INVOKE(&pbi->common.rtcd.dequant, idct_add)
                        (qcoeff, b->dequant,
                        *(b->base_dst) + b->dst, b->dst_stride);
                }
                else
                {
                    IDCT_INVOKE(RTCD_VTABLE(idct), idct1_scalar_add)
                        (qcoeff[0] * b->dequant[0],
                        *(b->base_dst) + b->dst, b->dst_stride,
                        *(b->base_dst) + b->dst, b->dst_stride);
                    ((int *)qcoeff)[0] = 0;
                }
            }
        }
    }
    else
    {
        short *DQC = xd->dequant_y1;

        if (mode != SPLITMV)
        {
            BLOCKD *b = &xd->block[24];
            short *qcoeff = b->qcoeff_base + b->qcoeff_offset;
            short *dqcoeff = b->dqcoeff_base + b->dqcoeff_offset;
            
            /* do 2nd order transform on the dc block */
            if (xd->eobs[24] > 1)
            {
                DEQUANT_INVOKE(&pbi->common.rtcd.dequant, block)(b, xd->dequant_y2);

                IDCT_INVOKE(RTCD_VTABLE(idct), iwalsh16)(&dqcoeff[0],
                    xd->qcoeff);
                ((int *)qcoeff)[0] = 0;
                ((int *)qcoeff)[1] = 0;
                ((int *)qcoeff)[2] = 0;
                ((int *)qcoeff)[3] = 0;
                ((int *)qcoeff)[4] = 0;
                ((int *)qcoeff)[5] = 0;
                ((int *)qcoeff)[6] = 0;
                ((int *)qcoeff)[7] = 0;
            }
            else
            {
                dqcoeff[0] = qcoeff[0] * b->dequant[0];
                IDCT_INVOKE(RTCD_VTABLE(idct), iwalsh1)(&dqcoeff[0],
                    xd->qcoeff);
                ((int *)qcoeff)[0] = 0;
            }

            /* override the dc dequant constant in order to preserve the
             * dc components
             */
            DQC = xd->dequant_y1_dc;
        }

        DEQUANT_INVOKE (&pbi->common.rtcd.dequant, idct_add_y_block)
                        (xd->qcoeff, xd->block[0].dequant,
                         xd->dst.y_buffer,
                         xd->dst.y_stride, xd->eobs);
    }

    DEQUANT_INVOKE (&pbi->common.rtcd.dequant, idct_add_uv_block)
                    (xd->qcoeff+16*16, xd->dequant_uv,
                     xd->dst.u_buffer, xd->dst.v_buffer,
                     xd->dst.uv_stride, xd->eobs+16);
}

void vp8_decode_frame_cl_finish(VP8D_COMP *pbi){

    //If using OpenCL, free all of the GPU buffers we've allocated.
    if (cl_initialized == CL_SUCCESS){
#if ENABLE_CL_IDCT_DEQUANT
        int i;
#endif

        //Wait for stuff to finish, just in case
        VP8_CL_FINISH(pbi->mb.cl_commands);

#if !ONE_CQ_PER_MB
        VP8_CL_FINISH(pbi->mb.block[0].cl_commands);
        VP8_CL_FINISH(pbi->mb.block[16].cl_commands);
        VP8_CL_FINISH(pbi->mb.block[20].cl_commands);
        clReleaseCommandQueue(pbi->mb.block[0].cl_commands);
        clReleaseCommandQueue(pbi->mb.block[16].cl_commands);
        clReleaseCommandQueue(pbi->mb.block[20].cl_commands);
#endif

#if ENABLE_CL_IDCT_DEQUANT || ENABLE_CL_SUBPIXEL
        //Free Predictor CL buffer
        if (pbi->mb.cl_predictor_mem != NULL)
            clReleaseMemObject(pbi->mb.cl_predictor_mem);
#endif

#if ENABLE_CL_IDCT_DEQUANT
        //Free other CL Block/MBlock buffers
        if (pbi->mb.cl_qcoeff_mem != NULL)
            clReleaseMemObject(pbi->mb.cl_qcoeff_mem);
        if (pbi->mb.cl_dqcoeff_mem != NULL)
            clReleaseMemObject(pbi->mb.cl_dqcoeff_mem);
        if (pbi->mb.cl_eobs_mem != NULL)
            clReleaseMemObject(pbi->mb.cl_eobs_mem);

        for (i = 0; i < 25; i++){
            clReleaseMemObject(pbi->mb.block[i].cl_dequant_mem);
            pbi->mb.block[i].cl_dequant_mem = NULL;
        }
#endif
    }
}
