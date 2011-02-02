/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp8/decoder/onyxd_int.h"
#include "vpx_ports/config.h"
#include "../../common/idct.h"
#include "vp8/common/blockd.h"
#include "dequantize_cl.h"


void vp8_dequant_dc_idct_add_y_block_cl
            (MACROBLOCKD *xd, short *q, short *dq, unsigned char *pre,
             unsigned char *dst, int stride, char *eobs, short *dc)
{
    int i, j;

    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
        {
            if (*eobs++ > 1){
                vp8_dequant_dc_idct_add_cl (q, dq, pre, dst, 16, stride, dc[0]);
            }
            else{
                //Note: dc[0] needs to be either verified for unchanging value,
                //      or this needs to become an offset just like everything else
                CL_FINISH(xd->cl_commands);
                CL_FINISH(cl_data.commands);
                vp8_dc_only_idct_add_cl (dc[0], pre, dst, 16, stride);
                CL_FINISH(xd->cl_commands);
                CL_FINISH(cl_data.commands);
            }

            q   += 16;
            pre += 4;
            dst += 4;
            dc  ++;
        }

        pre += 64 - 16;
        dst += 4*stride - 16;
    }
}

void vp8_dequant_idct_add_y_block_cl (VP8D_COMP *pbi, MACROBLOCKD *xd, unsigned char *dst)
{
    int i, j;

    short *q = xd->qcoeff;
    int q_offset = 0;
    short *dq = xd->block[0].dequant;
    unsigned char *pre = xd->predictor;
    int pre_offset = 0;
    unsigned char *dest_base = xd->dst.y_buffer;
    int dest_offset = 0;
    int stride = xd->dst.y_stride;
    char *eobs = xd->eobs;


    //vp8_dequant_idct_add_y_block_c(q, dq, pre, dest_base, stride, eobs);
    //return;

    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
        {
            printf("vp8_dequant_idct_add_y_block_cl\n");
            if (*eobs++ > 1){
                CL_FINISH(cl_data.commands);
                CL_FINISH(xd->cl_commands);
                vp8_dequant_idct_add_cl(&xd->block[0],dst, dest_offset, q_offset, pre_offset, 16, stride, pbi->dequant.idct_add);
                CL_FINISH(cl_data.commands);
                CL_FINISH(xd->cl_commands);
            }
            else
            {
                CL_FINISH(cl_data.commands);
                CL_FINISH(xd->cl_commands);
                //Another case where (q+offset)[0] and dq[0] need to become references
                //to cl_mem locations.
                vp8_dc_only_idct_add_cl ((q+q_offset)[0]*dq[0], pre+pre_offset, dst+dest_offset, 16, stride);
                CL_FINISH(cl_data.commands);
                CL_FINISH(xd->cl_commands);
                ((int *)(q+q_offset))[0] = 0;
            }

            q_offset   += 16;
            pre_offset += 4;
            dest_offset += 4;
        }

        pre_offset += 64 - 16;
        dest_offset += 4*stride - 16;
    }
}

void vp8_dequant_idct_add_uv_block_cl(VP8D_COMP *pbi, MACROBLOCKD *xd,
        vp8_dequant_idct_add_uv_block_fn_t idct_add_uv_block
)
{
    int i, j;

    short *q = xd->qcoeff;
    short *dq = xd->block[16].dequant;
    unsigned char *pre = xd->predictor;
    unsigned char *dstu = xd->dst.u_buffer;
    unsigned char *dstv = xd->dst.v_buffer;
    int stride = xd->dst.uv_stride;
    char *eobs = xd->eobs+16;

    int pre_offset = 16*16;
    int q_offset = 16*16;

    
    if (cl_initialized != CL_SUCCESS){
        DEQUANT_INVOKE (&pbi->dequant, idct_add_uv_block)
        (xd->qcoeff+16*16, xd->block[16].dequant,
         xd->predictor+16*16, xd->dst.u_buffer, xd->dst.v_buffer,
         xd->dst.uv_stride, xd->eobs+16);
        return;
    }

    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 2; j++)
        {
            printf("vp8_dequant_idct_add_uv_block_cl\n");
            if (*eobs++ > 1){
                //vp8_dequant_idct_add_cl (xd->block[16], q, dq, pre, dstu, 8, stride);
                CL_FINISH(cl_data.commands);
                CL_FINISH(xd->cl_commands);
                vp8_dequant_idct_add_cl(&xd->block[16], dstu, 0, q_offset, pre_offset, 8, stride, DEQUANT_INVOKE (&pbi->dequant, idct_add));
                CL_FINISH(cl_data.commands);
                CL_FINISH(xd->cl_commands);
            }
            else
            {
                //Another case where (q+offset)[0] and dq[0] need to become references
                //to cl_mem locations.
                CL_FINISH(cl_data.commands);
                CL_FINISH(xd->cl_commands);
                vp8_dc_only_idct_add_cl (*(q+q_offset)*dq[0], pre+pre_offset, dstu, 8, stride);
                ((int *)(q+q_offset))[0] = 0;
                CL_FINISH(cl_data.commands);
                CL_FINISH(xd->cl_commands);
            }

            q_offset    += 16;
            pre_offset  += 4;
            dstu += 4;
        }

        pre_offset  += 32 - 8;
        dstu += 4*stride - 8;
    }

    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 2; j++)
        {
            if (*eobs++ > 1){
                //vp8_dequant_idct_add_cl (q, dq, pre+pre_offset, dstv, 8, stride);
                CL_FINISH(cl_data.commands);
                CL_FINISH(xd->cl_commands);
                vp8_dequant_idct_add_cl (&xd->block[16], dstv, 0, q_offset, pre_offset, 8, stride, DEQUANT_INVOKE (&pbi->dequant, idct_add));
                CL_FINISH(cl_data.commands);
                CL_FINISH(xd->cl_commands);
            }
            else
            {
                //Another case where (q+offset)[0] and dq[0] need to become references
                //to cl_mem locations.
                CL_FINISH(cl_data.commands);
                CL_FINISH(xd->cl_commands);
                vp8_dc_only_idct_add_cl ((q+q_offset)[0]*dq[0], pre+pre_offset, dstv, 8, stride);
                CL_FINISH(cl_data.commands);
                CL_FINISH(xd->cl_commands);
                ((int *)(q+q_offset))[0] = 0;
            }

            q_offset    += 16;
            pre_offset  += 4;
            dstv += 4;
        }

        pre_offset  += 32 - 8;
        dstv += 4*stride - 8;
    }
}
