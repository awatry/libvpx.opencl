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


void vp8_inverse_transform_b(const vp8_idct_rtcd_vtable_t *rtcd, BLOCKD *b,
                             int pitch)
{
    if (*b->eob > 1)
    {
        IDCT_INVOKE(rtcd, idct16)(b->dqcoeff_base+b->dqcoeff_offset, b->predictor_base + b->predictor_offset, pitch,
              *(b->base_dst) + b->dst, b->dst_stride);
    }
    else
    {
        IDCT_INVOKE(rtcd, idct1_scalar_add)(b->dqcoeff_base[b->dqcoeff_offset], b->predictor_base + b->predictor_offset, pitch,
                         *(b->base_dst) + b->dst, b->dst_stride);
    }

}

static void recon_dcblock(MACROBLOCKD *x)
{
    BLOCKD *b = &x->block[24];
    int i;

    for (i = 0; i < 16; i++)
    {
        x->block[i].dqcoeff_base[x->block[i].dqcoeff_offset] = b->diff_base[i + b->diff_offset];
    }

}

void vp8_inverse_transform_mby(const vp8_idct_rtcd_vtable_t *rtcd, MACROBLOCKD *x)
{
    int i;

    if(x->mode_info_context->mbmi.mode != SPLITMV)
    {
        /* do 2nd order transform on the dc block */
        IDCT_INVOKE(rtcd, iwalsh16)(x->block[24].dqcoeff_base + x->block[24].dqcoeff_offset,
            x->block[24].diff_base + x->block[24].diff_offset);

        recon_dcblock(x);
    }

    for (i = 0; i < 16; i++)
    {
        vp8_inverse_transform_b(rtcd, &x->block[i], 16);
    }

}
void vp8_inverse_transform_mbuv(const vp8_idct_rtcd_vtable_t *rtcd, MACROBLOCKD *x)
{
    int i;

    for (i = 16; i < 24; i++)
    {
        vp8_inverse_transform_b(rtcd, &x->block[i], 8);
    }

}
