/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vpx_ports/config.h"
//#include "vpx_ports/opencl.h"
//#include "g_common.h"
//#include "pragmas.h"
#include "subpixel.h"
//#include "loopfilter.h"
//#include "recon.h"
//#include "idct.h"
#include "subpixel_cl.h"
#include "onyxc_int.h"


/*
extern void (*vp8_build_intra_predictors_mby_ptr)(MACROBLOCKD *x);
extern void vp8_build_intra_predictors_mby(MACROBLOCKD *x);
extern void vp8_build_intra_predictors_mby_neon(MACROBLOCKD *x);

extern void (*vp8_build_intra_predictors_mby_s_ptr)(MACROBLOCKD *x);
extern void vp8_build_intra_predictors_mby_s(MACROBLOCKD *x);
extern void vp8_build_intra_predictors_mby_s_neon(MACROBLOCKD *x);
*/

void vp8_arch_opencl_common_init(VP8_COMMON *ctx)
{
//#if CONFIG_RUNTIME_CPU_DETECT
    VP8_COMMON_RTCD *rtcd = &ctx->rtcd;

    /* Override default functions with OpenCL accelerated ones. */
    //rtcd->idct.idct1        = vp8_short_idct4x4llm_1_cl;
    //rtcd->idct.idct16       = vp8_short_idct4x4llm_cl;
    //rtcd->idct.idct1_scalar_add = vp8_dc_only_idct_add_cl;
    //rtcd->idct.iwalsh1      = vp8_short_inv_walsh4x4_1_cl;
    //rtcd->idct.iwalsh16     = vp8_short_inv_walsh4x4_cl;

    //rtcd->recon.copy16x16   = vp8_copy_mem16x16_cl;
    //rtcd->recon.copy8x8     = vp8_copy_mem8x8_cl;
    //rtcd->recon.copy8x4     = vp8_copy_mem8x4_cl;
    //rtcd->recon.recon       = vp8_recon_b_cl;
    //rtcd->recon.recon2      = vp8_recon2b_cl;
    //rtcd->recon.recon4      = vp8_recon4b_cl;
    //rtcd->recon.recon_mb    = vp8_recon_mb_cl;
    //rtcd->recon.recon_mby   = vp8_recon_mby_cl;

    rtcd->subpix.sixtap16x16   = vp8_sixtap_predict16x16_cl;
    rtcd->subpix.sixtap8x8     = vp8_sixtap_predict8x8_cl;
    rtcd->subpix.sixtap8x4     = vp8_sixtap_predict8x4_cl;
    rtcd->subpix.sixtap4x4     = vp8_sixtap_predict_cl;
    //rtcd->subpix.bilinear16x16 = vp8_bilinear_predict16x16_cl;
    //rtcd->subpix.bilinear8x8   = vp8_bilinear_predict8x8_cl;
    //rtcd->subpix.bilinear8x4   = vp8_bilinear_predict8x4_cl;
    //rtcd->subpix.bilinear4x4   = vp8_bilinear_predict4x4_cl;

    //rtcd->loopfilter.normal_mb_v = vp8_loop_filter_mbv_cl;
    //rtcd->loopfilter.normal_b_v  = vp8_loop_filter_bv_cl;
    //rtcd->loopfilter.normal_mb_h = vp8_loop_filter_mbh_cl;
    //rtcd->loopfilter.normal_b_h  = vp8_loop_filter_bh_cl;
    //rtcd->loopfilter.simple_mb_v = vp8_loop_filter_mbvs_cl;
    //rtcd->loopfilter.simple_b_v  = vp8_loop_filter_bvs_cl;
    //rtcd->loopfilter.simple_mb_h = vp8_loop_filter_mbhs_cl;
    //rtcd->loopfilter.simple_b_h  = vp8_loop_filter_bhs_cl;

#if CONFIG_POSTPROC || (CONFIG_VP8_ENCODER && CONFIG_PSNR)
    //rtcd->postproc.down        = vp8_mbpost_proc_down_cl;
    //rtcd->postproc.across      = vp8_mbpost_proc_across_ip_cl;
    //rtcd->postproc.downacross  = vp8_post_proc_down_and_across_cl;
    //rtcd->postproc.addnoise    = vp8_plane_add_noise_cl;
    //rtcd->postproc.blend_mb    = vp8_blend_mb_cl;
#endif

    

//#endif
    /* Pure C: */
    //vp8_build_intra_predictors_mby_ptr = vp8_build_intra_predictors_mby_cl;
    //vp8_build_intra_predictors_mby_s_ptr = vp8_build_intra_predictors_mby_s_cl;
    
}
