/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#ifndef SUBPIXEL_CL_H
#define SUBPIXEL_CL_H

/* Note:
 *
 * This platform is commonly built for runtime CPU detection. If you modify
 * any of the function mappings present in this file, be sure to also update
 * them in the function pointer initialization code
 */

extern prototype_subpixel_predict(vp8_sixtap_predict16x16_cl);
extern prototype_subpixel_predict(vp8_sixtap_predict8x8_cl);
extern prototype_subpixel_predict(vp8_sixtap_predict8x4_cl);
extern prototype_subpixel_predict(vp8_sixtap_predict_cl);
extern prototype_subpixel_predict(vp8_bilinear_predict16x16_cl);
extern prototype_subpixel_predict(vp8_bilinear_predict8x8_cl);
extern prototype_subpixel_predict(vp8_bilinear_predict8x4_cl);
extern prototype_subpixel_predict(vp8_bilinear_predict4x4_cl);


#if !CONFIG_RUNTIME_CPU_DETECT
#undef  vp8_subpix_sixtap16x16
#define vp8_subpix_sixtap16x16 vp8_sixtap_predict16x16_cl

#undef  vp8_subpix_sixtap8x8
#define vp8_subpix_sixtap8x8 vp8_sixtap_predict8x8_cl

#undef  vp8_subpix_sixtap8x4
#define vp8_subpix_sixtap8x4 vp8_sixtap_predict8x4_cl

#undef  vp8_subpix_sixtap4x4
#define vp8_subpix_sixtap4x4 vp8_sixtap_predict_cl

#undef  vp8_subpix_bilinear16x16
#define vp8_subpix_bilinear16x16 vp8_bilinear_predict16x16_cl

#undef  vp8_subpix_bilinear8x8
#define vp8_subpix_bilinear8x8 vp8_bilinear_predict8x8_cl

#undef  vp8_subpix_bilinear8x4
#define vp8_subpix_bilinear8x4 vp8_bilinear_predict8x4_cl

#undef  vp8_subpix_bilinear4x4
#define vp8_subpix_bilinear4x4 vp8_bilinear_predict4x4_cl

#endif

//typedef enum
//{
//    SIXTAP = 0,
//    BILINEAR = 1
//} SUBPIX_TYPE;

#endif
