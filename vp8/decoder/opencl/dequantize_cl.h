/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#ifndef DEQUANTIZE_CL_H
#define DEQUANTIZE_CL_H

#include "vp8/decoder/dequantize.h"
#include "vp8/common/opencl/vp8_opencl.h"

#define prototype_dequant_block_cl(sym) \
    void sym(BLOCKD *x)

#define prototype_dequant_idct_add_cl(sym) \
    void sym(short *input, short *dq, \
             unsigned char *pred, unsigned char *output, \
             int pitch, int stride)

#define prototype_dequant_dc_idct_add_cl(sym) \
    void sym(short *input, short *dq, \
             unsigned char *pred, unsigned char *output, \
             int pitch, int stride, \
             int dc)

#define prototype_dequant_dc_idct_add_y_block_cl(sym) \
    void sym(short *q, short *dq, \
             unsigned char *pre, unsigned char *dst, \
             int stride, char *eobs, short *dc)

#define prototype_dequant_idct_add_y_block_cl(sym) \
    void sym(short *q, short *dq, \
             unsigned char *pre, unsigned char *dst, \
             int stride, char *eobs)

#define prototype_dequant_idct_add_uv_block_cl(sym) \
    void sym(short *q, short *dq, \
             unsigned char *pre, unsigned char *dst_u, \
             unsigned char *dst_v, int stride, char *eobs)

const char *dequantCompileOptions = "";
const char *dequant_cl_file_name = "vp8/decoder/opencl/dequantize_cl.cl";


#endif
