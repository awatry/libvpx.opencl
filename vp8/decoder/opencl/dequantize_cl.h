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

#ifdef  __cplusplus
extern "C" {
#endif

#include "vp8/decoder/onyxd_int.h"
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

void vp8_dequantize_b_cl(BLOCKD *d);

//CL functions
void vp8_dequant_idct_add_cl(BLOCKD *b, unsigned char *dest_base,int dest_offset,
        int q_offset, int pred_offset, int pitch, int stride,
        vp8_dequant_idct_add_fn_t idct_add);

//C functions
void vp8_dequant_dc_idct_add_cl(short *input, short *dq, unsigned char *pred,
                               unsigned char *dest, int pitch, int stride,
                               int Dc);

//CL but using the wrong cl_commands and cl_mem
void vp8_dc_only_idct_add_cl(short input_dc, unsigned char *pred_ptr,
                            unsigned char *dst_ptr, int pitch, int stride);


void vp8_dequant_dc_idct_add_y_block_cl
            (MACROBLOCKD *xd, short *q, short *dq, unsigned char *pre,
             unsigned char *dst, int stride, char *eobs, short *dc);

void vp8_dequant_idct_add_y_block_cl (VP8D_COMP *pbi, MACROBLOCKD *xd, unsigned char *dst);

void vp8_dequant_idct_add_uv_block_cl(VP8D_COMP *pbi, MACROBLOCKD *xd,
        vp8_dequant_idct_add_uv_block_fn_t idct_add_uv_block
);

extern const char *dequantCompileOptions;
extern const char *dequant_cl_file_name;

#ifdef  __cplusplus
}
#endif

#endif
