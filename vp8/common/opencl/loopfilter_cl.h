/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#ifndef loopfilter_cl_h
#define loopfilter_cl_h

#include "../../../vpx_ports/mem.h"

#include "../onyxc_int.h"
#include "blockd_cl.h"
#include "../loopfilter.h"

typedef struct VP8_LOOPFILTER_ARGS{
    cl_int priority_level;
    cl_int num_levels;
    cl_mem block_offsets_mem;
    cl_mem priority_num_blocks_mem;
    cl_mem buf_mem;
    cl_mem offsets_mem;
    cl_mem pitches_mem;
    cl_mem lfi_mem;
    cl_mem filters_mem; //combination of dc_diffs, rows, cols, and filter_levels
    cl_int frame_type;
} VP8_LOOPFILTER_ARGS;

#define prototype_loopfilter_cl(sym) \
    void sym(MACROBLOCKD *x, VP8_LOOPFILTER_ARGS *args, \
                int num_planes, int num_blocks)\

#define prototype_loopfilter_block_cl(sym) \
    void sym(MACROBLOCKD*, unsigned char *y, unsigned char *u, unsigned char *v,\
             int ystride, int uv_stride, loop_filter_info *lfi, int filter_level)

void vp8_loop_filter_filters_init();

extern void vp8_loop_filter_frame_cl
(
    VP8_COMMON *cm,
    MACROBLOCKD *mbd
);

extern prototype_loopfilter_block_cl(vp8_lf_normal_mb_v_cl);
extern prototype_loopfilter_block_cl(vp8_lf_normal_b_v_cl);
extern prototype_loopfilter_block_cl(vp8_lf_normal_mb_h_cl);
extern prototype_loopfilter_block_cl(vp8_lf_normal_b_h_cl);
extern prototype_loopfilter_block_cl(vp8_lf_simple_mb_v_cl);
extern prototype_loopfilter_block_cl(vp8_lf_simple_b_v_cl);
extern prototype_loopfilter_block_cl(vp8_lf_simple_mb_h_cl);
extern prototype_loopfilter_block_cl(vp8_lf_simple_b_h_cl);

typedef prototype_loopfilter_block_cl((*vp8_lf_block_cl_fn_t));

#endif
