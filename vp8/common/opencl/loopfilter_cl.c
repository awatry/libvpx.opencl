/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include "../../../vpx_ports/config.h"
#include "loopfilter_cl.h"
#include "../onyxc_int.h"

#include "vp8_opencl.h"

const char *loopFilterCompileOptions = "-Ivp8/common/opencl -DVP8_FILTER_WEIGHT=128 -DVP8_FILTER_SHIFT=7 -DFILTER_OFFSET";
const char *loop_filter_cl_file_name = "vp8/common/opencl/filter_cl.cl";


typedef unsigned char uc;


prototype_loopfilter_cl(vp8_loop_filter_horizontal_edge_cl);
prototype_loopfilter_cl(vp8_loop_filter_vertical_edge_cl);
prototype_loopfilter_cl(vp8_mbloop_filter_horizontal_edge_cl);
prototype_loopfilter_cl(vp8_mbloop_filter_vertical_edge_cl);
prototype_loopfilter_cl(vp8_loop_filter_simple_horizontal_edge_cl);
prototype_loopfilter_cl(vp8_loop_filter_simple_vertical_edge_cl);

/* Horizontal MB filtering */
void vp8_loop_filter_mbh_cl(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                           int y_stride, int uv_stride, loop_filter_info *lfi, int simpler_lpf)
{
    (void) simpler_lpf;
    vp8_mbloop_filter_horizontal_edge_cl(y_ptr, y_stride, lfi->mbflim, lfi->lim, lfi->mbthr, 2);

    if (u_ptr)
        vp8_mbloop_filter_horizontal_edge_cl(u_ptr, uv_stride, lfi->uvmbflim, lfi->uvlim, lfi->uvmbthr, 1);

    if (v_ptr)
        vp8_mbloop_filter_horizontal_edge_cl(v_ptr, uv_stride, lfi->uvmbflim, lfi->uvlim, lfi->uvmbthr, 1);
}

void vp8_loop_filter_mbhs_cl(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                            int y_stride, int uv_stride, loop_filter_info *lfi, int simpler_lpf)
{
    (void) u_ptr;
    (void) v_ptr;
    (void) uv_stride;
    (void) simpler_lpf;
    vp8_loop_filter_simple_horizontal_edge_cl(y_ptr, y_stride, lfi->mbflim, lfi->lim, lfi->mbthr, 2);
}

/* Vertical MB Filtering */
void vp8_loop_filter_mbv_cl(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                           int y_stride, int uv_stride, loop_filter_info *lfi, int simpler_lpf)
{
    (void) simpler_lpf;
    vp8_mbloop_filter_vertical_edge_cl(y_ptr, y_stride, lfi->mbflim, lfi->lim, lfi->mbthr, 2);

    if (u_ptr)
        vp8_mbloop_filter_vertical_edge_cl(u_ptr, uv_stride, lfi->uvmbflim, lfi->uvlim, lfi->uvmbthr, 1);

    if (v_ptr)
        vp8_mbloop_filter_vertical_edge_cl(v_ptr, uv_stride, lfi->uvmbflim, lfi->uvlim, lfi->uvmbthr, 1);
}

void vp8_loop_filter_mbvs_cl(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                            int y_stride, int uv_stride, loop_filter_info *lfi, int simpler_lpf)
{
    (void) u_ptr;
    (void) v_ptr;
    (void) uv_stride;
    (void) simpler_lpf;
    vp8_loop_filter_simple_vertical_edge_cl(y_ptr, y_stride, lfi->mbflim, lfi->lim, lfi->mbthr, 2);
}

/* Horizontal B Filtering */
void vp8_loop_filter_bh_cl(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                          int y_stride, int uv_stride, loop_filter_info *lfi, int simpler_lpf)
{
    (void) simpler_lpf;
    vp8_loop_filter_horizontal_edge_cl(y_ptr + 4 * y_stride, y_stride, lfi->flim, lfi->lim, lfi->thr, 2);
    vp8_loop_filter_horizontal_edge_cl(y_ptr + 8 * y_stride, y_stride, lfi->flim, lfi->lim, lfi->thr, 2);
    vp8_loop_filter_horizontal_edge_cl(y_ptr + 12 * y_stride, y_stride, lfi->flim, lfi->lim, lfi->thr, 2);

    if (u_ptr)
        vp8_loop_filter_horizontal_edge_cl(u_ptr + 4 * uv_stride, uv_stride, lfi->uvflim, lfi->uvlim, lfi->uvthr, 1);

    if (v_ptr)
        vp8_loop_filter_horizontal_edge_cl(v_ptr + 4 * uv_stride, uv_stride, lfi->uvflim, lfi->uvlim, lfi->uvthr, 1);
}

void vp8_loop_filter_bhs_cl(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                           int y_stride, int uv_stride, loop_filter_info *lfi, int simpler_lpf)
{
    (void) u_ptr;
    (void) v_ptr;
    (void) uv_stride;
    (void) simpler_lpf;
    vp8_loop_filter_simple_horizontal_edge_cl(y_ptr + 4 * y_stride, y_stride, lfi->flim, lfi->lim, lfi->thr, 2);
    vp8_loop_filter_simple_horizontal_edge_cl(y_ptr + 8 * y_stride, y_stride, lfi->flim, lfi->lim, lfi->thr, 2);
    vp8_loop_filter_simple_horizontal_edge_cl(y_ptr + 12 * y_stride, y_stride, lfi->flim, lfi->lim, lfi->thr, 2);
}

/* Vertical B Filtering */
void vp8_loop_filter_bv_cl(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                          int y_stride, int uv_stride, loop_filter_info *lfi, int simpler_lpf)
{
    (void) simpler_lpf;
    vp8_loop_filter_vertical_edge_cl(y_ptr + 4, y_stride, lfi->flim, lfi->lim, lfi->thr, 2);
    vp8_loop_filter_vertical_edge_cl(y_ptr + 8, y_stride, lfi->flim, lfi->lim, lfi->thr, 2);
    vp8_loop_filter_vertical_edge_cl(y_ptr + 12, y_stride, lfi->flim, lfi->lim, lfi->thr, 2);

    if (u_ptr)
        vp8_loop_filter_vertical_edge_cl(u_ptr + 4, uv_stride, lfi->uvflim, lfi->uvlim, lfi->uvthr, 1);

    if (v_ptr)
        vp8_loop_filter_vertical_edge_cl(v_ptr + 4, uv_stride, lfi->uvflim, lfi->uvlim, lfi->uvthr, 1);
}

void vp8_loop_filter_bvs_cl(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr,
                           int y_stride, int uv_stride, loop_filter_info *lfi, int simpler_lpf)
{
    (void) u_ptr;
    (void) v_ptr;
    (void) uv_stride;
    (void) simpler_lpf;
    vp8_loop_filter_simple_vertical_edge_cl(y_ptr + 4, y_stride, lfi->flim, lfi->lim, lfi->thr, 2);
    vp8_loop_filter_simple_vertical_edge_cl(y_ptr + 8, y_stride, lfi->flim, lfi->lim, lfi->thr, 2);
    vp8_loop_filter_simple_vertical_edge_cl(y_ptr + 12, y_stride, lfi->flim, lfi->lim, lfi->thr, 2);
}

int cl_init_loop_filter() {
    int err;

    // Create the filter compute program from the file-defined source code
    if ( cl_load_program(&cl_data.loop_filter_program, loop_filter_cl_file_name,
            loopFilterCompileOptions) != CL_SUCCESS )
        return CL_TRIED_BUT_FAILED;

    // Create the compute kernel in the program we wish to run

    //Note that this kernel is temporarily the memcpy kernel...
    //After implementing the loop filter kernels, this should be changed.
    CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_loop_filter_horizontal_edge_kernel,"vp8_memcpy_kernel");

    //CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_block_variation_kernel,"vp8_block_variation_kernel");
    //CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_sixtap_predict8x8_kernel,"vp8_sixtap_predict8x8_kernel");
    //CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_sixtap_predict8x4_kernel,"vp8_sixtap_predict8x4_kernel");
    //CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_sixtap_predict16x16_kernel,"vp8_sixtap_predict16x16_kernel");
    //CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_bilinear_predict4x4_kernel,"vp8_bilinear_predict4x4_kernel");
    //CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_bilinear_predict8x4_kernel,"vp8_bilinear_predict8x4_kernel");
    //CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_bilinear_predict8x8_kernel,"vp8_bilinear_predict8x8_kernel");
    //CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_bilinear_predict16x16_kernel,"vp8_bilinear_predict16x16_kernel");
    //CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_memcpy_kernel,"vp8_memcpy_kernel");

    return CL_SUCCESS;
}

void cl_destroy_loop_filter(){

    if (cl_data.loop_filter_program)
        clReleaseProgram(cl_data.loop_filter_program);

    CL_RELEASE_KERNEL(cl_data.vp8_loop_filter_horizontal_edge_kernel);
    
    
    cl_data.loop_filter_program = NULL;
}
