
/*
 *  Copyright (c) 2011The WebM project authors. All Rights Reserved.
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

#include "vpx_config.h"
#include "vp8_opencl.h"
#include "blockd_cl.h"

const char *loopFilterCompileOptions = "-Ivp8/common/opencl";
const char *loop_filter_cl_file_name = "vp8/common/opencl/loopfilter.cl";

typedef unsigned char uc;

extern void vp8_loop_filter_frame
(
    VP8_COMMON *cm,
    MACROBLOCKD *mbd,
    int default_filt_lvl
);

typedef struct VP8_LOOP_MEM{
    cl_int num_blocks;
    cl_int num_planes;
    cl_mem offsets_mem;
    cl_mem pitches_mem;
    cl_mem dc_diffs_mem;
    cl_mem rows_mem;
    cl_mem cols_mem;
    cl_mem threads_mem;
} VP8_LOOP_MEM;

VP8_LOOP_MEM loop_mem;

prototype_loopfilter_cl(vp8_loop_filter_horizontal_edge_cl);
prototype_loopfilter_cl(vp8_loop_filter_vertical_edge_cl);
prototype_loopfilter_cl(vp8_mbloop_filter_horizontal_edge_cl);
prototype_loopfilter_cl(vp8_mbloop_filter_vertical_edge_cl);
prototype_loopfilter_cl(vp8_loop_filter_simple_horizontal_edge_cl);
prototype_loopfilter_cl(vp8_loop_filter_simple_vertical_edge_cl);

/* Horizontal MB filtering */
void vp8_loop_filter_mbh_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                            int y_stride, int uv_stride, cl_mem lfi_mem, int filter_level)
{
    int err;
    int offsets[3] = {y_off, u_off, v_off};
    int strides[3] = {y_stride, uv_stride, uv_stride};
    int sizes[3] = {16, 8, 8};
    
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, loop_mem.num_planes * sizeof(cl_int), offsets,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.pitches_mem, loop_mem.num_planes * sizeof(cl_int), strides,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.threads_mem, loop_mem.num_planes * sizeof(cl_int), sizes,,);
    vp8_mbloop_filter_horizontal_edge_cl(x, buf_base, 3, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_TRUE, loop_mem.threads_mem, 16);
}

void vp8_loop_filter_mbhs_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                            int y_stride, int uv_stride, cl_mem lfi_mem, int filter_level)
{
    int err;
    int sizes[] = { 16, 8 };
    (void) uv_stride;

    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, sizeof(cl_int), &y_off,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.pitches_mem, sizeof(cl_int), &y_stride,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.threads_mem, sizeof(cl_int), &sizes[0],,);

    vp8_loop_filter_simple_horizontal_edge_cl(x, buf_base, 1, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_TRUE, loop_mem.threads_mem, 16);
}

/* Vertical MB Filtering */
void vp8_loop_filter_mbv_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                           int y_stride, int uv_stride, cl_mem lfi_mem, int filter_level)
{
    int err;
    int offsets[3] = {y_off, u_off, v_off};
    int strides[3] = {y_stride, uv_stride, uv_stride};
    int sizes[3] = {16, 8, 8};

    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, loop_mem.num_planes * sizeof(cl_int), offsets,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.pitches_mem, loop_mem.num_planes * sizeof(cl_int), strides,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.threads_mem, loop_mem.num_planes * sizeof(cl_int), sizes,,);
    vp8_mbloop_filter_vertical_edge_cl(x, buf_base, 3, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_TRUE, loop_mem.threads_mem, 16);
}

void vp8_loop_filter_mbvs_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                            int y_stride, int uv_stride, cl_mem lfi_mem, int filter_level)
{
    int err;
    int sizes[] = { 16, 8 };
    (void) uv_stride;

    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, sizeof(cl_int), &y_off,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.pitches_mem, sizeof(cl_int), &y_stride,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.threads_mem, sizeof(cl_int), &sizes[0],,);

    vp8_loop_filter_simple_vertical_edge_cl(x, buf_base, 1, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_TRUE, loop_mem.threads_mem, 16);
}

/* Horizontal B Filtering */
void vp8_loop_filter_bh_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                          int y_stride, int uv_stride, cl_mem lfi_mem, int filter_level)
{

    int err;
    int off;
    int sizes[] = { 16, 8 };

    int offsets[3] = {y_off + 4*y_stride, u_off + 4*uv_stride, v_off + 4*uv_stride};
    int strides[3] = {y_stride, uv_stride, uv_stride};
    int size[3] = {16, 8, 8};

    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, loop_mem.num_planes * sizeof(cl_int), offsets,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.pitches_mem, loop_mem.num_planes * sizeof(cl_int), strides,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.threads_mem, loop_mem.num_planes * sizeof(cl_int), size,,);
    vp8_loop_filter_horizontal_edge_cl(x, buf_base, 3, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_mem, 16);

    VP8_CL_SET_BUF(x->cl_commands, loop_mem.pitches_mem, sizeof(cl_int), &y_stride,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.threads_mem, sizeof(cl_int), &sizes[0],,);

    off = y_off + 8*y_stride;
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, sizeof(cl_int), &off,,);
    vp8_loop_filter_horizontal_edge_cl(x, buf_base, 1, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_mem, 16);

    off = y_off + 12 * y_stride;
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, sizeof(cl_int), &off,,);
    vp8_loop_filter_horizontal_edge_cl(x, buf_base, 1, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_mem, 16);

}

void vp8_loop_filter_bhs_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                           int y_stride, int uv_stride, cl_mem lfi_mem, int filter_level)
{
    int err;
    int sizes[] = { 16, 8 };
    int off;
    (void) uv_stride;

    VP8_CL_SET_BUF(x->cl_commands, loop_mem.pitches_mem, sizeof(cl_int), &y_stride,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.threads_mem, sizeof(cl_int), &sizes[0],,);

    off = y_off + 4 * y_stride;
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, sizeof(cl_int), &off,,);
    vp8_loop_filter_simple_horizontal_edge_cl(x, buf_base, 1, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_mem, 16);

    off = y_off + 8 * y_stride;
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, sizeof(cl_int), &off,,);
    vp8_loop_filter_simple_horizontal_edge_cl(x, buf_base, 1, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_mem, 16);

    off = y_off + 12 * y_stride;
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, sizeof(cl_int), &off,,);
    vp8_loop_filter_simple_horizontal_edge_cl(x, buf_base, 1, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_mem, 16);
}

/* Vertical B Filtering */
void vp8_loop_filter_bv_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                          int y_stride, int uv_stride, cl_mem lfi_mem, int filter_level)
{
    int err;
    int sizes[] = { 16, 8 };
    int off;

    int offsets[3] = {y_off + 4, u_off + 4, v_off + 4};
    int strides[3] = {y_stride, uv_stride, uv_stride};
    int size[3] = {16, 8, 8};
    
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, loop_mem.num_planes * sizeof(cl_int), offsets,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.pitches_mem, loop_mem.num_planes * sizeof(cl_int), strides,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.threads_mem, loop_mem.num_planes * sizeof(cl_int), size,,);
    vp8_loop_filter_vertical_edge_cl(x, buf_base, 3, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_mem, 16);

    VP8_CL_SET_BUF(x->cl_commands, loop_mem.pitches_mem, sizeof(cl_int), &y_stride,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.threads_mem, sizeof(cl_int), &sizes[0],,);

    off = y_off + 8;
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, sizeof(cl_int), &off,,);
    vp8_loop_filter_vertical_edge_cl(x, buf_base, 1, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_mem, 16);

    off = y_off + 12;
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, sizeof(cl_int), &off,,);
    vp8_loop_filter_vertical_edge_cl(x, buf_base, 1, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_mem, 16);
    
}

void vp8_loop_filter_bvs_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                           int y_stride, int uv_stride, cl_mem lfi_mem, int filter_level)
{
    int err;
    int sizes[] = { 16, 8 };
    int off = y_off + 4;
    (void) uv_stride;

    VP8_CL_SET_BUF(x->cl_commands, loop_mem.pitches_mem, sizeof(cl_int), &y_stride,,);
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.threads_mem, sizeof(cl_int), &sizes[0],,);
    
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, sizeof(cl_int), &off,,);
    vp8_loop_filter_simple_vertical_edge_cl(x, buf_base, 1, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_mem, 16);

    off = y_off + 8;
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, sizeof(cl_int), &off,,);
    vp8_loop_filter_simple_vertical_edge_cl(x, buf_base, 1, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_mem, 16);

    off = y_off + 12;
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, sizeof(cl_int), &off,,);
    vp8_loop_filter_simple_vertical_edge_cl(x, buf_base, 1, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_mem, 16);
}

void vp8_init_loop_filter_cl(VP8_COMMON *cm)
{
    loop_filter_info *lfi = cm->lf_info;
    int sharpness_lvl = cm->sharpness_level;
    int frame_type = cm->frame_type;
    int i, j;

    int block_inside_limit = 0;
    int HEVThresh;
    const int yhedge_boost  = 2;

    /* For each possible value for the loop filter fill out a "loop_filter_info" entry. */
    for (i = 0; i <= MAX_LOOP_FILTER; i++)
    {
        int filt_lvl = i;

        if (frame_type == KEY_FRAME)
        {
            if (filt_lvl >= 40)
                HEVThresh = 2;
            else if (filt_lvl >= 15)
                HEVThresh = 1;
            else
                HEVThresh = 0;
        }
        else
        {
            if (filt_lvl >= 40)
                HEVThresh = 3;
            else if (filt_lvl >= 20)
                HEVThresh = 2;
            else if (filt_lvl >= 15)
                HEVThresh = 1;
            else
                HEVThresh = 0;
        }

        /* Set loop filter paramaeters that control sharpness. */
        block_inside_limit = filt_lvl >> (sharpness_lvl > 0);
        block_inside_limit = block_inside_limit >> (sharpness_lvl > 4);

        if (sharpness_lvl > 0)
        {
            if (block_inside_limit > (9 - sharpness_lvl))
                block_inside_limit = (9 - sharpness_lvl);
        }

        if (block_inside_limit < 1)
            block_inside_limit = 1;

        for (j = 0; j < 16; j++)
        {
            lfi[i].lim[j] = block_inside_limit;
            lfi[i].mbflim[j] = filt_lvl + yhedge_boost;
            lfi[i].flim[j] = filt_lvl;
            lfi[i].thr[j] = HEVThresh;
        }
    }
}

/* Put vp8_init_loop_filter() in vp8dx_create_decompressor(). Only call vp8_frame_init_loop_filter() while decoding
 * each frame. Check last_frame_type to skip the function most of times.
 */
void vp8_frame_init_loop_filter_cl(loop_filter_info *lfi, int frame_type)
{
    int HEVThresh;
    int i, j;

    /* For each possible value for the loop filter fill out a "loop_filter_info" entry. */
    for (i = 0; i <= MAX_LOOP_FILTER; i++)
    {
        int filt_lvl = i;

        if (frame_type == KEY_FRAME)
        {
            if (filt_lvl >= 40)
                HEVThresh = 2;
            else if (filt_lvl >= 15)
                HEVThresh = 1;
            else
                HEVThresh = 0;
        }
        else
        {
            if (filt_lvl >= 40)
                HEVThresh = 3;
            else if (filt_lvl >= 20)
                HEVThresh = 2;
            else if (filt_lvl >= 15)
                HEVThresh = 1;
            else
                HEVThresh = 0;
        }

        for (j = 0; j < 16; j++)
        {
            lfi[i].thr[j] = HEVThresh;
        }
    }
}


//This might not need to be copied from loopfilter.c
void vp8_adjust_mb_lf_value_cl(MACROBLOCKD *mbd, int *filter_level)
{
    MB_MODE_INFO *mbmi = &mbd->mode_info_context->mbmi;

    if (mbd->mode_ref_lf_delta_enabled)
    {
        /* Apply delta for reference frame */
        *filter_level += mbd->ref_lf_deltas[mbmi->ref_frame];

        /* Apply delta for mode */
        if (mbmi->ref_frame == INTRA_FRAME)
        {
            /* Only the split mode BPRED has a further special case */
            if (mbmi->mode == B_PRED)
                *filter_level +=  mbd->mode_lf_deltas[0];
        }
        else
        {
            /* Zero motion mode */
            if (mbmi->mode == ZEROMV)
                *filter_level +=  mbd->mode_lf_deltas[1];

            /* Split MB motion mode */
            else if (mbmi->mode == SPLITMV)
                *filter_level +=  mbd->mode_lf_deltas[3];

            /* All other inter motion modes (Nearest, Near, New) */
            else
                *filter_level +=  mbd->mode_lf_deltas[2];
        }

        /* Range check */
        if (*filter_level > MAX_LOOP_FILTER)
            *filter_level = MAX_LOOP_FILTER;
        else if (*filter_level < 0)
            *filter_level = 0;
    }
}

int cl_free_loop_mem(){
    int err = 0;
    if (loop_mem.dc_diffs_mem != NULL) err |= clReleaseMemObject(loop_mem.dc_diffs_mem);
    if (loop_mem.offsets_mem != NULL) err |= clReleaseMemObject(loop_mem.offsets_mem);
    if (loop_mem.pitches_mem != NULL) err |= clReleaseMemObject(loop_mem.pitches_mem);
    if (loop_mem.threads_mem != NULL) err |= clReleaseMemObject(loop_mem.threads_mem);
    if (loop_mem.rows_mem != NULL) err |= clReleaseMemObject(loop_mem.rows_mem);
    if (loop_mem.cols_mem != NULL) err |= clReleaseMemObject(loop_mem.cols_mem);
    loop_mem.dc_diffs_mem = NULL;
    loop_mem.offsets_mem = NULL;
    loop_mem.pitches_mem = NULL;
    loop_mem.threads_mem = NULL;
    loop_mem.rows_mem = NULL;
    loop_mem.cols_mem = NULL;

    loop_mem.num_blocks = 0;
    loop_mem.num_planes = 0;

    return err;
}

int cl_grow_loop_mem(int num_blocks, int num_planes){

    int err;
    int num_iter = num_blocks * num_planes;

    //Don't reallocate if the memory is already large enough
    if (num_iter < loop_mem.num_blocks*loop_mem.num_planes)
        return CL_SUCCESS;

    //free all first.
    cl_free_loop_mem();

    //Now re-allocate the memory in the right size
    loop_mem.dc_diffs_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(cl_int)*num_iter, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    loop_mem.offsets_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(cl_int)*num_iter, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    loop_mem.pitches_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(cl_int)*num_iter, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    loop_mem.threads_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(cl_int)*num_iter, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    loop_mem.rows_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(cl_int)*num_iter, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    loop_mem.cols_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(cl_int)*num_iter, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }

    loop_mem.num_blocks = num_blocks;
    loop_mem.num_planes = num_planes;

    return CL_SUCCESS;
}

//Start of externally callable functions.

int cl_init_loop_filter() {
    int err;

    // Create the filter compute program from the file-defined source code
    if ( cl_load_program(&cl_data.loop_filter_program, loop_filter_cl_file_name,
            loopFilterCompileOptions) != CL_SUCCESS )
        return VP8_CL_TRIED_BUT_FAILED;

    // Create the compute kernels in the program we wish to run
    VP8_CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_loop_filter_horizontal_edge_kernel,"vp8_loop_filter_horizontal_edge_kernel");
    VP8_CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_loop_filter_vertical_edge_kernel,"vp8_loop_filter_vertical_edge_kernel");
    VP8_CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_mbloop_filter_horizontal_edge_kernel,"vp8_mbloop_filter_horizontal_edge_kernel");
    VP8_CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_mbloop_filter_vertical_edge_kernel,"vp8_mbloop_filter_vertical_edge_kernel");
    VP8_CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_loop_filter_simple_horizontal_edge_kernel,"vp8_loop_filter_simple_horizontal_edge_kernel");
    VP8_CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_loop_filter_simple_vertical_edge_kernel,"vp8_loop_filter_simple_vertical_edge_kernel");

    loop_mem.num_blocks = 0;
    loop_mem.num_planes = 0;
    loop_mem.dc_diffs_mem = NULL;
    loop_mem.offsets_mem = NULL;
    loop_mem.pitches_mem = NULL;
    loop_mem.threads_mem = NULL;
    loop_mem.rows_mem = NULL;
    loop_mem.cols_mem = NULL;

    return CL_SUCCESS;
}

void cl_destroy_loop_filter(){

    if (cl_data.loop_filter_program)
        clReleaseProgram(cl_data.loop_filter_program);

    VP8_CL_RELEASE_KERNEL(cl_data.vp8_loop_filter_horizontal_edge_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_loop_filter_vertical_edge_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_mbloop_filter_horizontal_edge_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_mbloop_filter_vertical_edge_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_loop_filter_simple_horizontal_edge_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_loop_filter_simple_vertical_edge_kernel);

    cl_free_loop_mem();

    cl_data.loop_filter_program = NULL;
}


void vp8_loop_filter_set_baselines_cl(MACROBLOCKD *mbd, int default_filt_lvl, int *baseline_filter_level){
    int alt_flt_enabled = mbd->segmentation_enabled;
    int i;

    if (alt_flt_enabled)
    {
        for (i = 0; i < MAX_MB_SEGMENTS; i++)
        {
            /* Abs value */
            if (mbd->mb_segement_abs_delta == SEGMENT_ABSDATA)
                baseline_filter_level[i] = mbd->segment_feature_data[MB_LVL_ALT_LF][i];
            /* Delta Value */
            else
            {
                baseline_filter_level[i] = default_filt_lvl + mbd->segment_feature_data[MB_LVL_ALT_LF][i];
                baseline_filter_level[i] = (baseline_filter_level[i] >= 0) ? ((baseline_filter_level[i] <= MAX_LOOP_FILTER) ? baseline_filter_level[i] : MAX_LOOP_FILTER) : 0;  /* Clamp to valid range */
            }
        }
    }
    else
    {
        for (i = 0; i < MAX_MB_SEGMENTS; i++)
            baseline_filter_level[i] = default_filt_lvl;
    }
}

//Note: Assume that mbd->mode_info_context is set for this macroblock
int vp8_loop_filter_level(MACROBLOCKD *mbd, int baseline_filter_level[] ){
    
    int Segment = (mbd->segmentation_enabled) ? mbd->mode_info_context->mbmi.segment_id : 0;

    return vp8_adjust_mb_lf_value(mbd, baseline_filter_level[Segment]);
}

void vp8_loop_filter_macroblock_cl(int mb_row, int mb_col, int dc_diff, int y_off, int u_off, int v_off, int filter_level, VP8_COMMON *cm,
        MACROBLOCKD *mbd,  cl_mem lfi_mem, YV12_BUFFER_CONFIG *post)
{
    int err, ret;
    LOOPFILTERTYPE filter_type = cm->filter_type;

    cl_grow_loop_mem(1, 3);

    if (filter_type == NORMAL_LOOPFILTER){
        if (filter_level){
            if (mb_col > 0){
                vp8_loop_filter_mbv_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, lfi_mem, filter_level);
            }
            if (dc_diff > 0){
                vp8_loop_filter_bv_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, lfi_mem, filter_level);
            }
            if (mb_row > 0){
                vp8_loop_filter_mbh_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, lfi_mem, filter_level);
            }
            if (dc_diff > 0){
                vp8_loop_filter_bh_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, lfi_mem, filter_level);
            }
        }
    } else {
        if (filter_level){
            if (mb_col > 0){
                vp8_loop_filter_mbvs_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, lfi_mem, filter_level);
            }
            if (dc_diff > 0){
                vp8_loop_filter_bvs_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, lfi_mem, filter_level);
            }
            if (mb_row > 0){
                vp8_loop_filter_mbhs_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, lfi_mem, filter_level);
            }
            if (dc_diff > 0){
                vp8_loop_filter_bhs_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, lfi_mem, filter_level);
            }
        }
    }
}

void vp8_loop_filter_add_macroblock_cl(VP8_COMMON *cm, int mb_row, int mb_col,
        MACROBLOCKD *mbd, YV12_BUFFER_CONFIG *post, int row[], int col[], int dc_diffs[], int y_offsets[], int u_offsets[], int v_offsets[],
        int filter_levels[], int baseline_filter_level[], int pos)
{
    int y_offset = 16 * (mb_col + (mb_row*cm->mb_cols)) + mb_row * (post->y_stride * 16 - post->y_width);
    int uv_offset = 8 * (mb_col + (mb_row*cm->mb_cols)) + mb_row * (post->uv_stride * 8 - post->uv_width);

    unsigned char *buf_base = post->buffer_alloc;
    y_offsets[pos] = post->y_buffer - buf_base + y_offset;;
    u_offsets[pos] = post->u_buffer - buf_base + uv_offset;
    v_offsets[pos] = post->v_buffer - buf_base + uv_offset;

    mbd->mode_info_context = cm->mi + ((mb_row * (cm->mb_cols+1) + mb_col));

    /* Distance of Mb to the various image edges.
     * These specified to 8th pel as they are always compared to values that are in 1/8th pel units
     * Apply any context driven MB level adjustment
     */
    filter_levels[pos] = vp8_loop_filter_level(mbd, baseline_filter_level);

    row[pos] = mb_row;
    col[pos] = mb_col;
    dc_diffs[pos] = mbd->mode_info_context->mbmi.dc_diff;
}

void vp8_loop_filter_priority_cl(int priority, VP8_COMMON *cm, MACROBLOCKD *mbd, cl_mem lfi_mem, int baseline_filter_level[],
        YV12_BUFFER_CONFIG *post )
{
    int mb_row, mb_col, mb_cols = cm->mb_cols, mb_rows = cm->mb_rows;

    int current_pos = 0, i;
    int max_size = (mb_cols > mb_rows) ? mb_rows : mb_cols; //Min(height,width)
    int y_offsets[max_size];
    int u_offsets[max_size];
    int v_offsets[max_size];
    int filter_levels[max_size];
    int rows[max_size];
    int cols[max_size];
    int dc_diffs[max_size];

    //Calculate offsets/filter_levels/dc_diffs for all MBs in current priority
    for (mb_row = 0; mb_row <= priority && mb_row < cm->mb_rows; mb_row++){
        //First row is done left to right, subsequent rows are offset two
        //to the left to prevent corruption of a pure diagonal scan that
        //is offset by 1.
        if (mb_row == 0){
            mb_col = priority - mb_row;
        } else {
            mb_col = priority - 2 * mb_row;
        }

        //Skip non-existant MBs
        if ((mb_col > -1 && (mb_col < mb_cols)) && (mb_row < cm->mb_rows)){
            vp8_loop_filter_add_macroblock_cl(cm, mb_row, mb_col,
                mbd, post, rows, cols, dc_diffs, y_offsets, u_offsets, v_offsets,
                filter_levels, baseline_filter_level, current_pos++
            );
        }
    }

    //Process all of the MBs in the current priority
    for (i = 0; i < current_pos; i++){
        vp8_loop_filter_macroblock_cl(rows[i], cols[i], dc_diffs[i],
            y_offsets[i], u_offsets[i], v_offsets[i],
            filter_levels[i], cm, mbd, lfi_mem, post
        );
    }

}

void vp8_loop_filter_frame_cl
(
    VP8_COMMON *cm,
    MACROBLOCKD *mbd,
    int default_filt_lvl
)
{
    YV12_BUFFER_CONFIG *post = cm->frame_to_show;
    loop_filter_info *lfi = cm->lf_info;
    FRAME_TYPE frame_type = cm->frame_type;

    int baseline_filter_level[MAX_MB_SEGMENTS];
    int err, priority;

    cl_mem lfi_mem;
    VP8_CL_CREATE_BUF(mbd->cl_commands, lfi_mem, , sizeof(loop_filter_info)*(MAX_LOOP_FILTER+1), cm->lf_info,, );

    mbd->mode_info_context = cm->mi; /* Point at base of Mb MODE_INFO list */

    /* Note the baseline filter values for each segment */
    vp8_loop_filter_set_baselines_cl(mbd, default_filt_lvl, baseline_filter_level);

    /* Initialize the loop filter for this frame. */
    if ((cm->last_filter_type != cm->filter_type) || (cm->last_sharpness_level != cm->sharpness_level))
        vp8_init_loop_filter_cl(cm);
    else if (frame_type != cm->last_frame_type)
        vp8_frame_init_loop_filter_cl(lfi, frame_type);

    VP8_CL_SET_BUF(mbd->cl_commands, post->buffer_mem, post->buffer_size, post->buffer_alloc,
            vp8_loop_filter_frame(cm,mbd,default_filt_lvl),);

    //Maximum priority = 2*(Height-1) + Width in Macroblocks
    for (priority = 0; priority < 2 * (cm->mb_rows - 1) + cm->mb_cols ; priority++){
        vp8_loop_filter_priority_cl(priority, cm, mbd, lfi_mem, baseline_filter_level, post);
    }

    //Retrieve buffer contents
    err = clEnqueueReadBuffer(mbd->cl_commands, post->buffer_mem, CL_FALSE, 0, post->buffer_size, post->buffer_alloc, 0, NULL, NULL);
    VP8_CL_CHECK_SUCCESS(mbd->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read loop filter output!\n",
        ,
    );

    clReleaseMemObject(lfi_mem);
    
    VP8_CL_FINISH(mbd->cl_commands);
}
