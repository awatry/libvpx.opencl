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

prototype_loopfilter_cl(vp8_loop_filter_horizontal_edge_cl);
prototype_loopfilter_cl(vp8_loop_filter_vertical_edge_cl);
prototype_loopfilter_cl(vp8_mbloop_filter_horizontal_edge_cl);
prototype_loopfilter_cl(vp8_mbloop_filter_vertical_edge_cl);
prototype_loopfilter_cl(vp8_loop_filter_simple_horizontal_edge_cl);
prototype_loopfilter_cl(vp8_loop_filter_simple_vertical_edge_cl);

/* Horizontal MB filtering */
void vp8_loop_filter_mbh_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                            int y_stride, int uv_stride, loop_filter_info *lfi, int filter_level)
{
    vp8_mbloop_filter_horizontal_edge_cl(x, buf_base, y_off, y_stride, lfi->mbflim, lfi->lim, lfi->thr, 2, 1);
    vp8_mbloop_filter_horizontal_edge_cl(x, buf_base, u_off, uv_stride, lfi->mbflim, lfi->lim, lfi->thr, 1, 1);
    vp8_mbloop_filter_horizontal_edge_cl(x, buf_base, v_off, uv_stride, lfi->mbflim, lfi->lim, lfi->thr, 1, 1);
}

void vp8_loop_filter_mbhs_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                            int y_stride, int uv_stride, loop_filter_info *lfi, int filter_level)
{
    (void) uv_stride;
    vp8_loop_filter_simple_horizontal_edge_cl(x, buf_base, y_off, y_stride, lfi->mbflim, lfi->lim, lfi->thr, 2, 1);
}

/* Vertical MB Filtering */
void vp8_loop_filter_mbv_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                           int y_stride, int uv_stride, loop_filter_info *lfi, int filter_level)
{

    vp8_mbloop_filter_vertical_edge_cl(x, buf_base, y_off, y_stride, lfi->mbflim, lfi->lim, lfi->thr, 2, 1);
    vp8_mbloop_filter_vertical_edge_cl(x, buf_base, u_off, uv_stride, lfi->mbflim, lfi->lim, lfi->thr, 1, 1);
    vp8_mbloop_filter_vertical_edge_cl(x, buf_base, v_off, uv_stride, lfi->mbflim, lfi->lim, lfi->thr, 1, 1);
}

void vp8_loop_filter_mbvs_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                            int y_stride, int uv_stride, loop_filter_info *lfi, int filter_level)
{
    (void) uv_stride;
    vp8_loop_filter_simple_vertical_edge_cl(x, buf_base, y_off, y_stride, lfi->mbflim, lfi->lim, lfi->thr, 2, 1);
}

/* Horizontal B Filtering */
void vp8_loop_filter_bh_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                          int y_stride, int uv_stride, loop_filter_info *lfi, int filter_level)
{

    vp8_loop_filter_horizontal_edge_cl(x, buf_base, y_off + 4 * y_stride, y_stride, lfi->flim, lfi->lim, lfi->thr, 2, 1);
    vp8_loop_filter_horizontal_edge_cl(x, buf_base, y_off + 8 * y_stride, y_stride, lfi->flim, lfi->lim, lfi->thr, 2, 1);
    vp8_loop_filter_horizontal_edge_cl(x, buf_base, y_off + 12 * y_stride, y_stride, lfi->flim, lfi->lim, lfi->thr, 2, 1);
    vp8_loop_filter_horizontal_edge_cl(x, buf_base, u_off + 4 * uv_stride, uv_stride, lfi->flim, lfi->lim, lfi->thr, 1, 1);
    vp8_loop_filter_horizontal_edge_cl(x, buf_base, v_off + 4 * uv_stride, uv_stride, lfi->flim, lfi->lim, lfi->thr, 1, 1);

}

void vp8_loop_filter_bhs_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                           int y_stride, int uv_stride, loop_filter_info *lfi, int filter_level)
{
    (void) uv_stride;

    vp8_loop_filter_simple_horizontal_edge_cl(x, buf_base, y_off + 4 * y_stride, y_stride, lfi->flim, lfi->lim, lfi->thr, 2, 1);
    vp8_loop_filter_simple_horizontal_edge_cl(x, buf_base, y_off + 8 * y_stride, y_stride, lfi->flim, lfi->lim, lfi->thr, 2, 1);
    vp8_loop_filter_simple_horizontal_edge_cl(x, buf_base, y_off + 12 * y_stride, y_stride, lfi->flim, lfi->lim, lfi->thr, 2, 1);
}

/* Vertical B Filtering */
void vp8_loop_filter_bv_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                          int y_stride, int uv_stride, loop_filter_info *lfi, int filter_level)
{

    vp8_loop_filter_vertical_edge_cl(x, buf_base, y_off + 4, y_stride, lfi->flim, lfi->lim, lfi->thr, 2, 1);
    vp8_loop_filter_vertical_edge_cl(x, buf_base, y_off + 8, y_stride, lfi->flim, lfi->lim, lfi->thr, 2, 1);
    vp8_loop_filter_vertical_edge_cl(x, buf_base, y_off + 12, y_stride, lfi->flim, lfi->lim, lfi->thr, 2, 1);

    vp8_loop_filter_vertical_edge_cl(x, buf_base, u_off + 4, uv_stride, lfi->flim, lfi->lim, lfi->thr, 1, 1);
    vp8_loop_filter_vertical_edge_cl(x, buf_base, v_off + 4, uv_stride, lfi->flim, lfi->lim, lfi->thr, 1, 1);
}

void vp8_loop_filter_bvs_cl(MACROBLOCKD *x, cl_mem buf_base, int y_off, int u_off, int v_off,
                           int y_stride, int uv_stride, loop_filter_info *lfi, int filter_level)
{
    (void) uv_stride;

    vp8_loop_filter_simple_vertical_edge_cl(x, buf_base, y_off + 4, y_stride, lfi->flim, lfi->lim, lfi->thr, 2, 1);
    vp8_loop_filter_simple_vertical_edge_cl(x, buf_base, y_off + 8, y_stride, lfi->flim, lfi->lim, lfi->thr, 2, 1);
    vp8_loop_filter_simple_vertical_edge_cl(x, buf_base, y_off + 12, y_stride, lfi->flim, lfi->lim, lfi->thr, 2, 1);
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

void vp8_loop_filter_macroblock_cl(int mb_row, int mb_col, VP8_COMMON *cm,
        MACROBLOCKD *mbd, int baseline_filter_level[],
        YV12_BUFFER_CONFIG *post) {

    int mb_cols = cm->mb_cols;
    loop_filter_info *lfi = cm->lf_info;
    int filter_level;
    LOOPFILTERTYPE filter_type = cm->filter_type;

    int y_offset = 16 * (mb_col + (mb_row*mb_cols)) + mb_row * (post->y_stride * 16 - post->y_width);
    int uv_offset = 8 * (mb_col + (mb_row*mb_cols)) + mb_row * (post->uv_stride * 8 - post->uv_width);

    unsigned char *buf_base = post->buffer_alloc;
    int y_off = post->y_buffer - buf_base + y_offset;
    int u_off = post->u_buffer - buf_base + uv_offset;
    int v_off = post->v_buffer - buf_base + uv_offset;

    mbd->mode_info_context = cm->mi + ((mb_row * (mb_cols+1) + mb_col));

    /* Distance of Mb to the various image edges.
     * These specified to 8th pel as they are always compared to values that are in 1/8th pel units
     * Apply any context driven MB level adjustment
     */
    filter_level = vp8_loop_filter_level(mbd, baseline_filter_level);

    if (filter_level)
    {
        if (mb_col > 0){
            if (filter_type == NORMAL_LOOPFILTER)
                vp8_loop_filter_mbv_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, &lfi[filter_level], filter_level);
            else
                vp8_loop_filter_mbvs_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, &lfi[filter_level], filter_level);
        }

        if (mbd->mode_info_context->mbmi.dc_diff > 0){
            if (filter_type == NORMAL_LOOPFILTER)
                vp8_loop_filter_bv_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, &lfi[filter_level], filter_level);
            else
                vp8_loop_filter_bvs_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, &lfi[filter_level], filter_level);
        }

        /* don't apply across umv border */
        if (mb_row > 0){
            if (filter_type == NORMAL_LOOPFILTER)
                vp8_loop_filter_mbh_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, &lfi[filter_level], filter_level);
            else
                vp8_loop_filter_mbhs_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, &lfi[filter_level], filter_level);
        }

        if (mbd->mode_info_context->mbmi.dc_diff > 0){
            if (filter_type == NORMAL_LOOPFILTER)
                vp8_loop_filter_bh_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, &lfi[filter_level], filter_level);
            else
                vp8_loop_filter_bhs_cl(mbd, post->buffer_mem, y_off, u_off, v_off, post->y_stride, post->uv_stride, &lfi[filter_level], filter_level);
        }
    }
}


void vp8_loop_filter_priority_cl(int priority, VP8_COMMON *cm, MACROBLOCKD *mbd, int baseline_filter_level[],
        YV12_BUFFER_CONFIG *post )
{
    int mb_row, mb_col;    
    
    //Process all MBs in current priority
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
        if ((mb_col > -1 && (mb_col < cm->mb_cols)) && (mb_row < cm->mb_rows)){
            //printf("Loop filter for row %d col %d\n", mb_row, mb_col);
            vp8_loop_filter_macroblock_cl(mb_row, mb_col, cm, mbd, baseline_filter_level, post);
        }
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
        vp8_loop_filter_priority_cl(priority, cm, mbd, baseline_filter_level, post);
    }

    //Retrieve buffer contents
    err = clEnqueueReadBuffer(mbd->cl_commands, post->buffer_mem, CL_FALSE, 0, post->buffer_size, post->buffer_alloc, 0, NULL, NULL);
    VP8_CL_CHECK_SUCCESS(mbd->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read loop filter output!\n",
        ,
    );

    VP8_CL_FINISH(mbd->cl_commands);
}
