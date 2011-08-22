
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
    cl_mem threads_yuv_mem;
    cl_mem threads_y_mem;
    cl_mem filter_levels_mem;
    cl_mem dc_diffs_mem;
    cl_mem rows_mem;
    cl_mem cols_mem;
} VP8_LOOP_MEM;

VP8_LOOP_MEM loop_mem;

prototype_loopfilter_cl(vp8_loop_filter_horizontal_edge_cl);
prototype_loopfilter_cl(vp8_loop_filter_vertical_edge_cl);
prototype_loopfilter_cl(vp8_mbloop_filter_horizontal_edge_cl);
prototype_loopfilter_cl(vp8_mbloop_filter_vertical_edge_cl);
prototype_loopfilter_cl(vp8_loop_filter_simple_horizontal_edge_cl);
prototype_loopfilter_cl(vp8_loop_filter_simple_vertical_edge_cl);

/* Horizontal MB filtering */
void vp8_loop_filter_mbh_cl(MACROBLOCKD *x, cl_mem buf_base, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                            int y_stride, int uv_stride, cl_mem lfi_mem, cl_mem filter_level, cl_mem apply_filter)
{
    int err;
    int block;

    cl_int offsets[num_blocks][3];
    for( block = 0; block < num_blocks; block++){
        offsets[block][0] = y_offsets[block];
        offsets[block][1] = u_offsets[block];
        offsets[block][2] = v_offsets[block];
    }
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, num_blocks * 3 * sizeof(cl_int), offsets,,);

    vp8_mbloop_filter_horizontal_edge_cl(x, buf_base, 3, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_TRUE, loop_mem.threads_yuv_mem, 16, apply_filter);
}

void vp8_loop_filter_mbhs_cl(MACROBLOCKD *x, cl_mem buf_base, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                            int y_stride, int uv_stride, cl_mem lfi_mem, cl_mem filter_level, cl_mem apply_filter)
{
    int err;

    cl_int offsets[num_blocks], block;
    for( block = 0; block < num_blocks; block++){
        offsets[block] = y_offsets[block];
    }

    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, num_blocks * sizeof(cl_int), offsets,,);
    vp8_loop_filter_simple_horizontal_edge_cl(x, buf_base, 1, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_TRUE, loop_mem.threads_y_mem, 16, apply_filter);
}

/* Vertical MB Filtering */
void vp8_loop_filter_mbv_cl(MACROBLOCKD *x, cl_mem buf_base, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                           int y_stride, int uv_stride, cl_mem lfi_mem, cl_mem filter_level, cl_mem apply_filter)
{
    int err, block;

    cl_int offsets[num_blocks][3];
    for( block = 0; block < num_blocks; block++){
        offsets[block][0] = y_offsets[block];
        offsets[block][1] = u_offsets[block];
        offsets[block][2] = v_offsets[block];
    }
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, 3 * num_blocks * sizeof(cl_int), offsets,,);

    vp8_mbloop_filter_vertical_edge_cl(x, buf_base, 3, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_TRUE, loop_mem.threads_yuv_mem, 16, apply_filter);
}

void vp8_loop_filter_mbvs_cl(MACROBLOCKD *x, cl_mem buf_base, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                            int y_stride, int uv_stride, cl_mem lfi_mem, cl_mem filter_level, cl_mem apply_filter)
{
    int err;

    cl_int offsets[num_blocks], block;
    for( block = 0; block < num_blocks; block++){
        offsets[block] = y_offsets[block];
    }
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, num_blocks * sizeof(cl_int), offsets,,);

    vp8_loop_filter_simple_vertical_edge_cl(x, buf_base, 1, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_TRUE, loop_mem.threads_y_mem, 16, apply_filter);
}

/* Horizontal B Filtering */
void vp8_loop_filter_bh_cl(MACROBLOCKD *x, cl_mem buf_base, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                          int y_stride, int uv_stride, cl_mem lfi_mem, cl_mem filter_level, cl_mem apply_filter)
{

    int err;
    
    cl_int yuv_offsets[num_blocks][3], block;
    cl_int offsets[num_blocks];
    for( block = 0; block < num_blocks; block++){
        yuv_offsets[block][0] = y_offsets[block] + 4*y_stride;
        yuv_offsets[block][1] = u_offsets[block] + 4*uv_stride;
        yuv_offsets[block][2] = v_offsets[block] + 4*uv_stride;
    }
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, num_blocks * 3 * sizeof(cl_int), yuv_offsets,,);
    vp8_loop_filter_horizontal_edge_cl(x, buf_base, 3, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_yuv_mem, 16, apply_filter);

    for( block = 0; block < num_blocks; block++){
        offsets[block] = y_offsets[block] + 8*y_stride;
    }
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, num_blocks * sizeof(cl_int), offsets,,);
    vp8_loop_filter_horizontal_edge_cl(x, buf_base, 1, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_y_mem, 16, apply_filter);

    for( block = 0; block < num_blocks; block++){
        offsets[block] = y_offsets[block] + 12*y_stride;
    }
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, num_blocks * sizeof(cl_int), offsets,,);
    vp8_loop_filter_horizontal_edge_cl(x, buf_base, 1, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_y_mem, 16, apply_filter);

}

void vp8_loop_filter_bhs_cl(MACROBLOCKD *x, cl_mem buf_base, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                           int y_stride, int uv_stride, cl_mem lfi_mem, cl_mem filter_level, cl_mem apply_filter)
{
    int err;
    int block;

    cl_int offsets[num_blocks];
    for( block = 0; block < num_blocks; block++){
        offsets[block] = y_offsets[block] + 4*y_stride;
    }
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, num_blocks * sizeof(cl_int), offsets,,);
    vp8_loop_filter_simple_horizontal_edge_cl(x, buf_base, 1, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_y_mem, 16, apply_filter);

    for( block = 0; block < num_blocks; block++){
        offsets[block] = y_offsets[block] + 8*y_stride;
    }
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, num_blocks * sizeof(cl_int), offsets,,);
    vp8_loop_filter_simple_horizontal_edge_cl(x, buf_base, 1, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_y_mem, 16, apply_filter);

    for( block = 0; block < num_blocks; block++){
        offsets[block] = y_offsets[block] + 12*y_stride;
    }
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, num_blocks * sizeof(cl_int), offsets,,);
    vp8_loop_filter_simple_horizontal_edge_cl(x, buf_base, 1, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_y_mem, 16, apply_filter);
}

/* Vertical B Filtering */
void vp8_loop_filter_bv_cl(MACROBLOCKD *x, cl_mem buf_base, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                          int y_stride, int uv_stride, cl_mem lfi_mem, cl_mem filter_level, cl_mem apply_filter)
{
    int err;

    cl_int offsets[num_blocks];
    int block, yuv_offsets[num_blocks][3];
    for( block = 0; block < num_blocks; block++){
        yuv_offsets[block][0] = y_offsets[block] + 4;
        yuv_offsets[block][1] = u_offsets[block] + 4;
        yuv_offsets[block][2] = v_offsets[block] + 4;
    }
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, num_blocks*3*sizeof(cl_int), yuv_offsets,,);

    vp8_loop_filter_vertical_edge_cl(x, buf_base, 3, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_yuv_mem, 16, apply_filter);
    for( block = 0; block < num_blocks; block++){
        offsets[block] = y_offsets[block] + 8;
    }
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, num_blocks * sizeof(cl_int), offsets,,);
    vp8_loop_filter_vertical_edge_cl(x, buf_base, 1, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_y_mem, 16, apply_filter);

    for( block = 0; block < num_blocks; block++){
        offsets[block] = y_offsets[block] + 12;
    }
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, num_blocks * sizeof(cl_int), offsets,,);
    vp8_loop_filter_vertical_edge_cl(x, buf_base, 1, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_y_mem, 16, apply_filter);
    
}

void vp8_loop_filter_bvs_cl(MACROBLOCKD *x, cl_mem buf_base, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                           int y_stride, int uv_stride, cl_mem lfi_mem, cl_mem filter_level, cl_mem apply_filter)
{
    int err;

    int block;
    cl_int offsets[num_blocks];
    for( block = 0; block < num_blocks; block++){
        offsets[block] = y_offsets[block] + 4;
    }
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, num_blocks * sizeof(cl_int), offsets,,);

    vp8_loop_filter_simple_vertical_edge_cl(x, buf_base, 1, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_y_mem, 16, apply_filter);

    for( block = 0; block < num_blocks; block++){
        offsets[block] = y_offsets[block] + 8;
    }
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, num_blocks * sizeof(cl_int), offsets,,);
    vp8_loop_filter_simple_vertical_edge_cl(x, buf_base, 1, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_y_mem, 16, apply_filter);

    for( block = 0; block < num_blocks; block++){
        offsets[block] = y_offsets[block] + 12;
    }
    VP8_CL_SET_BUF(x->cl_commands, loop_mem.offsets_mem, num_blocks * sizeof(cl_int), offsets,,);

    vp8_loop_filter_simple_vertical_edge_cl(x, buf_base, 1, num_blocks, loop_mem.offsets_mem, loop_mem.pitches_mem, lfi_mem, filter_level, CL_FALSE, loop_mem.threads_y_mem, 16, apply_filter);
}

int cl_free_loop_mem(){
    int err = 0;

    if (loop_mem.dc_diffs_mem != NULL) err |= clReleaseMemObject(loop_mem.dc_diffs_mem);
    if (loop_mem.offsets_mem != NULL) err |= clReleaseMemObject(loop_mem.offsets_mem);
    if (loop_mem.pitches_mem != NULL) err |= clReleaseMemObject(loop_mem.pitches_mem);
    if (loop_mem.threads_y_mem != NULL) err |= clReleaseMemObject(loop_mem.threads_y_mem);
    if (loop_mem.threads_yuv_mem != NULL) err |= clReleaseMemObject(loop_mem.threads_yuv_mem);
    if (loop_mem.rows_mem != NULL) err |= clReleaseMemObject(loop_mem.rows_mem);
    if (loop_mem.cols_mem != NULL) err |= clReleaseMemObject(loop_mem.cols_mem);
    if (loop_mem.filter_levels_mem != NULL) err |= clReleaseMemObject(loop_mem.filter_levels_mem);
    loop_mem.dc_diffs_mem = NULL;
    loop_mem.offsets_mem = NULL;
    loop_mem.pitches_mem = NULL;
    loop_mem.threads_y_mem = NULL;
    loop_mem.threads_yuv_mem = NULL;
    loop_mem.rows_mem = NULL;
    loop_mem.cols_mem = NULL;
    loop_mem.filter_levels_mem = NULL;

    loop_mem.num_blocks = 0;
    loop_mem.num_planes = 0;

    return err;
}

int cl_populate_loop_mem(MACROBLOCKD *mbd, YV12_BUFFER_CONFIG *post){
    int err;

    cl_int threads_y[1];
    cl_int threads_yuv[3];
    cl_int pitches[3];

    threads_y[0] = 16;
    threads_yuv[0] = 16;
    threads_yuv[1] = 8;
    threads_yuv[2] = 8;

    pitches[0] = post->y_stride;
    pitches[1] = post->uv_stride;
    pitches[2] = post->uv_stride;


    VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.threads_y_mem, sizeof(cl_int), threads_y, , err);
    VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.threads_yuv_mem, sizeof(cl_int)*3, threads_yuv, , err);
    VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.pitches_mem, sizeof(cl_int)*3, pitches, , err);

    return err;
}

int cl_grow_loop_mem(MACROBLOCKD *mbd, YV12_BUFFER_CONFIG *post, int num_blocks){

    int err;
    int num_planes = 3;
    int num_iter = num_blocks * num_planes;

    //Don't reallocate if the memory is already large enough
    if (num_iter <= loop_mem.num_blocks*loop_mem.num_planes)
        return CL_SUCCESS;

    //free all first.
    cl_free_loop_mem();

    //Now re-allocate the memory in the right size
    loop_mem.dc_diffs_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(cl_int)*num_blocks, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    loop_mem.offsets_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(cl_int)*num_iter, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    loop_mem.pitches_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(cl_int)*3, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    loop_mem.threads_y_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    loop_mem.threads_yuv_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(cl_int)*3, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    loop_mem.rows_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(cl_int)*num_blocks, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    loop_mem.cols_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(cl_int)*num_blocks, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    loop_mem.filter_levels_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(cl_int)*num_blocks, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }

    loop_mem.num_blocks = num_blocks;
    loop_mem.num_planes = num_planes;

    return cl_populate_loop_mem(mbd, post);
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
    loop_mem.threads_y_mem = NULL;
    loop_mem.threads_yuv_mem = NULL;
    loop_mem.rows_mem = NULL;
    loop_mem.cols_mem = NULL;
    loop_mem.filter_levels_mem = NULL;

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

//Note: Assumes that mbd->mode_info_context is set for this macroblock
int vp8_loop_filter_level(MACROBLOCKD *mbd, int baseline_filter_level[] ){
    
    int Segment = (mbd->segmentation_enabled) ? mbd->mode_info_context->mbmi.segment_id : 0;

    return vp8_adjust_mb_lf_value(mbd, baseline_filter_level[Segment]);
}

void vp8_loop_filter_macroblocks_cl(int num_blocks, int mb_rows[], int mb_cols[],
    int dc_diffs[], int *y_offsets, int *u_offsets, int *v_offsets, int *filter_levels,
    VP8_COMMON *cm, MACROBLOCKD *mbd,  cl_mem lfi_mem, YV12_BUFFER_CONFIG *post
)
{
    LOOPFILTERTYPE filter_type = cm->filter_type;
    int err;
    cl_grow_loop_mem(mbd, post, num_blocks);

    //Set the rows, cols, and dc_diffs buffers
    VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.dc_diffs_mem, sizeof(cl_int)*num_blocks, dc_diffs,,)
    VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.cols_mem, sizeof(cl_int)*num_blocks, mb_cols,,)
    VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.rows_mem, sizeof(cl_int)*num_blocks, mb_rows,,)
    VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.filter_levels_mem, sizeof(cl_int)*num_blocks, filter_levels,,)

    if (filter_type == NORMAL_LOOPFILTER){
        vp8_loop_filter_mbv_cl(mbd, post->buffer_mem, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, lfi_mem, loop_mem.filter_levels_mem, loop_mem.cols_mem );
        vp8_loop_filter_bv_cl(mbd, post->buffer_mem, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, lfi_mem, loop_mem.filter_levels_mem, loop_mem.dc_diffs_mem );
        vp8_loop_filter_mbh_cl(mbd, post->buffer_mem, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, lfi_mem, loop_mem.filter_levels_mem, loop_mem.rows_mem );
        vp8_loop_filter_bh_cl(mbd, post->buffer_mem, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, lfi_mem, loop_mem.filter_levels_mem, loop_mem.dc_diffs_mem );
    } else {
        vp8_loop_filter_mbvs_cl(mbd, post->buffer_mem, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, lfi_mem, loop_mem.filter_levels_mem, loop_mem.cols_mem );
        vp8_loop_filter_bvs_cl(mbd, post->buffer_mem, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, lfi_mem, loop_mem.filter_levels_mem, loop_mem.dc_diffs_mem );
        vp8_loop_filter_mbhs_cl(mbd, post->buffer_mem, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, lfi_mem, loop_mem.filter_levels_mem, loop_mem.rows_mem );
        vp8_loop_filter_bhs_cl(mbd, post->buffer_mem, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, lfi_mem, loop_mem.filter_levels_mem, loop_mem.dc_diffs_mem );
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

    int current_pos = 0;
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
    vp8_loop_filter_macroblocks_cl(current_pos, rows, cols, dc_diffs, y_offsets, u_offsets, v_offsets, filter_levels, cm, mbd, lfi_mem, post);
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
    int pitches[3] = {post->y_stride, post->uv_stride, post->uv_stride};

    mbd->mode_info_context = cm->mi; /* Point at base of Mb MODE_INFO list */

    /* Note the baseline filter values for each segment */
    vp8_loop_filter_set_baselines_cl(mbd, default_filt_lvl, baseline_filter_level);

    /* Initialize the loop filter for this frame. */
    if ((cm->last_filter_type != cm->filter_type) || (cm->last_sharpness_level != cm->sharpness_level))
        vp8_init_loop_filter(cm);
    else if (frame_type != cm->last_frame_type)
        vp8_frame_init_loop_filter(lfi, frame_type);

    VP8_CL_CREATE_BUF(mbd->cl_commands, lfi_mem, , sizeof(loop_filter_info)*(MAX_LOOP_FILTER+1), cm->lf_info,, );
    cl_grow_loop_mem(mbd, post, 30); //Default to allocating enough for 480p
    
    VP8_CL_SET_BUF(mbd->cl_commands, post->buffer_mem, post->buffer_size, post->buffer_alloc,
            vp8_loop_filter_frame(cm,mbd,default_filt_lvl),);

    VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.pitches_mem, sizeof(cl_int)*3, pitches, vp8_loop_filter_frame(cm,mbd,default_filt_lvl),);

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
