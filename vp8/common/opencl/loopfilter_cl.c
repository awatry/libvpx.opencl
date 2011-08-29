
/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
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
#include "vpx_mem/vpx_mem.h"
#include "vp8_opencl.h"
#include "blockd_cl.h"

#if NVIDIA
#define USE_MAPPED_BUFFERS 0
#define MAP_PITCHES 0
#define MAP_OFFSETS 0
#else
#define USE_MAPPED_BUFFERS 1
#define MAP_PITCHES 1
#define MAP_OFFSETS 1
#endif

const char *loopFilterCompileOptions = "";
const char *loop_filter_cl_file_name = "vp8/common/opencl/loopfilter";

static int frame_num = 0;
static int prior_y_stride = 0;
static int prior_uv_stride = 0;
static LOOPFILTERTYPE prior_filter_type = 0;
static int prior_mbrows = 0;
static int prior_mbcols = 0;
static int priority_offset;
static cl_int *priority_offsets = NULL;
static int recalculate_offsets = 1;

typedef unsigned char uc;

extern void vp8_loop_filter_frame
(
    VP8_COMMON *cm,
    MACROBLOCKD *mbd,
    int default_filt_lvl
);

typedef struct VP8_LOOP_MEM{
    cl_int num_blocks;
    cl_mem offsets_mem;
    cl_mem pitches_mem;
    cl_mem filters_mem;
} VP8_LOOP_MEM;

VP8_LOOP_MEM loop_mem;
cl_mem lfi_mem = NULL;

prototype_loopfilter_cl(vp8_loop_filter_horizontal_edge_cl);
prototype_loopfilter_cl(vp8_loop_filter_vertical_edge_cl);
prototype_loopfilter_cl(vp8_mbloop_filter_horizontal_edge_cl);
prototype_loopfilter_cl(vp8_mbloop_filter_vertical_edge_cl);
prototype_loopfilter_cl(vp8_loop_filter_simple_horizontal_edge_cl);
prototype_loopfilter_cl(vp8_loop_filter_simple_vertical_edge_cl);

/* Horizontal MB filtering */
void vp8_loop_filter_mbh_cl(MACROBLOCKD *x, VP8_LOOPFILTER_ARGS *args, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                            int y_stride, int uv_stride, cl_int filter_type)
{
    args->filter_type= filter_type;
    args->use_mbflim = CL_TRUE;
    vp8_mbloop_filter_horizontal_edge_cl(x, args, 3, num_blocks, 16);
}

void vp8_loop_filter_mbhs_cl(MACROBLOCKD *x, VP8_LOOPFILTER_ARGS *args, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                            int y_stride, int uv_stride, cl_int filter_type)
{
    args->filter_type= filter_type;
    args->use_mbflim = CL_TRUE;
    args->cur_iter = 4;
    
    vp8_loop_filter_simple_horizontal_edge_cl(x, args, 1, num_blocks, 16);
}

/* Vertical MB Filtering */
void vp8_loop_filter_mbv_cl(MACROBLOCKD *x, VP8_LOOPFILTER_ARGS *args, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                           int y_stride, int uv_stride, cl_int filter_type)
{
    args->filter_type= filter_type;
    args->use_mbflim = CL_TRUE;
    args->cur_iter = 0;
    vp8_mbloop_filter_vertical_edge_cl(x, args, 3, num_blocks, 16);
}

void vp8_loop_filter_mbvs_cl(MACROBLOCKD *x, VP8_LOOPFILTER_ARGS *args, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                            int y_stride, int uv_stride, cl_int filter_type)
{
    args->filter_type= filter_type;
    args->use_mbflim = CL_TRUE;
    args->cur_iter = 0;
    
    vp8_loop_filter_simple_vertical_edge_cl(x, args, 1, num_blocks, 16);
}

/* Horizontal B Filtering */
void vp8_loop_filter_bh_cl(MACROBLOCKD *x, VP8_LOOPFILTER_ARGS *args, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                          int y_stride, int uv_stride, cl_int filter_type)
{
    args->filter_type= filter_type;
    args->use_mbflim = CL_FALSE;
    
    args->cur_iter = 0;
    vp8_loop_filter_horizontal_edge_cl(x, args, 3, num_blocks, 16);
    
    args->cur_iter = 3;
    vp8_loop_filter_horizontal_edge_cl(x, args, 1, num_blocks, 16);
    
    args->cur_iter = 4;
    vp8_loop_filter_horizontal_edge_cl(x, args, 1, num_blocks, 16);

}

void vp8_loop_filter_bhs_cl(MACROBLOCKD *x, VP8_LOOPFILTER_ARGS *args, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                           int y_stride, int uv_stride, cl_int filter_type)
{
    args->filter_type= filter_type;
    args->use_mbflim = CL_FALSE;

    args->cur_iter = 5;
    vp8_loop_filter_simple_horizontal_edge_cl(x, args, 1, num_blocks, 16);
    args->cur_iter = 6;
    vp8_loop_filter_simple_horizontal_edge_cl(x, args, 1, num_blocks, 16);
    args->cur_iter = 7;
    vp8_loop_filter_simple_horizontal_edge_cl(x, args, 1, num_blocks, 16);
}

/* Vertical B Filtering */
void vp8_loop_filter_bv_cl(MACROBLOCKD *x, VP8_LOOPFILTER_ARGS *args, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                          int y_stride, int uv_stride, cl_int filter_type)
{
    args->filter_type= filter_type;
    args->use_mbflim = CL_FALSE;

    //Do YUV, and then two more Y iterations
    args->cur_iter = 1;
    vp8_loop_filter_vertical_edge_cl(x, args, 3, num_blocks, 16);
    
    args->cur_iter = 6;
    vp8_loop_filter_vertical_edge_cl(x, args, 1, num_blocks, 16);
    
    args->cur_iter = 7;
    vp8_loop_filter_vertical_edge_cl(x, args, 1, num_blocks, 16);
}    

void vp8_loop_filter_bvs_cl(MACROBLOCKD *x, VP8_LOOPFILTER_ARGS *args, int num_blocks, int *y_offsets, int *u_offsets, int *v_offsets,
                           int y_stride, int uv_stride, cl_int filter_type)
{
    args->filter_type= filter_type;
    args->use_mbflim = CL_FALSE;

    //3 Y plane iterations
    args->cur_iter = 1;
    vp8_loop_filter_simple_vertical_edge_cl(x, args, 1, num_blocks, 16);
    args->cur_iter = 2;
    vp8_loop_filter_simple_vertical_edge_cl(x, args, 1, num_blocks, 16);
    args->cur_iter = 3;
    vp8_loop_filter_simple_vertical_edge_cl(x, args, 1, num_blocks, 16);
}

int cl_free_loop_mem(){
    int err = 0;

    if (loop_mem.offsets_mem != NULL) err |= clReleaseMemObject(loop_mem.offsets_mem);
    if (loop_mem.pitches_mem != NULL) err |= clReleaseMemObject(loop_mem.pitches_mem);
    if (loop_mem.filters_mem != NULL) err |= clReleaseMemObject(loop_mem.filters_mem);
    loop_mem.offsets_mem = NULL;
    loop_mem.pitches_mem = NULL;
    loop_mem.filters_mem = NULL;

    loop_mem.num_blocks = 0;

    return err;
}

int cl_populate_loop_mem(MACROBLOCKD *mbd, YV12_BUFFER_CONFIG *post){
    int err;

#if USE_MAPPED_BUFFERS
    cl_int *pitches = NULL;
    VP8_CL_MAP_BUF(mbd->cl_commands, loop_mem.pitches_mem, pitches, 3*sizeof(cl_int),,err)
    pitches[0] = post->y_stride;
    pitches[1] = post->uv_stride;
    pitches[2] = post->uv_stride;
    VP8_CL_UNMAP_BUF(mbd->cl_commands, loop_mem.pitches_mem, pitches,,err)
#else
    cl_int pitches[3] = {post->y_stride, post->uv_stride, post->uv_stride};
    VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.pitches_mem, 3*sizeof(cl_int), pitches,,);
#endif
    return err;
}

int cl_grow_loop_mem(MACROBLOCKD *mbd, YV12_BUFFER_CONFIG *post, int num_blocks, VP8_COMMON *cm){

    int err;

    //Don't reallocate if the memory is already large enough
    if (num_blocks <= loop_mem.num_blocks)
        return CL_SUCCESS;

    recalculate_offsets = 1;
    
    //free all first.
    cl_free_loop_mem();

    //Now re-allocate the memory in the right size
    loop_mem.offsets_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, sizeof(cl_int)*cm->MBs*16, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    loop_mem.pitches_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, sizeof(cl_int)*3, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
#if USE_MAPPED_BUFFERS
    loop_mem.filters_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, sizeof(cl_int)*num_blocks*4, NULL, &err);
#else
    loop_mem.filters_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(cl_int)*num_blocks*4, NULL, &err);
#endif
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }

    loop_mem.num_blocks = num_blocks;

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
    
    VP8_CL_CALC_LOCAL_SIZE(cl_data.vp8_loop_filter_horizontal_edge_kernel,&cl_data.vp8_loop_filter_horizontal_edge_kernel_size);
    VP8_CL_CALC_LOCAL_SIZE(cl_data.vp8_loop_filter_vertical_edge_kernel,&cl_data.vp8_loop_filter_vertical_edge_kernel_size);
    VP8_CL_CALC_LOCAL_SIZE(cl_data.vp8_loop_filter_simple_horizontal_edge_kernel,&cl_data.vp8_loop_filter_simple_horizontal_edge_kernel_size);
    VP8_CL_CALC_LOCAL_SIZE(cl_data.vp8_loop_filter_simple_vertical_edge_kernel,&cl_data.vp8_loop_filter_simple_vertical_edge_kernel_size);
    VP8_CL_CALC_LOCAL_SIZE(cl_data.vp8_mbloop_filter_horizontal_edge_kernel,&cl_data.vp8_mbloop_filter_horizontal_edge_kernel_size);
    VP8_CL_CALC_LOCAL_SIZE(cl_data.vp8_mbloop_filter_vertical_edge_kernel,&cl_data.vp8_mbloop_filter_vertical_edge_kernel_size);
    
    loop_mem.num_blocks = 0;
    loop_mem.offsets_mem = NULL;
    loop_mem.pitches_mem = NULL;
    loop_mem.filters_mem = NULL;

    return CL_SUCCESS;
}

void cl_destroy_loop_filter(){

    cl_free_loop_mem();

    if (lfi_mem != NULL){
        clReleaseMemObject(lfi_mem);
        lfi_mem = NULL;
    }
   
    if (priority_offsets != NULL){
        free(priority_offsets);
        priority_offsets = NULL;
    }
    
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_loop_filter_horizontal_edge_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_loop_filter_vertical_edge_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_mbloop_filter_horizontal_edge_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_mbloop_filter_vertical_edge_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_loop_filter_simple_horizontal_edge_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_loop_filter_simple_vertical_edge_kernel);

    if (cl_data.loop_filter_program)
        clReleaseProgram(cl_data.loop_filter_program);
   
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

void vp8_loop_filter_build_offsets(MACROBLOCKD *mbd, int num_blocks, 
        int *y_offsets, int *u_offsets, int *v_offsets, YV12_BUFFER_CONFIG *post, 
        VP8_COMMON *cm, LOOPFILTERTYPE filter_type, int priority_level
)
{
    int y_stride = post->y_stride;
    int uv_stride = post->uv_stride;
    int blk,err;
    cl_int *offsets = NULL;
    
    priority_offsets[priority_level] = priority_offset;
    
    if (filter_type == NORMAL_LOOPFILTER){
        //Map the offsets buffer
        VP8_CL_MAP_BUF(mbd->cl_commands, loop_mem.offsets_mem, offsets, sizeof(cl_int)*cm->MBs*16,,)
                
        //populate it with the correct offsets for current filter type
        for (blk = 0; blk < num_blocks; blk++){
            int y_off, u_off, v_off;
            y_off = y_offsets[blk];
            u_off = u_offsets[blk];
            v_off = v_offsets[blk];
            
            //MBV offsets
            offsets[priority_offset + blk * 3 + 0] = y_off;
            offsets[priority_offset + blk * 3 + 1] = u_off;
            offsets[priority_offset + blk * 3 + 2] = v_off;

            //BV Offsets
            offsets[priority_offset + num_blocks * 3 + blk * 3 + 0] = y_off+4;
            offsets[priority_offset + num_blocks * 3 + blk * 3 + 1] = u_off+4;
            offsets[priority_offset + num_blocks * 3 + blk * 3 + 2] = v_off+4;
            offsets[priority_offset + num_blocks * 6 + blk]         = y_off+8;
            offsets[priority_offset + num_blocks * 7 + blk]         = y_off+12;

            //MBH Offsets
            offsets[priority_offset + num_blocks * 8 + blk * 3 + 0] = y_off;
            offsets[priority_offset + num_blocks * 8 + blk * 3 + 1] = u_off;
            offsets[priority_offset + num_blocks * 8 + blk * 3 + 2] = v_off;

            //BH Offsets
            offsets[priority_offset + num_blocks * 11 + blk * 3 + 0] = y_off+4*y_stride;
            offsets[priority_offset + num_blocks * 11 + blk * 3 + 1] = u_off+4*uv_stride;
            offsets[priority_offset + num_blocks * 11 + blk * 3 + 2] = v_off+4*uv_stride;
            offsets[priority_offset + num_blocks * 14 + blk]         = y_off+8*y_stride;
            offsets[priority_offset + num_blocks * 15 + blk]         = y_off+12*y_stride;
        }
        priority_offset = priority_offset + num_blocks*16;
    } else {
        //Simple filter

        //Map the offsets buffer
        VP8_CL_MAP_BUF(mbd->cl_commands, loop_mem.offsets_mem, offsets, sizeof(cl_int)*num_blocks*8,,)
        //populate it with the correct offsets for current filter type
        for (blk = 0; blk < num_blocks; blk++){
            int y_off = y_offsets[blk];

            //MBVS offsets
            offsets[priority_offset + blk] = y_off;

            //BVS Offsets
            offsets[priority_offset + num_blocks + blk]    = y_off+4;
            offsets[priority_offset + num_blocks *2 + blk] = y_off+8;
            offsets[priority_offset + num_blocks *3 + blk] = y_off+12;

            //MBHS Offsets
            offsets[priority_offset + num_blocks * 4 + blk] = y_off;

            //BHS Offsets
            offsets[priority_offset + num_blocks * 5 + blk] = y_off + 4 * y_stride;
            offsets[priority_offset + num_blocks * 6 + blk] = y_off + 8 * y_stride;
            offsets[priority_offset + num_blocks * 7 + blk] = y_off + 12 * y_stride;
        }
        priority_offset = priority_offset + num_blocks*8;
    }
    
    //Unmap the offsets buffer
    VP8_CL_UNMAP_BUF(mbd->cl_commands, loop_mem.offsets_mem, offsets,,);
}

void vp8_loop_filter_macroblocks_cl(int num_blocks, int mb_rows[], int mb_cols[],
    int dc_diffs[], int *y_offsets, int *u_offsets, int *v_offsets, int *filter_levels,
    VP8_COMMON *cm, MACROBLOCKD *mbd,  cl_mem lfi_mem, YV12_BUFFER_CONFIG *post, int priority_level
)
{
    LOOPFILTERTYPE filter_type = cm->filter_type;
    int err;
    VP8_LOOPFILTER_ARGS args;

#if USE_MAPPED_BUFFERS
    cl_int *filters = NULL;
#else
    int filters[num_blocks*4];
#endif
    
    cl_grow_loop_mem(mbd, post, num_blocks, cm);
    
    args.buf_mem = post->buffer_mem;
    args.lfi_mem = lfi_mem;
    args.offsets_mem = loop_mem.offsets_mem;
    args.pitches_mem = loop_mem.pitches_mem;
    args.filters_mem = loop_mem.filters_mem;

#define COLS_LOCATION 1
#define DC_DIFFS_LOCATION 2
#define ROWS_LOCATION 3
    
#if USE_MAPPED_BUFFERS
    VP8_CL_MAP_BUF(mbd->cl_commands, loop_mem.filters_mem, filters, 4*num_blocks*sizeof(int),,);
#endif
    
    vpx_memcpy(filters, filter_levels, num_blocks*sizeof(int));
    vpx_memcpy(&filters[DC_DIFFS_LOCATION*num_blocks], dc_diffs, num_blocks*sizeof(int));
    vpx_memcpy(&filters[COLS_LOCATION*num_blocks], mb_cols, num_blocks*sizeof(int));
    vpx_memcpy(&filters[ROWS_LOCATION*num_blocks], mb_rows, num_blocks*sizeof(int));

#if USE_MAPPED_BUFFERS
    VP8_CL_UNMAP_BUF(mbd->cl_commands, loop_mem.filters_mem, filters, ,)
#else    
    //Set the filters_mem buffer with the filter_levels, rows, cols, and dc_diffs
    VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.filters_mem, sizeof(cl_int)*num_blocks*4, filters,,)
#endif

    //Fill loop_mem.offsets_mem will all offsets for this priority level.
    if (recalculate_offsets == 1)
        vp8_loop_filter_build_offsets(mbd, num_blocks, y_offsets, u_offsets, v_offsets, post, cm, filter_type, priority_level);

    args.priority_offset = priority_offsets[priority_level];
    
    if (filter_type == NORMAL_LOOPFILTER){
        vp8_loop_filter_mbv_cl(mbd, &args, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, COLS_LOCATION );
        vp8_loop_filter_bv_cl(mbd, &args, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, DC_DIFFS_LOCATION );
        vp8_loop_filter_mbh_cl(mbd, &args, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, ROWS_LOCATION );
        vp8_loop_filter_bh_cl(mbd, &args, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, DC_DIFFS_LOCATION );
    } else {
        vp8_loop_filter_mbvs_cl(mbd, &args, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, COLS_LOCATION );
        vp8_loop_filter_bvs_cl(mbd, &args, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, DC_DIFFS_LOCATION );
        vp8_loop_filter_mbhs_cl(mbd, &args, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, ROWS_LOCATION );
        vp8_loop_filter_bhs_cl(mbd, &args, num_blocks, y_offsets, u_offsets, v_offsets, post->y_stride, post->uv_stride, DC_DIFFS_LOCATION );
    }

}

void vp8_loop_filter_add_macroblock_cl(VP8_COMMON *cm, int mb_row, int mb_col,
        MACROBLOCKD *mbd, YV12_BUFFER_CONFIG *post, int row[], int col[], int dc_diffs[], int y_offsets[], int u_offsets[], int v_offsets[],
        int filter_levels[], int baseline_filter_level[], int pos)
{
    int y_offset = 16 * (mb_col + (mb_row*cm->mb_cols)) + mb_row * (post->y_stride * 16 - post->y_width);
    int uv_offset = 8 * (mb_col + (mb_row*cm->mb_cols)) + mb_row * (post->uv_stride * 8 - post->uv_width);

    unsigned char *buf_base = post->buffer_alloc;
    y_offsets[pos] = post->y_buffer - buf_base + y_offset;
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

        //Skip non-existent MBs
        if ((mb_col > -1 && (mb_col < mb_cols)) && (mb_row < cm->mb_rows)){
            vp8_loop_filter_add_macroblock_cl(cm, mb_row, mb_col,
                mbd, post, rows, cols, dc_diffs, y_offsets, u_offsets, v_offsets,
                filter_levels, baseline_filter_level, current_pos++
            );
        }
    }
    
    //Process all of the MBs in the current priority
    vp8_loop_filter_macroblocks_cl(current_pos, rows, cols, dc_diffs, y_offsets, u_offsets, v_offsets, filter_levels, cm, mbd, lfi_mem, post, priority);
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
    loop_filter_info *lfi_ptr = NULL;
    unsigned char *buf = NULL;

#if MAP_PITCHES
    cl_int *pitches = NULL;
#else
    int pitches[3] = {post->y_stride, post->uv_stride, post->uv_stride};    
#endif
   
    mbd->mode_info_context = cm->mi; /* Point at base of Mb MODE_INFO list */
    
    /* Note the baseline filter values for each segment */
    vp8_loop_filter_set_baselines_cl(mbd, default_filt_lvl, baseline_filter_level);
    
    /* Initialize the loop filter for this frame. */
    if ((cm->last_filter_type != cm->filter_type) || (cm->last_sharpness_level != cm->sharpness_level))
        vp8_init_loop_filter(cm);
    else if (frame_type != cm->last_frame_type)
        vp8_frame_init_loop_filter(lfi, frame_type);

    cl_grow_loop_mem(mbd, post, 30, cm); //Default to allocating enough for 480p
    priority_offset = 0;
    
#if USE_MAPPED_BUFFERS
    if (lfi_mem == NULL){
        VP8_CL_CREATE_MAPPED_BUF(mbd->cl_commands, lfi_mem, lfi_ptr, sizeof(loop_filter_info)*(MAX_LOOP_FILTER+1), , );
    } else {
        //map the buffer
        VP8_CL_MAP_BUF(mbd->cl_commands, lfi_mem, lfi_ptr, sizeof(loop_filter_info)*(MAX_LOOP_FILTER+1),,);
    }
    vpx_memcpy(lfi_ptr, cm->lf_info, sizeof(loop_filter_info)*(MAX_LOOP_FILTER+1));
    VP8_CL_UNMAP_BUF(mbd->cl_commands, lfi_mem, lfi_ptr,,)
#else
     if (lfi_mem == NULL){
        VP8_CL_CREATE_BUF(mbd->cl_commands, lfi_mem, , sizeof(loop_filter_info)*(MAX_LOOP_FILTER+1), cm->lf_info,, );
     } else {
        VP8_CL_SET_BUF(mbd->cl_commands, lfi_mem, sizeof(loop_filter_info)*(MAX_LOOP_FILTER+1), cm->lf_info,,);
     }
#endif

#if USE_MAPPED_BUFFERS
    VP8_CL_MAP_BUF(mbd->cl_commands, post->buffer_mem, buf, post->buffer_size, vp8_loop_filter_frame(cm,mbd,default_filt_lvl),);
    vpx_memcpy(buf, post->buffer_alloc, post->buffer_size);
    VP8_CL_UNMAP_BUF(mbd->cl_commands, post->buffer_mem, buf,,);
#else
    VP8_CL_SET_BUF(mbd->cl_commands, post->buffer_mem, post->buffer_size, post->buffer_alloc,
            vp8_loop_filter_frame(cm,mbd,default_filt_lvl),);
#endif

    //Determine if offsets need to be recalculated
    recalculate_offsets = 0;
    if (frame_num++ == 0)
        recalculate_offsets = 1;
    else if (post->y_stride != prior_y_stride)
        recalculate_offsets = 1;
    else if (post->uv_stride != prior_uv_stride)
        recalculate_offsets = 1;
    else if (prior_filter_type != cm->filter_type)
        recalculate_offsets = 1;
    else if (prior_mbrows != cm->mb_rows)
        recalculate_offsets = 1;
    else if (prior_mbcols != cm->mb_cols)
        recalculate_offsets = 1;
    
    if (recalculate_offsets == 1){
        #if MAP_PITCHES
            VP8_CL_MAP_BUF(mbd->cl_commands, loop_mem.pitches_mem, pitches, sizeof(cl_int)*3,,)
            pitches[0] = post->y_stride;
            pitches[1] = post->uv_stride;
            pitches[2] = post->uv_stride;
            VP8_CL_UNMAP_BUF(mbd->cl_commands, loop_mem.pitches_mem, pitches,,);
        #else
            VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.pitches_mem, sizeof(cl_int)*3, pitches, vp8_loop_filter_frame(cm,mbd,default_filt_lvl),);
        #endif
        
        if (priority_offsets != NULL)
            free(priority_offsets);

        priority_offsets = malloc(sizeof(cl_int)*(2 * (cm->mb_rows - 1) + cm->mb_cols));
        if (priority_offsets == NULL){
            vp8_loop_filter_frame(cm,mbd,default_filt_lvl);
            return;
        }

        prior_y_stride = post->y_stride;        
        prior_uv_stride = post->uv_stride;
        prior_filter_type = cm->filter_type;
        prior_mbrows = cm->mb_rows;
        prior_mbcols = cm->mb_cols;
    }
    
    //Maximum priority = 2*(Height-1) + Width in Macroblocks
    for (priority = 0; priority < 2 * (cm->mb_rows - 1) + cm->mb_cols ; priority++){
        vp8_loop_filter_priority_cl(priority, cm, mbd, lfi_mem, baseline_filter_level, post);
    }

    //Retrieve buffer contents
#if USE_MAPPED_BUFFERS
    buf = clEnqueueMapBuffer(mbd->cl_commands, post->buffer_mem, CL_TRUE, CL_MAP_READ, 0, post->buffer_size, 0, NULL, NULL, &err); \
    vpx_memcpy(post->buffer_alloc, buf, post->buffer_size);
    VP8_CL_UNMAP_BUF(mbd->cl_commands, post->buffer_mem, buf,,);
#else
    err = clEnqueueReadBuffer(mbd->cl_commands, post->buffer_mem, CL_FALSE, 0, post->buffer_size, post->buffer_alloc, 0, NULL, NULL);
#endif
    
    VP8_CL_CHECK_SUCCESS(mbd->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read loop filter output!\n",
        ,
    );

    VP8_CL_FINISH(mbd->cl_commands);
}
