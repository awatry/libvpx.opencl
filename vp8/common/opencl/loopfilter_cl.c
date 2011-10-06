/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "loopfilter_cl.h"
#include "../onyxc_int.h"

#include "vpx_config.h"
#include "vpx_mem/vpx_mem.h"
#include "vp8_opencl.h"
#include "blockd_cl.h"

//Disable usage of mapped buffers for performance increase on Nvidia hardware
#if 0 
#define USE_MAPPED_BUFFERS 0
#define MAP_FILTERS 0
#define MAP_OFFSETS 0
#else
#define USE_MAPPED_BUFFERS 1
#define MAP_FILTERS 1
#define MAP_OFFSETS 1
#endif

#define STRINGIFY(x) #x
#if SIMD_WIDTH == 1
#define SIMD_STRING "1"
#else
#define SIMD_STRING "16"
#endif

#define COLS_LOCATION 1
#define DC_DIFFS_LOCATION 2
#define ROWS_LOCATION 3
#if VP8_LOOP_FILTER_MULTI_LEVEL
const char *loopFilterCompileOptions = "-D COLS_LOCATION=1 -D DC_DIFFS_LOCATION=2 -D ROWS_LOCATION=3 -D VP8_LOOP_FILTER_MULTI_LEVEL=1 -D MAX_LOOP_FILTER=63 -D SIMD_WIDTH="SIMD_STRING;
#else
const char *loopFilterCompileOptions = "-D COLS_LOCATION=1 -D DC_DIFFS_LOCATION=2 -D ROWS_LOCATION=3 -D VP8_LOOP_FILTER_MULTI_LEVEL=0 -D MAX_LOOP_FILTER=63 -D SIMD_WIDTH="SIMD_STRING;
#endif
const char *loop_filter_cl_file_name = "vp8/common/opencl/loopfilter";

typedef struct VP8_LOOP_SETTINGS{
    int y_stride;
    int uv_stride;
    LOOPFILTERTYPE filter_type;
    int mbrows;
    int mbcols;
} VP8_LOOP_SETTINGS;

static VP8_LOOP_SETTINGS prior_settings;

static int frame_num = 0;
static cl_int *block_offsets = NULL;
static cl_int *priority_num_blocks = NULL;
static int recalculate_offsets = 1;
static int max_blocks = 0;

typedef unsigned char uc;

extern void vp8_loop_filter_frame
(
    VP8_COMMON *cm,
    MACROBLOCKD *mbd
);

typedef struct VP8_LOOP_MEM{
    cl_int num_blocks;
    cl_mem offsets_mem;
    cl_mem pitches_mem;
    cl_mem filters_mem;
    
    cl_mem block_offsets_mem;
    cl_mem priority_num_blocks_mem;
} VP8_LOOP_MEM;

VP8_LOOP_MEM loop_mem;
cl_mem lfi_mem = NULL;

prototype_loopfilter_cl(vp8_loop_filter_all_edges_cl);
prototype_loopfilter_cl(vp8_loop_filter_simple_all_edges_cl);

int cl_free_loop_mem(){
    int err = 0;

    if (block_offsets != NULL) free(block_offsets);
    if (priority_num_blocks != NULL) free(priority_num_blocks);
    block_offsets = NULL;
    priority_num_blocks = NULL;
    
    if (loop_mem.offsets_mem != NULL) err |= clReleaseMemObject(loop_mem.offsets_mem);
    if (loop_mem.pitches_mem != NULL) err |= clReleaseMemObject(loop_mem.pitches_mem);
    if (loop_mem.filters_mem != NULL) err |= clReleaseMemObject(loop_mem.filters_mem);
    if (loop_mem.block_offsets_mem != NULL) err |= clReleaseMemObject(loop_mem.block_offsets_mem);
    if (loop_mem.priority_num_blocks_mem != NULL) err |= clReleaseMemObject(loop_mem.priority_num_blocks_mem);
    loop_mem.offsets_mem = NULL;
    loop_mem.pitches_mem = NULL;
    loop_mem.filters_mem = NULL;
    loop_mem.block_offsets_mem = NULL;
    loop_mem.priority_num_blocks_mem = NULL;

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
    VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.pitches_mem, 3*sizeof(cl_int), pitches,,err);
#endif
    return err;
}

int cl_grow_loop_mem(MACROBLOCKD *mbd, YV12_BUFFER_CONFIG *post, VP8_COMMON *cm){

    int err;

    int num_blocks = cm->MBs;
    int priority_levels = 2*(cm->mb_rows - 1) + cm->mb_cols;
    
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
    loop_mem.filters_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, sizeof(cl_int)*cm->MBs*4, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }

    //Number of blocks that have already been processed at the beginning of a
    //given priority level.
    block_offsets = malloc( sizeof(cl_int) * priority_levels );
    if (block_offsets == NULL){
        cl_destroy(mbd->cl_commands, VP8_CL_TRIED_BUT_FAILED);
        return VP8_CL_TRIED_BUT_FAILED;
    }
    loop_mem.block_offsets_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, sizeof(cl_int) * priority_levels, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }

    //Number of blocks to be processed in a given priority level.
    priority_num_blocks = malloc( sizeof(cl_int) * priority_levels );
    if (priority_num_blocks == NULL){
        cl_destroy(mbd->cl_commands, VP8_CL_TRIED_BUT_FAILED);
        return VP8_CL_TRIED_BUT_FAILED;
    }
    loop_mem.priority_num_blocks_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, sizeof(cl_int) * priority_levels, NULL, &err);
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
    VP8_CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_loop_filter_all_edges_kernel,"vp8_loop_filter_all_edges_kernel");
    VP8_CL_CALC_LOCAL_SIZE(cl_data.vp8_loop_filter_all_edges_kernel,&cl_data.vp8_loop_filter_all_edges_kernel_size);
    
    if (cl_data.vp8_loop_filter_all_edges_kernel_size < 16){
        VP8_CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_loop_filter_horizontal_edges_kernel,"vp8_loop_filter_horizontal_edges_kernel");
        VP8_CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_loop_filter_vertical_edges_kernel,"vp8_loop_filter_vertical_edges_kernel");
        VP8_CL_CALC_LOCAL_SIZE(cl_data.vp8_loop_filter_horizontal_edges_kernel,&cl_data.vp8_loop_filter_horizontal_edges_kernel_size);
        VP8_CL_CALC_LOCAL_SIZE(cl_data.vp8_loop_filter_vertical_edges_kernel,&cl_data.vp8_loop_filter_vertical_edges_kernel_size);
    } else {
        cl_data.vp8_loop_filter_horizontal_edges_kernel = NULL;
        cl_data.vp8_loop_filter_vertical_edges_kernel = NULL;
    }

    VP8_CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_loop_filter_simple_all_edges_kernel,"vp8_loop_filter_simple_all_edges_kernel");
    VP8_CL_CALC_LOCAL_SIZE(cl_data.vp8_loop_filter_simple_all_edges_kernel,&cl_data.vp8_loop_filter_simple_all_edges_kernel_size);
    
    if (cl_data.vp8_loop_filter_simple_all_edges_kernel_size < 16){
        VP8_CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_loop_filter_simple_horizontal_edges_kernel,"vp8_loop_filter_simple_horizontal_edges_kernel");
        VP8_CL_CREATE_KERNEL(cl_data,loop_filter_program,vp8_loop_filter_simple_vertical_edges_kernel,"vp8_loop_filter_simple_vertical_edges_kernel");
        VP8_CL_CALC_LOCAL_SIZE(cl_data.vp8_loop_filter_simple_horizontal_edges_kernel,&cl_data.vp8_loop_filter_simple_horizontal_edges_kernel_size);
        VP8_CL_CALC_LOCAL_SIZE(cl_data.vp8_loop_filter_simple_vertical_edges_kernel,&cl_data.vp8_loop_filter_simple_vertical_edges_kernel_size);
    } else {
        cl_data.vp8_loop_filter_simple_horizontal_edges_kernel = NULL;
        cl_data.vp8_loop_filter_simple_vertical_edges_kernel = NULL;
    }
    
    loop_mem.num_blocks = 0;
    loop_mem.offsets_mem = NULL;
    loop_mem.pitches_mem = NULL;
    loop_mem.filters_mem = NULL;
    loop_mem.block_offsets_mem = NULL;
    loop_mem.priority_num_blocks_mem = NULL;
    block_offsets = NULL;
    priority_num_blocks = NULL;
    
    return CL_SUCCESS;
}

void cl_destroy_loop_filter(){

    cl_free_loop_mem();

    if (lfi_mem != NULL){
        clReleaseMemObject(lfi_mem);
        lfi_mem = NULL;
    }
   
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_loop_filter_all_edges_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_loop_filter_horizontal_edges_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_loop_filter_vertical_edges_kernel);

    VP8_CL_RELEASE_KERNEL(cl_data.vp8_loop_filter_simple_all_edges_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_loop_filter_simple_horizontal_edges_kernel);
    VP8_CL_RELEASE_KERNEL(cl_data.vp8_loop_filter_simple_vertical_edges_kernel);

    if (cl_data.loop_filter_program)
        clReleaseProgram(cl_data.loop_filter_program);
   
    cl_data.loop_filter_program = NULL;
}


/* Generate the list of filtering values per priority level*/
void vp8_loop_filter_build_filter_offsets(cl_int *filters, int level, 
        cl_int *filter_levels, cl_int *dc_diffs, cl_int *mb_rows, cl_int *mb_cols
)
{
    int offset = block_offsets[level]*4;
    int num_blocks = priority_num_blocks[level];

    vpx_memcpy(&filters[offset], filter_levels, num_blocks*sizeof(cl_int));
    vpx_memcpy(&filters[offset+DC_DIFFS_LOCATION*num_blocks], dc_diffs, num_blocks*sizeof(cl_int));
    vpx_memcpy(&filters[offset+COLS_LOCATION*num_blocks], mb_cols, num_blocks*sizeof(cl_int));
    vpx_memcpy(&filters[offset+ROWS_LOCATION*num_blocks], mb_rows, num_blocks*sizeof(cl_int));
    
}

void vp8_loop_filter_build_offsets(MACROBLOCKD *mbd, int num_blocks, 
        cl_int *y_offsets, cl_int *u_offsets, cl_int *v_offsets, YV12_BUFFER_CONFIG *post, 
        VP8_COMMON *cm, LOOPFILTERTYPE filter_type, int priority_level, cl_int *offsets
)
{
    int blk;
    
    if (filter_type == NORMAL_LOOPFILTER){
                
        offsets += block_offsets[priority_level]*3;
                
        //populate it with the correct offsets for current filter type
        for (blk = 0; blk < num_blocks; blk++){
            //MBV/MBH offsets
            offsets[blk * 3 + 0] = y_offsets[blk];
            offsets[blk * 3 + 1] = u_offsets[blk];
            offsets[blk * 3 + 2] = v_offsets[blk];
        }
    } else {
        //Simple filter
        offsets += block_offsets[priority_level];
        
        //populate it with the correct offsets for current filter type
        for (blk = 0; blk < num_blocks; blk++){
            //MB[HV]S offsets
            offsets[blk] = y_offsets[blk];
        }
    }
}

/* Filter all Macroblocks in a given priority level */
void vp8_loop_filter_macroblocks_cl(
        VP8_COMMON *cm, MACROBLOCKD *mbd, int priority_level, int num_levels, VP8_LOOPFILTER_ARGS *args
)
{
    LOOPFILTERTYPE filter_type = cm->filter_type;
    int num_blocks = priority_num_blocks[priority_level];
    
    args->priority_level = priority_level;
    args->num_levels = num_levels;
    
    if (num_levels > 1){
        int max = 0;
        int level;
        for (level = priority_level; level < priority_level+num_levels; level++){
            if (priority_num_blocks[level] > max)
                max = priority_num_blocks[level];
        }
        num_blocks = max;
    }
    
    if (filter_type == NORMAL_LOOPFILTER){
        vp8_loop_filter_all_edges_cl(mbd, args, 3, num_blocks);
    } else {
        vp8_loop_filter_simple_all_edges_cl(mbd, args, 1, num_blocks);
    }

}

void vp8_loop_filter_add_macroblock_cl(VP8_COMMON *cm, int mb_row, int mb_col,
        MACROBLOCKD *mbd, YV12_BUFFER_CONFIG *post, cl_int row[], cl_int col[], cl_int dc_diffs[], int y_offsets[], int u_offsets[], int v_offsets[],
        cl_int filter_levels[], loop_filter_info_n *lfi_n, int pos)
{
    int y_offset = 16 * (mb_col + (mb_row*cm->mb_cols)) + mb_row * (post->y_stride * 16 - post->y_width);
    int uv_offset = 8 * (mb_col + (mb_row*cm->mb_cols)) + mb_row * (post->uv_stride * 8 - post->uv_width);

    int mode_index, seg, ref_frame, filter_level;
    
    unsigned char *buf_base = post->buffer_alloc;
    y_offsets[pos] = post->y_buffer - buf_base + y_offset;
    u_offsets[pos] = post->u_buffer - buf_base + uv_offset;
    v_offsets[pos] = post->v_buffer - buf_base + uv_offset;

    mbd->mode_info_context = cm->mi + ((mb_row * (cm->mb_cols+1) + mb_col));

    /* Distance of Mb to the various image edges.
     * These specified to 8th pel as they are always compared to values that are in 1/8th pel units
     * Apply any context driven MB level adjustment
     */
    
    mode_index = lfi_n->mode_lf_lut[mbd->mode_info_context->mbmi.mode];
    seg = mbd->mode_info_context->mbmi.segment_id;
    ref_frame = mbd->mode_info_context->mbmi.ref_frame;
    filter_level = lfi_n->lvl[seg][ref_frame][mode_index];
    
    filter_levels[pos] = filter_level;
    row[pos] = mb_row;
    col[pos] = mb_col;
    dc_diffs[pos] = ! (mbd->mode_info_context->mbmi.mode != B_PRED &&
                            mbd->mode_info_context->mbmi.mode != SPLITMV &&
                            mbd->mode_info_context->mbmi.mb_skip_coeff);
}

void vp8_loop_filter_build_priority(int priority, VP8_COMMON *cm, MACROBLOCKD *mbd,
        YV12_BUFFER_CONFIG *post, int *current_blocks, 
        cl_int *y_offsets, cl_int *u_offsets, cl_int *v_offsets, 
        cl_int *dc_diffs, cl_int *rows, cl_int *cols, cl_int *filter_levels,
        cl_int *offsets
)
{
    int mb_row, mb_col, mb_cols = cm->mb_cols;
    int priority_mbs = 0;
    int start_block = *current_blocks;
    
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
                filter_levels, &cm->lf_info, *current_blocks
            );
            current_blocks[0]++;
            priority_mbs++;
        }
    }
    
    if (recalculate_offsets == 1){
        //Set the block/num_blocks for the current level
        priority_num_blocks[priority] = priority_mbs;

        if (priority == 0)
            block_offsets[0] = 0;
        else
            block_offsets[priority] = block_offsets[priority-1] + priority_num_blocks[priority-1];

        vp8_loop_filter_build_offsets(mbd, priority_mbs, 
            &y_offsets[start_block], &u_offsets[start_block], &v_offsets[start_block], 
            post, cm, cm->filter_type, priority, offsets
        );
    }
}

void vp8_loop_filter_offsets_copy(VP8_COMMON *cm, MACROBLOCKD *mbd, 
        cl_int *dc_diffs, cl_int *rows, cl_int *cols, cl_int *filter_levels, int levels
){
    int err, level;
    
    cl_int *filters;

#if MAP_FILTERS
    //Always copy the dc_diffs, rows, cols, and filter_offsets values
    VP8_CL_MAP_BUF(mbd->cl_commands, loop_mem.filters_mem, filters, 4*cm->MBs*sizeof(cl_int),,);
#else
    filters = malloc(4*cm->MBs*sizeof(cl_int));
    if (filters == NULL){
        cl_destroy(mbd->cl_commands, VP8_CL_TRIED_BUT_FAILED);
    }
#endif
    
    for (level = 0; level < levels; level++){
        if (level > 0){
            filter_levels = &filter_levels[priority_num_blocks[level-1]];
            rows = &rows[priority_num_blocks[level-1]];
            cols = &cols[priority_num_blocks[level-1]];
            dc_diffs = &dc_diffs[priority_num_blocks[level-1]];
        }
        vp8_loop_filter_build_filter_offsets(filters, level, 
                filter_levels, dc_diffs, rows, cols);
    }
    
#if MAP_FILTERS
    VP8_CL_UNMAP_BUF(mbd->cl_commands, loop_mem.filters_mem, filters, ,)
#else
    VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.filters_mem, 4*cm->MBs*sizeof(cl_int), filters, vp8_loop_filter_frame(cm,mbd),)
    free(filters);
#endif
}

void vp8_loop_filter_frame_cl
(
    VP8_COMMON *cm,
    MACROBLOCKD *mbd
)
{
    YV12_BUFFER_CONFIG *post = cm->frame_to_show;
    VP8_LOOP_SETTINGS current_settings;
    
    int err, priority;
#if USE_MAPPED_BUFFERS
    loop_filter_info *lfi_ptr = NULL;
    unsigned char *buf = NULL;
#endif

    cl_int *offsets = NULL;
    size_t offsets_size;
    cl_int y_offsets[cm->MBs];
    cl_int u_offsets[cm->MBs];
    cl_int v_offsets[cm->MBs];
    cl_int dc_diffs[cm->MBs];
    cl_int rows[cm->MBs];
    cl_int cols[cm->MBs];
    cl_int filter_levels[cm->MBs];
    int num_levels = 2 * (cm->mb_rows - 1) + cm->mb_cols;
    int current_blocks = 0;
    
    VP8_LOOPFILTER_ARGS args;
    
    /* Initialize the loop filter for this frame. */
    vp8_loop_filter_frame_init( cm, mbd, cm->filter_level);

#if USE_MAPPED_BUFFERS
    if (lfi_mem == NULL){
        VP8_CL_CREATE_MAPPED_BUF(mbd->cl_commands, lfi_mem, lfi_ptr, sizeof(loop_filter_info_n), , );
    } else {
        //map the buffer
        VP8_CL_MAP_BUF(mbd->cl_commands, lfi_mem, lfi_ptr, sizeof(loop_filter_info_n),,);
    }
    vpx_memcpy(lfi_ptr, &cm->lf_info, sizeof(loop_filter_info_n));
    VP8_CL_UNMAP_BUF(mbd->cl_commands, lfi_mem, lfi_ptr,,)
#else
     if (lfi_mem == NULL){
        VP8_CL_CREATE_BUF(mbd->cl_commands, lfi_mem, , sizeof(loop_filter_info_n), &cm->lf_info,, );
     } else {
        VP8_CL_SET_BUF(mbd->cl_commands, lfi_mem, sizeof(loop_filter_info_n), &cm->lf_info,,);
     }
#endif

#if USE_MAPPED_BUFFERS
    VP8_CL_MAP_BUF(mbd->cl_commands, post->buffer_mem, buf, post->frame_size, vp8_loop_filter_frame(cm,mbd),);
    vpx_memcpy(buf, post->buffer_alloc, post->frame_size);
    VP8_CL_UNMAP_BUF(mbd->cl_commands, post->buffer_mem, buf,,);
#else
    VP8_CL_SET_BUF(mbd->cl_commands, post->buffer_mem, post->frame_size, post->buffer_alloc,
            vp8_loop_filter_frame(cm,mbd),);
#endif

    current_settings.filter_type = cm->filter_type;
    current_settings.y_stride = post->y_stride;
    current_settings.uv_stride = post->uv_stride;
    current_settings.mbcols = cm->mb_cols;
    current_settings.mbrows = cm->mb_rows;
    
    //Determine if offsets need to be recalculated
    recalculate_offsets = 0;
    if (frame_num++ == 0)
        recalculate_offsets = 1;
    else if (memcmp(&current_settings, &prior_settings, sizeof(VP8_LOOP_SETTINGS))){
        recalculate_offsets = 1;
    }
    
    if (recalculate_offsets == 1){
        if (cm->MBs <= loop_mem.num_blocks)
            cl_populate_loop_mem(mbd, post); //populate pitches_mem
        else
            cl_grow_loop_mem(mbd, post, cm);
        
        //Copy the current frame's settings for later re-use
        memcpy(&prior_settings, &current_settings, sizeof(VP8_LOOP_SETTINGS));
        
        //map offsets_mem
        if (cm->filter_type == NORMAL_LOOPFILTER)
            offsets_size = sizeof(cl_int)*cm->MBs*3;
        else
            offsets_size = sizeof(cl_int)*cm->MBs;

        max_blocks = 0;
            
#if MAP_OFFSETS
        VP8_CL_MAP_BUF(mbd->cl_commands, loop_mem.offsets_mem, offsets, offsets_size,,)
#else
        offsets = malloc(offsets_size);
        if (offsets == NULL){
            cl_destroy(mbd->cl_commands, VP8_CL_TRIED_BUT_FAILED);
            vp8_loop_filter_frame(cm, mbd);
            return;
        }
#endif
    }

    args.buf_mem = post->buffer_mem;
    args.lfi_mem = lfi_mem;
    args.offsets_mem = loop_mem.offsets_mem;
    args.pitches_mem = loop_mem.pitches_mem;
    args.filters_mem = loop_mem.filters_mem;
    args.block_offsets_mem = loop_mem.block_offsets_mem;
    args.priority_num_blocks_mem = loop_mem.priority_num_blocks_mem;
    args.frame_type = cm->frame_type;
    
    //Maximum priority = 2*(Height-1) + Width in Macroblocks
    //First identify all Macroblocks that will be processed and their priority
    //levels for processing
    for (priority = 0; priority < num_levels ; priority++){
        vp8_loop_filter_build_priority(priority, cm, mbd, post, &current_blocks,
                y_offsets, u_offsets, v_offsets, dc_diffs, rows, cols, filter_levels, 
                offsets
        );
        if (max_blocks < priority_num_blocks[priority]){
            max_blocks = priority_num_blocks[priority];
        }
    }
    
    if (recalculate_offsets == 1){
#if MAP_OFFSETS
        VP8_CL_UNMAP_BUF(mbd->cl_commands, loop_mem.offsets_mem, offsets,,);
#else
        VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.offsets_mem, offsets_size, offsets, vp8_loop_filter_frame(cm, mbd), )
        free(offsets);
        offsets = NULL;
#endif
        
        //Now re-send the block_offsets/priority_num_blocks buffers
        VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.priority_num_blocks_mem, sizeof(cl_int)*num_levels, priority_num_blocks, vp8_loop_filter_frame(cm, mbd), )
        VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.block_offsets_mem, sizeof(cl_int)*num_levels, block_offsets, vp8_loop_filter_frame(cm, mbd), )
    }
    
    //Copy any needed buffer contents to the CL device
    vp8_loop_filter_offsets_copy(cm, mbd, dc_diffs, rows, cols, filter_levels, num_levels);
    
    //Actually process the various priority levels
    for (priority = 0; priority < num_levels ; priority++){
#if VP8_LOOP_FILTER_MULTI_LEVEL
        int end_level = priority;
        if (priority_num_blocks[priority]*48 < cl_data.vp8_loop_filter_all_edges_kernel_size){
            while(++end_level < num_levels){
                if (priority_num_blocks[end_level]*48 > cl_data.vp8_loop_filter_all_edges_kernel_size){
                    break;
                }
            }
            end_level--;
        }

        vp8_loop_filter_macroblocks_cl(cm, mbd, priority, (end_level-priority+1), &args);
        priority = end_level;
#else
        vp8_loop_filter_macroblocks_cl(cm,  mbd, priority, 1, &args);
#endif
    }

    //Retrieve buffer contents
#if USE_MAPPED_BUFFERS
    buf = clEnqueueMapBuffer(mbd->cl_commands, post->buffer_mem, CL_TRUE, CL_MAP_READ, 0, post->frame_size, 0, NULL, NULL, &err); \
    vpx_memcpy(post->buffer_alloc, buf, post->frame_size);
    VP8_CL_UNMAP_BUF(mbd->cl_commands, post->buffer_mem, buf,,);
#else
    err = clEnqueueReadBuffer(mbd->cl_commands, post->buffer_mem, CL_FALSE, 0, post->frame_size, post->buffer_alloc, 0, NULL, NULL);
#endif
    
    VP8_CL_CHECK_SUCCESS(mbd->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read loop filter output!\n",
        ,
    );

    VP8_CL_FINISH(mbd->cl_commands);
}
