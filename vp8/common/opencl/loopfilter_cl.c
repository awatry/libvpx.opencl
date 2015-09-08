/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdio.h>
#include "loopfilter_cl.h"
#include "../onyxc_int.h"

#include "vpx_config.h"
#include "vpx_mem/vpx_mem.h"
#include "vp8_opencl.h"
#include "blockd_cl.h"

//Disable usage of mapped buffers for performance increase on Nvidia hardware
#if ARCH_ARM
#define USE_MAPPED_BUFFERS 0
#define MAP_FILTERS 0
#define MAP_OFFSETS 0
#else
#define USE_MAPPED_BUFFERS 1
#define MAP_FILTERS 1
#define MAP_OFFSETS 1
#endif

#define SKIP_NON_FILTERED_MBS 0

#define STR(x) STRINGIFY(x)
#define STRINGIFY(x) #x

#define VP8_SIMD_STRING " -D SIMD_WIDTH=" STR(SIMD_WIDTH)
#define VP8_LF_COMBINE_PLANES_STR " -D COMBINE_PLANES="

#define VP8_LF_UINT_BUFFER_STR " -D MEM_IS_UINT="

#define COLS_LOCATION 1
#define DC_DIFFS_LOCATION 2
#define ROWS_LOCATION 3
const char *loopFilterCompileOptions = "-D COLS_LOCATION=1 -D DC_DIFFS_LOCATION=2 -D ROWS_LOCATION=3 -D MAX_LOOP_FILTER=63" VP8_SIMD_STRING;
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

    memset(&loop_mem, 0, sizeof(struct VP8_LOOP_MEM));

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

    //XXX: This could be modified to allocate one buffer and then clCreateSubBuffer
    //Now re-allocate the memory in the right size
    loop_mem.offsets_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_ONLY|VP8_CL_MEM_ALLOC_TYPE, sizeof(cl_int)*cm->MBs*3, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    //XXX: This probably only needs to be allocated once... if we didn't free all loop mem first
    loop_mem.pitches_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_ONLY|VP8_CL_MEM_ALLOC_TYPE, sizeof(cl_int)*3, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    loop_mem.filters_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_ONLY|VP8_CL_MEM_ALLOC_TYPE, sizeof(cl_int)*cm->MBs*4, NULL, &err);
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
    loop_mem.block_offsets_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_ONLY|VP8_CL_MEM_ALLOC_TYPE, sizeof(cl_int) * priority_levels, NULL, &err);
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
    loop_mem.priority_num_blocks_mem = clCreateBuffer(cl_data.context, CL_MEM_READ_ONLY|VP8_CL_MEM_ALLOC_TYPE, sizeof(cl_int) * priority_levels, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error creating loop filter buffer\n");
        return err;
    }
    
    loop_mem.num_blocks = num_blocks;

    return cl_populate_loop_mem(mbd, post);
}

static char* vp8_cl_build_lf_compile_opts(){
    //Build program compile-time options
    //If CPU (or forced), use combined planes
    size_t lf_co_size = strlen(loopFilterCompileOptions)+1;
    char *type_str;
    char *lf_opts = malloc(lf_co_size + strlen(VP8_LF_COMBINE_PLANES_STR) + 1 + strlen(VP8_LF_UINT_BUFFER_STR)+1);
    if (lf_opts == NULL){
        return NULL;
    }

    type_str = malloc(2);
    if (type_str == NULL){
        free(lf_opts);
        return NULL;
    }
    cl_data.vp8_loop_filter_combine_planes = (cl_data.device_type == CL_DEVICE_TYPE_CPU);
    sprintf(type_str,"%d", cl_data.vp8_loop_filter_combine_planes);
    
    lf_opts = strcpy(lf_opts, loopFilterCompileOptions);
    lf_opts = strcat(lf_opts, VP8_LF_COMBINE_PLANES_STR);
    lf_opts = strcat(lf_opts, type_str);
    
    cl_data.vp8_loop_filter_uint_buffer = (cl_data.device_type != CL_DEVICE_TYPE_CPU);
    sprintf(type_str,"%d", cl_data.vp8_loop_filter_uint_buffer);
    lf_opts = strcat(lf_opts, VP8_LF_UINT_BUFFER_STR);
    lf_opts = strcat(lf_opts, type_str);
    
    free(type_str);
    return lf_opts;
}

//Start of externally callable functions.

int cl_init_loop_filter() {
    int err;

    char *lf_opts = vp8_cl_build_lf_compile_opts();
    if (lf_opts == NULL)
        return VP8_CL_TRIED_BUT_FAILED;
    
    // Create the filter compute program from the file-defined source code
    if ( cl_load_program(&cl_data.loop_filter_program, loop_filter_cl_file_name,
            lf_opts) != CL_SUCCESS )
        return VP8_CL_TRIED_BUT_FAILED;
    
    free(lf_opts);

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
    
    memset(&loop_mem, 0, sizeof(struct VP8_LOOP_MEM));
    block_offsets = NULL;
    priority_num_blocks = NULL;

    vp8_loop_filter_filters_init();

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

    if (num_blocks == 0)
        return;

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
    
    /*
    if (num_levels > 1){
        int max = 0;
        int level;
        for (level = priority_level; level < priority_level+num_levels; level++){
            if (priority_num_blocks[level] > max)
                max = priority_num_blocks[level];
        }
        num_blocks = max;
    }
    */
#if SKIP_NON_FILTERED_MBS
    if (num_blocks == 0){
        return;
    }
#endif
    
    if (filter_type == NORMAL_LOOPFILTER){
        vp8_loop_filter_all_edges_cl(mbd, args, 3, num_blocks);
    } else {
        vp8_loop_filter_simple_all_edges_cl(mbd, args, 1, num_blocks);
    }

}

void vp8_loop_filter_add_macroblock_cl(VP8_COMMON *cm, int mb_row, int mb_col,
        MACROBLOCKD *mbd, YV12_BUFFER_CONFIG *post, cl_int row[], cl_int col[], cl_int dc_diffs[], int y_offsets[], int u_offsets[], int v_offsets[],
        cl_int filter_levels[], loop_filter_info_n *lfi_n, int pos, int filter_level)
{
    int y_offset = 16 * (mb_col + (mb_row*cm->mb_cols)) + mb_row * (post->y_stride * 16 - post->y_width);
    int uv_offset = 8 * (mb_col + (mb_row*cm->mb_cols)) + mb_row * (post->uv_stride * 8 - post->uv_width);

    unsigned char *buf_base = post->buffer_alloc;
    y_offsets[pos] = post->y_buffer - buf_base + y_offset;
    u_offsets[pos] = post->u_buffer - buf_base + uv_offset;
    v_offsets[pos] = post->v_buffer - buf_base + uv_offset;

    /* Distance of Mb to the various image edges.
     * These specified to 8th pel as they are always compared to values that are in 1/8th pel units
     * Apply any context driven MB level adjustment
     */
    
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

		/*
		 * So here's the idea for how to do 1-column offset filtering:
		 * 1) If you're in col 0, or the previous macroblock was NOT filtered, filter the MBV edge
		 * 2) Block Vertical edges as normal
		 * 3) MBH Edge as normal
		 * 4) BH edges as normal
		 * 5) If the MB to the right exists and will be filtered, do its MBV edge during the previous MB
		 * 
		 * By doing this, it might be possible to lag each row by one column instead of 2.
		 * Worth a shot at least, given that there's a 30+% possible performance gain on the table.
		 * 
		 * Credit to Gaute Strokkenes on the Codec Developers group
		*/
		
        //Skip non-existent MBs
        if ((mb_col > -1 && (mb_col < mb_cols)) && (mb_row < cm->mb_rows)){
            
            int mode_index, seg, ref_frame, filter_level;
            mbd->mode_info_context = cm->mi + ((mb_row * (cm->mb_cols+1) + mb_col));
            mode_index = cm->lf_info.mode_lf_lut[mbd->mode_info_context->mbmi.mode];
            seg = mbd->mode_info_context->mbmi.segment_id;
            ref_frame = mbd->mode_info_context->mbmi.ref_frame;
            filter_level = cm->lf_info.lvl[seg][ref_frame][mode_index];
            
#if SKIP_NON_FILTERED_MBS
            if (filter_level <= 0 || (mb_row == 0 && mb_col == 0 && (mbd->mode_info_context->mbmi.mode != B_PRED &&
                            mbd->mode_info_context->mbmi.mode != SPLITMV &&
                            mbd->mode_info_context->mbmi.mb_skip_coeff)))
                continue;
#endif
            
            vp8_loop_filter_add_macroblock_cl(cm, mb_row, mb_col,
                mbd, post, rows, cols, dc_diffs, y_offsets, u_offsets, v_offsets,
                filter_levels, &cm->lf_info, *current_blocks, filter_level
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

    int num_blocks = priority_num_blocks[levels-1] + block_offsets[levels-1];
    
#if MAP_FILTERS
    //Always copy the dc_diffs, rows, cols, and filter_offsets values
    VP8_CL_MAP_BUF(mbd->cl_commands, loop_mem.filters_mem, filters, 4*num_blocks*sizeof(cl_int),,);
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
    VP8_CL_SET_BUF(mbd->cl_commands, loop_mem.filters_mem, 4*num_blocks*sizeof(cl_int), filters, vp8_loop_filter_frame(cm,mbd),)
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
#endif
    cl_uint *buf = NULL;

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
    int i;
    
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

#if USE_MAPPED_BUFFERS || 1
    VP8_CL_MAP_BUF(mbd->cl_commands, post->buffer_mem, buf, post->frame_size * sizeof(cl_uint), vp8_loop_filter_frame(cm,mbd),);
    //Copy frame to GPU and convert from uchar to uint
    if (cl_data.vp8_loop_filter_uint_buffer){
        for (i = 0; i < post->frame_size; i++){
            buf[i] = (cl_uint)post->buffer_alloc[i];
        }
    } else {
        vpx_memcpy(buf, post->buffer_alloc, post->frame_size);
    }
    
    VP8_CL_UNMAP_BUF(mbd->cl_commands, post->buffer_mem, buf,,);
#else
    VP8_CL_SET_BUF(mbd->cl_commands, post->buffer_mem, post->frame_size, post->buffer_alloc,
            vp8_loop_filter_frame(cm,mbd),);
#endif

#if SKIP_NON_FILTERED_MBS
    recalculate_offsets = 1;
#else
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
#endif

    
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
        vp8_loop_filter_macroblocks_cl(cm,  mbd, priority, 1, &args);
    }

    //Retrieve buffer contents
#if 1 || USE_MAPPED_BUFFERS && (!defined(CL_MEM_USE_PERSISTENT_MEM_AMD) || (CL_MEM_USE_PERSISTENT_MEM_AMD != VP8_CL_MEM_ALLOC_TYPE))
    buf = clEnqueueMapBuffer(mbd->cl_commands, post->buffer_mem, CL_TRUE, CL_MAP_READ, 0, post->frame_size * sizeof(cl_uint), 0, NULL, NULL, &err); \
    if (cl_data.vp8_loop_filter_uint_buffer){
        for (i = 0; i < post->frame_size; i++){
            post->buffer_alloc[i] = (unsigned char)buf[i];
        }
    } else {
        vpx_memcpy(post->buffer_alloc, buf, post->frame_size);
    }
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
