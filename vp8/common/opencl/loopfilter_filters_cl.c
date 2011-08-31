/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "vpx_ports/config.h"
#include "vp8_opencl.h"
#include "blockd_cl.h"
#include "loopfilter_cl.h"

typedef unsigned char uc;

static int first_run = 1;
#define NUM_KERNELS 5
static VP8_LOOPFILTER_ARGS filter_args[NUM_KERNELS];

#define VP8_CL_SET_LOOP_ARG(kernel, current, newargs, argnum, type, name) \
    if (current->name != newargs->name){ \
        err |= clSetKernelArg(kernel, argnum, sizeof (type), &newargs->name); \
        current->name = newargs->name; \
    }\


void vp8_loop_filter_horizontal_edges_cl( MACROBLOCKD *x, 
        VP8_LOOPFILTER_ARGS *args, int num_planes, int num_blocks
);

void vp8_loop_filter_vertical_edges_cl( MACROBLOCKD *x, 
        VP8_LOOPFILTER_ARGS *args, int num_planes, int num_blocks
);


static int vp8_loop_filter_cl_run(
    cl_command_queue cq,
    cl_kernel kernel,
    size_t max_local_size,
    VP8_LOOPFILTER_ARGS *args,
    int num_planes,
    int num_blocks,
    VP8_LOOPFILTER_ARGS *current_args
){

    size_t global[3] = {16, num_planes, num_blocks};
    size_t local[3] = {16, num_planes, 1};
    int err;

    if ((max_local_size < 16 * num_planes )){
        local[1] = 1; //Drop down to 1 plane
        if (max_local_size < 16){
            local[0] = 1; //Finally drop to 1 thread per group if necessary.
                          //At this point it'd be better to probably disable CL
        }
    }
    
    if (first_run){
        memset(filter_args, -1, sizeof(VP8_LOOPFILTER_ARGS)*NUM_KERNELS);
        first_run = 0;
    }
    
    err = 0;
    VP8_CL_SET_LOOP_ARG(kernel, current_args, args, 0, cl_mem, buf_mem)
    VP8_CL_SET_LOOP_ARG(kernel, current_args, args, 1, cl_mem, offsets_mem)
    VP8_CL_SET_LOOP_ARG(kernel, current_args, args, 2, cl_mem, pitches_mem)
    VP8_CL_SET_LOOP_ARG(kernel, current_args, args, 3, cl_mem, lfi_mem)
    VP8_CL_SET_LOOP_ARG(kernel, current_args, args, 4, cl_mem, filters_mem)
    VP8_CL_SET_LOOP_ARG(kernel, current_args, args, 5, cl_int, use_mbflim)
    VP8_CL_SET_LOOP_ARG(kernel, current_args, args, 6, cl_int, filter_type)
    VP8_CL_SET_LOOP_ARG(kernel, current_args, args, 7, cl_int, cur_iter)
    VP8_CL_SET_LOOP_ARG(kernel, current_args, args, 8, cl_int, priority_offset)
    VP8_CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",,err
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(cq, kernel, 3, NULL, global, local , 0, NULL, NULL);
    
    VP8_CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);,err
    );
        
    return CL_SUCCESS;
}

//Filters both Macroblock and Block horizontal/vertical edges
void vp8_loop_filter_all_edges_cl
(
    MACROBLOCKD *x,
    VP8_LOOPFILTER_ARGS *args,
    int num_planes,
    int num_blocks
)
{
    
    size_t local = cl_data.vp8_loop_filter_all_edges_kernel_size;
    if (local < 16){
        //Handle Vertical and Horizontal edges in 2 passes.
        vp8_loop_filter_vertical_edges_cl(x, args, num_planes, num_blocks);
        vp8_loop_filter_horizontal_edges_cl(x, args, num_planes, num_blocks);
        return;
    }
    
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_all_edges_kernel, 
        local, args, num_planes, num_blocks, &filter_args[0]
    );
}


//Filters both Macroblock and Block horizontal edges
void vp8_loop_filter_horizontal_edges_cl
(
    MACROBLOCKD *x,
    VP8_LOOPFILTER_ARGS *args,
    int num_planes,
    int num_blocks
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_horizontal_edges_kernel, 
        cl_data.vp8_loop_filter_horizontal_edges_kernel_size, 
        args, num_planes, num_blocks, &filter_args[1]
    );
}

//Filters both Macroblock and Block edges
void vp8_loop_filter_vertical_edges_cl
(
    MACROBLOCKD *x,
    VP8_LOOPFILTER_ARGS *args,
    int num_planes,
    int num_blocks
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_vertical_edges_kernel, 
        cl_data.vp8_loop_filter_vertical_edges_kernel_size, 
        args, num_planes, num_blocks, &filter_args[2]
    );
}

void vp8_loop_filter_simple_horizontal_edge_cl
(
    MACROBLOCKD *x,
    VP8_LOOPFILTER_ARGS *args,
    int num_planes,
    int num_blocks
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_simple_horizontal_edge_kernel,
        cl_data.vp8_loop_filter_simple_horizontal_edge_kernel_size,
        args, num_planes, num_blocks, &filter_args[3]
    );
}

void vp8_loop_filter_simple_vertical_edge_cl
(
    MACROBLOCKD *x,
    VP8_LOOPFILTER_ARGS *args,
    int num_planes,
    int num_blocks
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_simple_vertical_edge_kernel,
        cl_data.vp8_loop_filter_simple_vertical_edge_kernel_size,
        args, num_planes, num_blocks, &filter_args[4]
    );
}
