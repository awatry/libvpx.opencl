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

typedef unsigned char uc;

typedef struct VP8_LOOPFILTER_ARGS{
    cl_mem buf_mem;
    cl_mem offsets_mem;
    cl_mem pitches_mem;
    cl_mem lfi_mem;
    cl_mem filter_level_mem;
    cl_int use_mbflim;
    cl_mem threads_mem;
    cl_mem apply_filter_mem;
} VP8_LOOPFILTER_ARGS;

static int first_run = 0;
static VP8_LOOPFILTER_ARGS filter_args[6];

#define VP8_CL_SET_LOOP_ARG(kernel, current, name, type, argnum) \
    if (current->name != name){ \
        err |= clSetKernelArg(kernel, argnum, sizeof (type), &name); \
        current->name = name; \
    }\


static void vp8_loop_filter_cl_run(
    cl_command_queue cq,
    cl_kernel kernel,
    cl_mem buf_mem,
    int num_planes,
    int num_blocks,
    cl_mem offsets_mem,
    cl_mem pitches_mem,
    cl_mem lfi_mem,
    cl_mem filter_level_mem,
    int use_mbflim,
    cl_mem threads_mem,
    int max_threads,
    cl_mem apply_filter_mem,
    VP8_LOOPFILTER_ARGS *current_args
){

    size_t global[3] = {max_threads, num_planes, num_blocks};
    int err;

    if (first_run == 0){
        memset(filter_args, -1, sizeof(VP8_LOOPFILTER_ARGS)*6);
        first_run = 1;
    }
    
    err = 0;
    VP8_CL_SET_LOOP_ARG(kernel, current_args, buf_mem, cl_mem, 0)
    VP8_CL_SET_LOOP_ARG(kernel, current_args, offsets_mem, cl_mem, 1)
    VP8_CL_SET_LOOP_ARG(kernel, current_args, pitches_mem, cl_mem, 2)
    VP8_CL_SET_LOOP_ARG(kernel, current_args, lfi_mem, cl_mem, 3)
    VP8_CL_SET_LOOP_ARG(kernel, current_args, filter_level_mem, cl_mem, 4)
    VP8_CL_SET_LOOP_ARG(kernel, current_args, use_mbflim, cl_int, 5)
    VP8_CL_SET_LOOP_ARG(kernel, current_args, threads_mem, cl_mem, 6)
    VP8_CL_SET_LOOP_ARG(kernel, current_args, apply_filter_mem, cl_mem, 7)
    VP8_CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",,
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(cq, kernel, 3, NULL, global, NULL , 0, NULL, NULL);
    VP8_CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);,
    );
}

void vp8_loop_filter_horizontal_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int num_planes,
    int num_blocks,
    cl_mem offsets_mem,
    cl_mem pitches_mem, /* pitch */
    cl_mem lfi_mem,
    cl_mem filter_level_mem,
    int use_mbflim,
    cl_mem threads_mem,
    int max_threads,
    cl_mem apply_filter_mem
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_horizontal_edge_kernel, s_base, num_planes, num_blocks, offsets_mem,
        pitches_mem, lfi_mem, filter_level_mem, use_mbflim, threads_mem, max_threads, apply_filter_mem, &filter_args[0]
    );
}

void vp8_loop_filter_vertical_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int num_planes,
    int num_blocks,
    cl_mem offsets_mem,
    cl_mem pitches_mem,
    cl_mem lfi_mem,
    cl_mem filter_level_mem,
    int use_mbflim,
    cl_mem threads_mem,
    int max_threads,
    cl_mem apply_filter_mem
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_vertical_edge_kernel, s_base, num_planes, num_blocks, offsets_mem,
        pitches_mem, lfi_mem, filter_level_mem, use_mbflim, threads_mem, max_threads, apply_filter_mem, &filter_args[1]
    );
}

void vp8_mbloop_filter_horizontal_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int num_planes,
    int num_blocks,
    cl_mem offsets_mem,
    cl_mem pitches_mem,
    cl_mem lfi_mem,
    cl_mem filter_level_mem,
    int use_mbflim,
    cl_mem threads_mem,
    int max_threads,
    cl_mem apply_filter_mem
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_mbloop_filter_horizontal_edge_kernel, s_base, num_planes, num_blocks, offsets_mem,
        pitches_mem, lfi_mem, filter_level_mem, use_mbflim, threads_mem, max_threads, apply_filter_mem, &filter_args[2]
    );
}


void vp8_mbloop_filter_vertical_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int num_planes,
    int num_blocks,
    cl_mem offsets_mem,
    cl_mem pitches_mem,
    cl_mem lfi_mem,
    cl_mem filter_level_mem,
    int use_mbflim,
    cl_mem threads_mem,
    int max_threads,
    cl_mem apply_filter_mem
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_mbloop_filter_vertical_edge_kernel, s_base, num_planes, num_blocks, offsets_mem,
        pitches_mem, lfi_mem, filter_level_mem, use_mbflim, threads_mem, max_threads, apply_filter_mem, &filter_args[3]
    );
}

void vp8_loop_filter_simple_horizontal_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int num_planes,
    int num_blocks,
    cl_mem offsets_mem,
    cl_mem pitches_mem,
    cl_mem lfi_mem,
    cl_mem filter_level_mem,
    int use_mbflim,
    cl_mem threads_mem,
    int max_threads,
    cl_mem apply_filter_mem
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_simple_horizontal_edge_kernel, s_base, num_planes, num_blocks, offsets_mem,
        pitches_mem, lfi_mem, filter_level_mem, use_mbflim, threads_mem, max_threads, apply_filter_mem, &filter_args[4]
    );
}

void vp8_loop_filter_simple_vertical_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int num_planes,
    int num_blocks,
    cl_mem offsets_mem,
    cl_mem pitches_mem,
    cl_mem lfi_mem,
    cl_mem filter_level_mem,
    int use_mbflim,
    cl_mem threads_mem,
    int max_threads,
    cl_mem apply_filter_mem
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_simple_vertical_edge_kernel, s_base, num_planes, num_blocks, offsets_mem,
        pitches_mem, lfi_mem, filter_level_mem, use_mbflim, threads_mem, max_threads, apply_filter_mem, &filter_args[5]
    );
}
