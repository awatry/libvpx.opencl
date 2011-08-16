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

#include "vpx_ports/config.h"
#include "vp8_opencl.h"
#include "blockd_cl.h"

//#include "loopfilter_cl.h"
//#include "../onyxc_int.h"

typedef unsigned char uc;

static void vp8_loop_filter_cl_run(
    cl_command_queue cq,
    cl_kernel kernel,
    cl_mem buf_mem,
    int num_planes,
    int offsets[],
    int *pitches,
    cl_mem lfi_mem,
    int filter_level,
    int use_mbflim,
    int *thread_counts
){

    size_t global[3] = {thread_counts[0], num_planes, 1};
    int err, i;

    cl_mem offsets_mem, pitches_mem, threads_mem;

    for ( i = 0 ; i < num_planes; i++){
        if (thread_counts[i] > global[0])
            global[0] = thread_counts[i];
    }

    VP8_CL_CREATE_BUF(cq, offsets_mem, , sizeof(cl_int)*num_planes, offsets,, );
    VP8_CL_CREATE_BUF(cq, pitches_mem, , sizeof(cl_int)*num_planes, pitches,, );
    VP8_CL_CREATE_BUF(cq, threads_mem, , sizeof(cl_int)*num_planes, thread_counts,, );

    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof (cl_mem), &buf_mem);
    err |= clSetKernelArg(kernel, 1, sizeof (cl_mem), &offsets_mem);
    err |= clSetKernelArg(kernel, 2, sizeof (cl_mem), &pitches_mem);
    err |= clSetKernelArg(kernel, 3, sizeof (cl_mem), &lfi_mem);
    err |= clSetKernelArg(kernel, 4, sizeof (cl_int), &filter_level);
    err |= clSetKernelArg(kernel, 5, sizeof (cl_int), &use_mbflim);
    err |= clSetKernelArg(kernel, 6, sizeof (cl_mem), &threads_mem);
    VP8_CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",,
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(cq, kernel, 3, NULL, global, NULL , 0, NULL, NULL);
    VP8_CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);,
    );

    VP8_CL_FINISH(cq);
    clReleaseMemObject(offsets_mem);
    clReleaseMemObject(pitches_mem);
    clReleaseMemObject(threads_mem);
    //VP8_CL_FINISH(cq);

}

void vp8_loop_filter_horizontal_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int num_planes,
    int s_off[],
    int p[], /* pitch */
    cl_mem lfi_mem,
    int filter_level,
    int use_mbflim,
    int count[]
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_horizontal_edge_kernel, s_base, num_planes, s_off,
        p, lfi_mem, filter_level, use_mbflim, count
    );
}

void vp8_loop_filter_vertical_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int num_planes,
    int s_off[],
    int p[],
    cl_mem lfi_mem,
    int filter_level,
    int use_mbflim,
    int count[]
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_vertical_edge_kernel, s_base, num_planes, s_off,
        p, lfi_mem, filter_level, use_mbflim, count
    );
}

void vp8_mbloop_filter_horizontal_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int num_planes,
    int s_off[],
    int p[],
    cl_mem lfi_mem,
    int filter_level,
    int use_mbflim,
    int count[]
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_mbloop_filter_horizontal_edge_kernel, s_base, num_planes, s_off,
        p, lfi_mem, filter_level, use_mbflim, count
    );
}


void vp8_mbloop_filter_vertical_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int num_planes,
    int s_off[],
    int p[],
    cl_mem lfi_mem,
    int filter_level,
    int use_mbflim,
    int count[]
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_mbloop_filter_vertical_edge_kernel, s_base, num_planes, s_off,
        p, lfi_mem, filter_level, use_mbflim, count
    );
}

void vp8_loop_filter_simple_horizontal_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int num_planes,
    int s_off[],
    int p[],
    cl_mem lfi_mem,
    int filter_level,
    int use_mbflim,
    int count[]
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_simple_horizontal_edge_kernel, s_base, num_planes, s_off,
        p, lfi_mem, filter_level, use_mbflim, count
    );
}

void vp8_loop_filter_simple_vertical_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int num_planes,
    int s_off[],
    int p[],
    cl_mem lfi_mem,
    int filter_level,
    int use_mbflim,
    int count[]
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_simple_vertical_edge_kernel, s_base, num_planes, s_off,
        p, lfi_mem, filter_level, use_mbflim, count
    );
}
