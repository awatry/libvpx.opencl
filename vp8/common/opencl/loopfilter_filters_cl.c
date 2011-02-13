/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
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

signed char vp8_signed_char_clamp(int t)
{
    t = (t < -128 ? -128 : t);
    t = (t > 127 ? 127 : t);
    return (signed char) t;
}


/* should we apply any filter at all ( 11111111 yes, 00000000 no) */
signed char vp8_filter_mask(signed char limit, signed char flimit,
                                     uc p3, uc p2, uc p1, uc p0, uc q0, uc q1, uc q2, uc q3)
{
    signed char mask = 0;
    mask |= (abs(p3 - p2) > limit) * -1;
    mask |= (abs(p2 - p1) > limit) * -1;
    mask |= (abs(p1 - p0) > limit) * -1;
    mask |= (abs(q1 - q0) > limit) * -1;
    mask |= (abs(q2 - q1) > limit) * -1;
    mask |= (abs(q3 - q2) > limit) * -1;
    mask |= (abs(p0 - q0) * 2 + abs(p1 - q1) / 2  > flimit * 2 + limit) * -1;
    mask = ~mask;
    return mask;
}

/* is there high variance internal edge ( 11111111 yes, 00000000 no) */
signed char vp8_hevmask(signed char thresh, uc p1, uc p0, uc q0, uc q1)
{
    signed char hev = 0;
    hev  |= (abs(p1 - p0) > thresh) * -1;
    hev  |= (abs(q1 - q0) > thresh) * -1;
    return hev;
}

static void vp8_loop_filter_cl_run(
    cl_command_queue cq,
    cl_kernel kernel,
    cl_mem buf_mem,
    int s_off,
    int p,
    const signed char *flimit,
    const signed char *limit,
    const signed char *thresh,
    int count
){
    size_t global = count * 2;
    int err;

    cl_mem flimit_mem;
    cl_mem limit_mem;
    cl_mem thresh_mem;

    CL_CREATE_BUF(cq, flimit_mem, , sizeof(uc)*16, flimit, );
    CL_CREATE_BUF(cq, limit_mem, , sizeof(uc)*16, limit, );
    CL_CREATE_BUF(cq, thresh_mem, , sizeof(uc)*16, thresh, );

    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof (cl_mem), &buf_mem);
    err |= clSetKernelArg(kernel, 1, sizeof (cl_int), &s_off);
    err |= clSetKernelArg(kernel, 2, sizeof (cl_int), &p);
    err |= clSetKernelArg(kernel, 3, sizeof (cl_mem), &flimit_mem);
    err |= clSetKernelArg(kernel, 4, sizeof (cl_mem), &limit_mem);
    err |= clSetKernelArg(kernel, 5, sizeof (cl_mem), &thresh_mem);
    err |= clSetKernelArg(kernel, 6, sizeof (cl_int), &count);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",,
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(cq, kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);,
    );

    clReleaseMemObject(flimit_mem);
    clReleaseMemObject(limit_mem);
    clReleaseMemObject(thresh_mem);

    CL_FINISH(cq);
}


static void vp8_filter(MACROBLOCKD *x, signed char mask, signed char hev, uc *base, int op1_off, int op0_off, int oq0_off, int oq1_off)
{
    int err;
    cl_mem op1_mem, op0_mem, oq0_mem, oq1_mem;
    size_t global = 1;

    uc *op1 = base + op1_off;
    uc *op0 = base + op0_off;
    uc *oq0 = base + oq0_off;
    uc *oq1 = base + oq1_off;

    cl_command_queue cq = x->cl_commands;
   
    op1_off = op0_off = oq0_off = oq1_off = 0;

    CL_CREATE_BUF(cq, op1_mem, , sizeof(uc), op1, );
    CL_CREATE_BUF(cq, op0_mem, , sizeof(uc), op0, );
    CL_CREATE_BUF(cq, oq0_mem, , sizeof(uc), oq0, );
    CL_CREATE_BUF(cq, oq1_mem, , sizeof(uc), oq1, );

    err = 0;
    err = clSetKernelArg(cl_data.vp8_filter_kernel, 0, sizeof (signed char), &mask);
    err |= clSetKernelArg(cl_data.vp8_filter_kernel, 1, sizeof (signed char), &hev);
    err |= clSetKernelArg(cl_data.vp8_filter_kernel, 2, sizeof (cl_mem), &op1_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_kernel, 3, sizeof (int), &op1_off);
    err |= clSetKernelArg(cl_data.vp8_filter_kernel, 4, sizeof (cl_mem), &op0_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_kernel, 5, sizeof (int), &op0_off);
    err |= clSetKernelArg(cl_data.vp8_filter_kernel, 6, sizeof (cl_mem), &oq0_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_kernel, 7, sizeof (int), &oq0_off);
    err |= clSetKernelArg(cl_data.vp8_filter_kernel, 8, sizeof (cl_mem), &oq1_mem);
    err |= clSetKernelArg(cl_data.vp8_filter_kernel, 9, sizeof (int), &oq1_off);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",,
    );

    /* Execute the kernel */
    err = clEnqueueNDRangeKernel(cq, cl_data.vp8_filter_kernel, 1, NULL, &global, NULL , 0, NULL, NULL);
    CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to execute kernel!\n",
        printf("err = %d\n",err);,
    );

    /* Read back the result data from the device */
    err = clEnqueueReadBuffer(cq, op1_mem, CL_FALSE, 0, sizeof(uc), op1, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(cq, op0_mem, CL_FALSE, 0, sizeof(uc), op0, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(cq, oq1_mem, CL_FALSE, 0, sizeof(uc), oq1, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(cq, oq0_mem, CL_FALSE, 0, sizeof(uc), oq0, 0, NULL, NULL);
    CL_CHECK_SUCCESS(cq, err != CL_SUCCESS,
        "Error: Failed to read loop filter output!\n",
        ,
    );

    clReleaseMemObject(op1_mem);
    clReleaseMemObject(op0_mem);
    clReleaseMemObject(oq0_mem);
    clReleaseMemObject(oq1_mem);

}

void vp8_loop_filter_horizontal_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int s_off,
    int p, /* pitch */
    const signed char *flimit,
    const signed char *limit,
    const signed char *thresh,
    int count
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_horizontal_edge_kernel, s_base, s_off,
        p, flimit, limit, thresh, count
    );
}

void vp8_loop_filter_vertical_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int s_off,
    int p,
    const signed char *flimit,
    const signed char *limit,
    const signed char *thresh,
    int count
)
{
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_vertical_edge_kernel, s_base, s_off,
        p, flimit, limit, thresh, count
    );
}

void vp8_mbloop_filter_horizontal_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int s_off,
    int p,
    const signed char *flimit,
    const signed char *limit,
    const signed char *thresh,
    int count
)
{

    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_mbloop_filter_horizontal_edge_kernel, s_base, s_off,
        p, flimit, limit, thresh, count
    );
}


void vp8_mbloop_filter_vertical_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int s_off,
    int p,
    const signed char *flimit,
    const signed char *limit,
    const signed char *thresh,
    int count
)
{

    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_mbloop_filter_vertical_edge_kernel, s_base, s_off,
        p, flimit, limit, thresh, count
    );

}

void vp8_loop_filter_simple_horizontal_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int s_off,
    int p,
    const signed char *flimit,
    const signed char *limit,
    const signed char *thresh,
    int count
)
{
    
    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_simple_horizontal_edge_kernel, s_base, s_off,
        p, flimit, limit, thresh, count
    );
}

void vp8_loop_filter_simple_vertical_edge_cl
(
    MACROBLOCKD *x,
    cl_mem s_base,
    int s_off,
    int p,
    const signed char *flimit,
    const signed char *limit,
    const signed char *thresh,
    int count
)
{

    vp8_loop_filter_cl_run(x->cl_commands,
        cl_data.vp8_loop_filter_simple_vertical_edge_kernel, s_base, s_off,
        p, flimit, limit, thresh, count
    );

}
