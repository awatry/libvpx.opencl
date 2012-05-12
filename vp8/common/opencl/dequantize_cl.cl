#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable


/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

__constant int cospi8sqrt2minus1 = 20091;
__constant int sinpi8sqrt2      = 35468;
__constant int rounding = 0;

void vp8_short_idct4x4llm(__global short*, short*, int);
void cl_memset_short(__global short*, int, size_t);

#define USE_VECTORS 0

__kernel void vp8_dequantize_b_kernel(
    __global short *dqcoeff_base,
    int dqcoeff_offset,
    __global short *qcoeff_base,
    int qcoeff_offset,
    __global short *dequant
)
{
    __global short *DQ  = dqcoeff_base + dqcoeff_offset;
    __global short *Q   = qcoeff_base  + qcoeff_offset;

#if USE_VECTORS
    vstore16(vload16(0,Q) * vload16(0,dequant), 0, DQ);
#else
    int tid = get_global_id(0);
    if (tid < 16)
    {
        DQ[tid] = Q[tid] * dequant[tid];
    }

#endif
}

__kernel void vp8_dequant_idct_add_kernel(
    __global short *input_base,
    int input_offset,
    __global short *dq,
    __global unsigned char *dest_base,
    int dest_offset,
    int stride
){
    int i;
	_global short *input = &input_base[input_offset];
	
#if USE_VECTORS
    vstore16( (short16)vload16(0,dq) * (short16)vload16(0,input) , 0, input);
#else
    for (i = 0; i < 16; i++)
    {
        input[i] = dq[i] * input[i];
    }
#endif

    vp8_short_idct4x4llm(input, &dest_base[dest_offset], stride, &dest_base[dest_offset], stride);

    cl_memset_short(input, 0, 32);

}


__kernel void vp8_dequant_dc_idct_add_kernel(
    __global short *qcoeff_base,
    int qcoeff_offset,

    __global short *dequant_base,
    int dequant_offset,

    __global unsigned char *dest,

    int stride,
    int Dc
)
{
    int i;
    short output[16];
    short *diff_ptr = output;
    int r, c;

    global short *input = &qcoeff_base[qcoeff_offset];
    global short *dq = &dequant_base[dequant_offset];

    input[0] = Dc;

#if USE_VECTORS
    vstore16( (short16)vload16(0,dq) * (short16)vload16(0,input) , 0, input);
#else
    for (i = 1; i < 16; i++)
    {
        input[i] = dq[i] * input[i];
    }
#endif
    
    vp8_short_idct4x4llm(input, dest, stride, dest, stride);

    cl_memset_short(input, 0, 32);
}




//Note that this has been copied from common/opencl/idctllm_cl.cl
void vp8_short_idct4x4llm(global short *input, global unsigned char *pred_ptr,
                            int pred_stride, global unsigned char *dst_ptr,
                            int dst_stride)
{
    int i;
    int r, c;
    int a1, b1, c1, d1;
    short output[16];
    global short *ip = input;
    short *op = output;
    int temp1, temp2;
    int shortpitch = 4;

    for (i = 0; i < 4; i++)
    {
        a1 = ip[0] + ip[8];
        b1 = ip[0] - ip[8];

        temp1 = (ip[4] * sinpi8sqrt2) >> 16;
        temp2 = ip[12] + ((ip[12] * cospi8sqrt2minus1) >> 16);
        c1 = temp1 - temp2;

        temp1 = ip[4] + ((ip[4] * cospi8sqrt2minus1) >> 16);
        temp2 = (ip[12] * sinpi8sqrt2) >> 16;
        d1 = temp1 + temp2;

        op[shortpitch*0] = a1 + d1;
        op[shortpitch*3] = a1 - d1;

        op[shortpitch*1] = b1 + c1;
        op[shortpitch*2] = b1 - c1;

        ip++;
        op++;
    }

    ip = output;
    op = output;

    for (i = 0; i < 4; i++)
    {
        a1 = ip[0] + ip[2];
        b1 = ip[0] - ip[2];

        temp1 = (ip[1] * sinpi8sqrt2) >> 16;
        temp2 = ip[3] + ((ip[3] * cospi8sqrt2minus1) >> 16);
        c1 = temp1 - temp2;

        temp1 = ip[1] + ((ip[1] * cospi8sqrt2minus1) >> 16);
        temp2 = (ip[3] * sinpi8sqrt2) >> 16;
        d1 = temp1 + temp2;


        op[0] = (a1 + d1 + 4) >> 3;
        op[3] = (a1 - d1 + 4) >> 3;

        op[1] = (b1 + c1 + 4) >> 3;
        op[2] = (b1 - c1 + 4) >> 3;

        ip += shortpitch;
        op += shortpitch;
    }

    ip = output;
    for (r = 0; r < 4; r++)
    {
        for (c = 0; c < 4; c++)
        {
            int a = ip[c] + pred_ptr[c] ;

            if (a < 0)
                a = 0;

            if (a > 255)
                a = 255;

            dst_ptr[c] = (unsigned char) a ;
        }
        ip += 4;
        dst_ptr += dst_stride;
        pred_ptr += pred_stride;
    }
}

void cl_memset_short(__global short *s, int c, size_t n) {
    int i;
    for (i = 0; i < n/2; i++)
        *s++ = c;
}
