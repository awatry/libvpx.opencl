/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant int cospi8sqrt2minus1 = 20091;
__constant int sinpi8sqrt2      = 35468;
__constant int rounding = 0;

//Note that this kernel has been copied from common/opencl/idctllm_cl.cl
void vp8_short_idct4x4llm(
    __global short *input,
    __global short *output,
    int pitch
)
{
    int i;
    int a1, b1, c1, d1;

    __global short *ip = input;
    __global short *op = output;
    int temp1, temp2;
    int shortpitch = pitch >> 1;

    for (i = 0; i < 4; i++)
    {
        a1 = ip[0] + ip[8];
        b1 = ip[0] - ip[8];

        temp1 = (ip[4] * sinpi8sqrt2 + rounding) >> 16;
        temp2 = ip[12] + ((ip[12] * cospi8sqrt2minus1 + rounding) >> 16);
        c1 = temp1 - temp2;

        temp1 = ip[4] + ((ip[4] * cospi8sqrt2minus1 + rounding) >> 16);
        temp2 = (ip[12] * sinpi8sqrt2 + rounding) >> 16;
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

        temp1 = (ip[1] * sinpi8sqrt2 + rounding) >> 16;
        temp2 = ip[3] + ((ip[3] * cospi8sqrt2minus1 + rounding) >> 16);
        c1 = temp1 - temp2;

        temp1 = ip[1] + ((ip[1] * cospi8sqrt2minus1 + rounding) >> 16);
        temp2 = (ip[3] * sinpi8sqrt2 + rounding) >> 16;
        d1 = temp1 + temp2;


        op[0] = (a1 + d1 + 4) >> 3;
        op[3] = (a1 - d1 + 4) >> 3;

        op[1] = (b1 + c1 + 4) >> 3;
        op[2] = (b1 - c1 + 4) >> 3;

        ip += shortpitch;
        op += shortpitch;
    }

}

void cl_memset(__global short *input, short val, size_t nbytes){
    __global short *cur = input;
    while (cur < input + nbytes){
        *cur == val;
        cur += sizeof(short);
    }
}

__kernel void vp8_dequantize_b_kernel(
    __global short *dqoeff_base,
    int dqcoeff_offset,
    __global short *qcoeff_base,
    int qcoeff_offset,
    __global short *dequant_base,
    int dequant_offset
)
{
    //int i;
    __global short *DQ  = dqcoeff_base + dqcoeff_offset;
    __global short *Q   = qcoeff_base + qcoeff_offset;
    __global short *DQC = dequant_base + quant_offset;

    short16 dqv = vload16(0,Q) * vload16(0,DQC);
    vstore16(dqv, 0, DQ);

    //for (i = 0; i < 16; i++)
    //{
    //    DQ[i] = Q[i] * DQC[i];
    //}
}

__kernel void vp8_dequant_idct_add_kernel(short *input, short *dq, unsigned char *pred,
                            unsigned char *dest, int pitch, int stride)
{
    short output[16];
    short *diff_ptr = output;
    int r, c;
    int i;

    vstore16( (short16)vload16(0,dq) * (short16)vload16(0,input) , 0, input);
    //for (i = 0; i < 16; i++)
    //{
    //    input[i] = dq[i] * input[i];
    //}

    /* the idct halves ( >> 1) the pitch */
    vp8_short_idct4x4llm_kernel(input, output, 4 << 1);

    cl_memset(input, 0, 32);

    for (r = 0; r < 4; r++)
    {
        for (c = 0; c < 4; c++)
        {
            int a = diff_ptr[c] + pred[c];

            if (a < 0)
                a = 0;

            if (a > 255)
                a = 255;

            dest[c] = (unsigned char) a;
        }

        dest += stride;
        diff_ptr += 4;
        pred += pitch;
    }
}

__kernel void vp8_dequant_dc_idct_add_kernel(
    __global short *input, 
    __global short *dq,
    __global unsigned char *pred,
    __global unsigned char *dest,
    int pitch,
    int stride,
    int Dc
)
{
    int i;
    short output[16];
    short *diff_ptr = output;
    int r, c;

    input[0] = (short)Dc;

    vstore16( (short16)vload16(0,dq) * (short16)vload16(0,input) , 0, input);
    //for (i = 1; i < 16; i++)
    //{
    //    input[i] = dq[i] * input[i];
    //}

    /* the idct halves ( >> 1) the pitch */
    vp8_short_idct4x4llm(input, output, 4 << 1);

    cl_memset(input, 0, 32);

    for (r = 0; r < 4; r++)
    {
        for (c = 0; c < 4; c++)
        {
            int a = diff_ptr[c] + pred[c];

            if (a < 0)
                a = 0;

            if (a > 255)
                a = 255;

            dest[c] = (unsigned char) a;
        }

        dest += stride;
        diff_ptr += 4;
        pred += pitch;
    }
}
