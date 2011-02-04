#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant int cospi8sqrt2minus1 = 20091;
__constant int sinpi8sqrt2      = 35468;
__constant int rounding = 0;


kernel void vp8_short_idct4x4llm_1st_pass_kernel(global short*,global short *,int);
kernel void vp8_short_idct4x4llm_2nd_pass_kernel(global short*,int);

__kernel void vp8_short_idct4x4llm_kernel(
    __global short *input,
    __global short *output,
    int pitch
){
    vp8_short_idct4x4llm_1st_pass_kernel(input,output,pitch);
    vp8_short_idct4x4llm_2nd_pass_kernel(output,pitch);
}

__kernel void vp8_short_idct4x4llm_1st_pass_kernel(
    __global short *ip,
    __global short *op,
    int pitch
)
{
    int i;
    int a1, b1, c1, d1;

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

    return;
}

__kernel void vp8_short_idct4x4llm_2nd_pass_kernel(
    __global short *output,
    int pitch
)
{
    int i;
    int a1, b1, c1, d1;

    int temp1, temp2;
    int shortpitch = pitch >> 1;
    __global short *ip = output;
    __global short *op = output;

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

    return;
}

__kernel void vp8_short_idct4x4llm_1_kernel(
    __global short *input,
    __global short *output,
    int pitch
)
{
    int a1;
    int out_offset;
    int shortpitch = pitch >> 1;

    //short4 a;
    a1 = ((input[0] + 4) >> 3);
    //a = a1;

    int tid = get_global_id(0);
    if (tid < 4){
        out_offset = shortpitch * tid;

        //vstore4(a,0,&output[out_offset];
        output[out_offset] = a1;
        output[out_offset+1] = a1;
        output[out_offset+2] = a1;
        output[out_offset+3] = a1;
    }
}

__kernel void vp8_dc_only_idct_add_kernel(
    short input_dc,
    __global unsigned char *pred_ptr,
    __global unsigned char *dst_ptr,
    int pitch,
    int stride
)
{
    int a1 = ((input_dc + 4) >> 3);
    int r, c;
    int pred_offset,dst_offset;

    int tid = get_global_id(0);
    if (tid < 16){
        r = tid / 4;
        c = tid % 4;

        pred_offset = r * pitch;
        dst_offset = r * stride;
        int a = a1 + pred_ptr[pred_offset + c] ;

        if (a < 0)
            a = 0;
        else if (a > 255)
            a = 255;

        dst_ptr[dst_offset + c] = (unsigned char) a ;
    }
}

__kernel void vp8_short_inv_walsh4x4_kernel(
    __global short *src_base,
    int src_offset,
    __global short *output_base,
    int out_offset
)
{

    __global short *input = src_base + src_offset;
    __global short *output = output_base + src_offset;

    //4-short vectors to calculate things in
    short4 a,b,c,d, a2v, b2v, c2v, d2v, a1t, b1t, c1t, d1t;
    short16 out;
    int tid = get_global_id(0);

    if (tid == 0){
        //first pass loop in vector form
        a = vload4(0,input) + vload4(3,input);
        b = vload4(1,input) + vload4(2,input);
        c = vload4(1,input) - vload4(2,input);
        d = vload4(0,input) - vload4(3,input);
        vstore4(a + b, 0, output);
        vstore4(c + d, 1, output);
        vstore4(a - b, 2, output);
        vstore4(d - c, 3, output);

        //2nd pass
        a = (short4)(output[0], output[4], output[8], output[12]);
        b = (short4)(output[1], output[5], output[9], output[13]);
        c = (short4)(output[1], output[5], output[9], output[13]);
        d = (short4)(output[0], output[4], output[8], output[12]);
        a1t = (short4)(output[3], output[7], output[11], output[15]);
        b1t = (short4)(output[2], output[6], output[10], output[14]);
        c1t = (short4)(output[2], output[6], output[10], output[14]);
        d1t = (short4)(output[3], output[7], output[11], output[15]);

        a = a + a1t + (short)3;
        b = b + b1t;
        c = c - c1t;
        d = d - d1t + (short)3;

        a2v = (a + b) >> (short)3;
        b2v = (c + d) >> (short)3;
        c2v = (a - b) >> (short)3;
        d2v = (d - c) >> (short)3;

        out.s048c = a2v;
        out.s159d = b2v;
        out.s26ae = c2v;
        out.s37bf = d2v;
        vstore16(out,0,output);
    }
}

__kernel void vp8_short_inv_walsh4x4_1_kernel(
    __global short *src_data,
    int src_offset,
    __global short *dst_data,
    int dst_offset
){
    int a1;
    int tid = get_global_id(0);
    short16 a;
    __global short *input = src_data + src_offset;
    __global short *output = dst_data + dst_offset;

    if (tid == 0)
    {
        a1 = ((input[0] + 3) >> 3);
        a = (short)a1; //Set all elements of vector to a1
        vstore16(a, 0, output);
    }
}
