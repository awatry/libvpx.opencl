#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant int cospi8sqrt2minus1 = 20091;
__constant int sinpi8sqrt2      = 35468;
__constant int rounding = 0;

__kernel void vp8_short_idct4x4llm_kernel(
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

__kernel void vp8_short_idct4x4llm_1_kernel(
    __global short *input,
    __global short *output,
    int pitch
)
{
    //int i;
    int a1;
    int out_offset;
    //__global short *op = output;
    int shortpitch = pitch >> 1;
    a1 = ((input[0] + 4) >> 3);

    int tid = get_global_id(0);
    if (tid < 4){
        out_offset = shortpitch * tid;
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
    __global short *input,
    __global short *output
)
{
    int i;
    int a1, b1, c1, d1;
    int a2, b2, c2, d2;
    __global short *ip = input;
    __global short *op = output;

    for (i = 0; i < 4; i++)
    {
        a1 = ip[0] + ip[12];
        b1 = ip[4] + ip[8];
        c1 = ip[4] - ip[8];
        d1 = ip[0] - ip[12];

        op[0] = a1 + b1;
        op[4] = c1 + d1;
        op[8] = a1 - b1;
        op[12] = d1 - c1;
        ip++;
        op++;
    }

    ip = output;
    op = output;

    for (i = 0; i < 4; i++)
    {
        a1 = ip[0] + ip[3];
        b1 = ip[1] + ip[2];
        c1 = ip[1] - ip[2];
        d1 = ip[0] - ip[3];

        a2 = a1 + b1;
        b2 = c1 + d1;
        c2 = a1 - b1;
        d2 = d1 - c1;

        op[0] = (a2 + 3) >> 3;
        op[1] = (b2 + 3) >> 3;
        op[2] = (c2 + 3) >> 3;
        op[3] = (d2 + 3) >> 3;

        ip += 4;
        op += 4;
    }

}

__kernel void vp8_short_inv_walsh4x4_1_kernel(
//    __global short *input,
    short input,
    __global short *output
){
    int a1;
    int tid;
    tid = get_global_id(0);

    if (tid < 16)
    {
        a1 = ((input + 3) >> 3);
        output[tid] = (short)a1;
    }
}
