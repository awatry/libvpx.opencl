#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

__constant int cospi8sqrt2minus1 = 20091;
__constant int sinpi8sqrt2      = 35468;
__constant int rounding = 0;


static const int cospi8sqrt2minus1 = 20091;
static const int sinpi8sqrt2      = 35468;

__kernel void vp8_short_idct4x4llm_c(global short *input, global unsigned char *pred_ptr,
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

__kernel void vp8_dc_only_idct_add_c(short input_dc, global unsigned char *pred_ptr,
                            int pred_stride, global unsigned char *dst_ptr,
                            int dst_stride)
{
    int a1 = ((input_dc + 4) >> 3);
    int r, c;

    for (r = 0; r < 4; r++)
    {
        for (c = 0; c < 4; c++)
        {
            int a = a1 + pred_ptr[c] ;

            if (a < 0)
                a = 0;

            if (a > 255)
                a = 255;

            dst_ptr[c] = (unsigned char) a ;
        }

        dst_ptr += dst_stride;
        pred_ptr += pred_stride;
    }

}

__kernel void vp8_short_inv_walsh4x4_c(global short *input, global short *output)
{
    int i;
    int a1, b1, c1, d1;
    int a2, b2, c2, d2;
    short *ip = input;
    short *op = output;

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

__kernel void vp8_short_inv_walsh4x4_1_c(global short *input, global short *output)
{
    int i;
    int a1;
    short *op = output;

    a1 = ((input[0] + 3) >> 3);

    for (i = 0; i < 4; i++)
    {
        op[0] = a1;
        op[1] = a1;
        op[2] = a1;
        op[3] = a1;
        op += 4;
    }
}