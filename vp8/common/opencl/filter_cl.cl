#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant int bilinear_filters[8][2] = {
    { 128, 0},
    { 112, 16},
    { 96, 32},
    { 80, 48},
    { 64, 64},
    { 48, 80},
    { 32, 96},
    { 16, 112}
};

__constant short sub_pel_filters[8][6] = {
    { 0, 0, 128, 0, 0, 0}, /* note that 1/8 pel positions are just as per alpha -0.5 bicubic */
    { 0, -6, 123, 12, -1, 0},
    { 2, -11, 108, 36, -8, 1}, /* New 1/4 pel 6 tap filter */
    { 0, -9, 93, 50, -6, 0},
    { 3, -16, 77, 77, -16, 3}, /* New 1/2 pel 6 tap filter */
    { 0, -6, 50, 93, -9, 0},
    { 1, -8, 36, 108, -11, 2}, /* New 1/4 pel 6 tap filter */
    { 0, -1, 12, 123, -6, 0},
};

void vp8_filter_block2d_first_pass(
    __global unsigned char *src_ptr,
    __local int *output_ptr,
    unsigned int src_pixels_per_line,
    unsigned int pixel_step,
    unsigned int output_height,
    unsigned int output_width,
    int filter_offset
){
    uint tid = get_global_id(0);
    uint i = tid;

    int src_offset;
    int Temp;
    int PS2 = 2*(int)pixel_step;
    int PS3 = 3*(int)pixel_step;

    __constant short *vp8_filter = sub_pel_filters[filter_offset];

    if (tid < (output_width*output_height)){
        for (i=0; i < output_width*output_height; i++){
            src_offset = i + (i/output_width * (src_pixels_per_line - output_width)) + PS2;

            Temp = (int)(src_ptr[src_offset - PS2]      * vp8_filter[0]) +
               (int)(src_ptr[src_offset - (int)pixel_step] * vp8_filter[1]) +
               (int)(src_ptr[src_offset]                * vp8_filter[2]) +
               (int)(src_ptr[src_offset + pixel_step]   * vp8_filter[3]) +
               (int)(src_ptr[src_offset + PS2]          * vp8_filter[4]) +
               (int)(src_ptr[src_offset + PS3]          * vp8_filter[5]) +
               (VP8_FILTER_WEIGHT >> 1);      /* Rounding */

            /* Normalize back to 0-255 */
            Temp = Temp >> VP8_FILTER_SHIFT;

            //Temp = (int)src_ptr[2];
            if (Temp < 0)
                Temp = 0;
            else if ( Temp > 255 )
                Temp = 255;

            output_ptr[i] = Temp;
        }
    }

    //Add a fence/barrier so that no 2nd pass stuff starts before 1st pass is done.
    //write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    barrier(CLK_GLOBAL_MEM_FENCE);
}

void vp8_filter_block2d_second_pass
(
    __local int *src_ptr,
    __global unsigned char *output_ptr,
    int output_pitch,
    unsigned int src_pixels_per_line,
    unsigned int pixel_step,
    unsigned int output_height,
    unsigned int output_width,
    int filter_offset
) {

    int out_offset;
    int src_offset;
    int Temp;
    int PS2 = 2*(int)pixel_step;
    int PS3 = 3*(int)pixel_step;

    unsigned int src_increment = src_pixels_per_line - output_width;

    uint i = get_global_id(0);

    __constant short *vp8_filter = sub_pel_filters[filter_offset];

    if (i < (output_width * output_height)){
        out_offset = i/output_width;
        src_offset = out_offset;

        src_offset = i + (src_offset * src_increment);
        out_offset = i%output_width + (out_offset * output_pitch);

        /* Apply filter */
        Temp = ((int)src_ptr[src_offset - PS2] * vp8_filter[0]) +
           ((int)src_ptr[src_offset -(int)pixel_step] * vp8_filter[1]) +
           ((int)src_ptr[src_offset]                  * vp8_filter[2]) +
           ((int)src_ptr[src_offset + pixel_step]     * vp8_filter[3]) +
           ((int)src_ptr[src_offset + PS2]       * vp8_filter[4]) +
           ((int)src_ptr[src_offset + PS3]       * vp8_filter[5]) +
           (VP8_FILTER_WEIGHT >> 1);   /* Rounding */

        /* Normalize back to 0-255 */
        Temp = Temp >> VP8_FILTER_SHIFT;
        if (Temp < 0)
            Temp = 0;
        else if (Temp > 255)
            Temp = 255;

        output_ptr[out_offset] = (unsigned char)Temp;
        //output_ptr[i] = (unsigned char)out_offset;
    }
}


__kernel void vp8_block_variation_kernel
(
    __global unsigned char  *src_ptr,
    int   src_pixels_per_line,
    __global int *HVar,
    __global int *VVar
)
{
    int i, j;
    __global unsigned char *Ptr = src_ptr;

    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
        {
            *HVar += abs((int)Ptr[j] - (int)Ptr[j+1]);
            *VVar += abs((int)Ptr[j] - (int)Ptr[j+src_pixels_per_line]);
        }

        Ptr += src_pixels_per_line;
    }
}

__kernel void vp8_sixtap_predict_kernel
(
    __global unsigned char  *src_ptr,
    int  src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    __global unsigned char *dst_ptr,
    int  dst_pitch
        ) {

    __local int FData[9*4]; /* Temp data buffer used in filtering */

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_first_pass(src_ptr, FData, src_pixels_per_line, 1, 9, 4, xoffset);

    /* then filter verticaly... */
    vp8_filter_block2d_second_pass(&FData[8], dst_ptr, dst_pitch, 4, 4, 4, 4, yoffset);
}

__kernel void vp8_sixtap_predict8x8_kernel
(
    __global unsigned char  *src_ptr,
    int  src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    __global unsigned char *dst_ptr,
    int  dst_pitch
)
{
    __local int FData[13*16];   /* Temp data bufffer used in filtering */

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_first_pass(src_ptr, FData, src_pixels_per_line, 1, 13, 8, xoffset);

    /* then filter verticaly... */
    vp8_filter_block2d_second_pass(&FData[16], dst_ptr, dst_pitch, 8, 8, 8, 8, yoffset);

}

__kernel void vp8_sixtap_predict8x4_kernel
(
    __global unsigned char  *src_ptr,
    int  src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    __global unsigned char *dst_ptr,
    int  dst_pitch
)
{
    __local int FData[13*16];   /* Temp data buffer used in filtering */

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_first_pass(src_ptr, FData, src_pixels_per_line, 1, 9, 8, xoffset);

    /* then filter verticaly... */
    vp8_filter_block2d_second_pass(&FData[16], dst_ptr, dst_pitch, 8, 8, 4, 8, yoffset);

}

__kernel void vp8_sixtap_predict16x16_kernel
(
    __global unsigned char  *src_ptr,
    int  src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    __global unsigned char *dst_ptr,
    int  dst_pitch
)
{
    __local int FData[21*24];   /* Temp data buffer used in filtering */

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_first_pass(src_ptr, FData, src_pixels_per_line, 1, 21, 16, xoffset);

    /* then filter verticaly... */
    vp8_filter_block2d_second_pass(&FData[32], dst_ptr, dst_pitch, 16, 16, 16, 16, yoffset);

}
