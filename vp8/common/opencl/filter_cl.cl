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

__kernel void vp8_filter_block2d_first_pass_kernel(
    __global unsigned char *src_ptr,
    __global int *output_ptr,
    unsigned int src_pixels_per_line,
    unsigned int pixel_step,
    unsigned int output_height,
    unsigned int output_width,
    __global short *vp8_filter)
{
    uint i = get_global_id(0);

    int src_offset;
    int Temp;
    int PS2 = 2*(int)pixel_step;
    int PS3 = 3*(int)pixel_step;

    if (i < (output_width*output_height)){
        src_offset = i + (i/output_width * (src_pixels_per_line - output_width)) + PS2;

        Temp = ((int)*(src_ptr+src_offset - PS2)      * vp8_filter[0]) +
           ((int)*(src_ptr+src_offset - (int)pixel_step) * vp8_filter[1]) +
           ((int)*(src_ptr+src_offset)                * vp8_filter[2]) +
           ((int)*(src_ptr+src_offset + pixel_step)   * vp8_filter[3]) +
           ((int)*(src_ptr+src_offset + PS2)          * vp8_filter[4]) +
           ((int)*(src_ptr+src_offset + PS3)          * vp8_filter[5]) +
           (VP8_FILTER_WEIGHT >> 1);      /* Rounding */

        /* Normalize back to 0-255 */
        Temp = Temp >> VP8_FILTER_SHIFT;
        if (Temp < 0)
            Temp = 0;
        else if ( Temp > 255 )
            Temp = 255;

        output_ptr[i] = Temp;
    }
}