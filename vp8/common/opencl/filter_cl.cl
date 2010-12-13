__kernel void vp8_filter_block2d_first_pass_kernel(
    __global    unsigned char *src_ptr,
    __global   int *output_ptr,
    unsigned int src_pixels_per_line,
    unsigned int pixel_step,
    unsigned int output_height,
    unsigned int output_width,
    __global   const short *vp8_filter
){

    uint tid = get_global_id(0);
    
    int out_offset,src_offset;
    int PS2 = 2*(int)pixel_step;

    src_offset = tid/output_width;
    out_offset = src_offset;
    /* ACW: Umm... the following line should probably just be out_offset=tid;*/
    out_offset = (tid - out_offset*output_width) + (out_offset * output_width);
    src_offset = tid + (src_offset * (src_pixels_per_line - output_width));
    int Temp = ((int)src_ptr[src_offset - PS2]           * vp8_filter[0]) +
           ((int)src_ptr[src_offset - (int)pixel_step]   * vp8_filter[1]) +
           ((int)src_ptr[src_offset]                     * vp8_filter[2]) +
           ((int)src_ptr[src_offset + pixel_step]        * vp8_filter[3]) +
           ((int)src_ptr[src_offset + PS2]               * vp8_filter[4]) +
           ((int)src_ptr[src_offset + 3*(int)pixel_step] * vp8_filter[5]) +
           (VP8_FILTER_WEIGHT >> 1);      /* Rounding */

    /* Normalize back to 0-255 */
    Temp = Temp >> VP8_FILTER_SHIFT;

    if (Temp < 0)
        Temp = 0;
    else if (Temp > 255)
        Temp = 255;

    output_ptr[out_offset] = Temp;
}