#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

typedef unsigned char uc;
typedef signed char sc;

__inline signed char vp8_filter_mask(sc, sc, uchar8);
__inline signed char vp8_simple_filter_mask(signed char, signed char, uc, uc, uc, uc);
__inline signed char vp8_hevmask(signed char, uchar4);

__inline uchar8 vp8_mbfilter(signed char mask,signed char hev,uchar8);

void vp8_simple_filter(signed char mask,global uc *base, int op1_off,int op0_off,int oq0_off,int oq1_off);

constant int threads[3] = {16, 8, 8};

#ifndef __CL_VERSION_1_0__
#define __CL_VERSION_1_0__ 100
#endif 

#if __OPENCL_VERSION__ == __CL_VERSION_1_0__
#define clamp(x,y,z) vp8_char_clamp(x)
char vp8_char_clamp(int in){
    if (in > 127)
        return 127;
    if (in < -128)
        return -128;

    return in;
}
#endif


typedef struct
{
    signed char lim[16];
    signed char flim[16];
    signed char thr[16];
    signed char mbflim[16];
} loop_filter_info;

typedef struct
{
    int mb_rows;
    int mb_cols;
} frame_info;

uchar4 vp8_filter(
    signed char mask,
    signed char hev,
    uchar4 base
)
{
    char4 pq = convert_char4(base) ^ 0x80;

    char vp8_filter;
    char2 Filter;
    char4 u;

    /* add outer taps if we have high edge variance */
    vp8_filter = sub_sat(pq.s0, pq.s3);
    vp8_filter &= hev;

    /* inner taps */
    vp8_filter = clamp(vp8_filter + 3 * (pq.s2 - pq.s1), -128, 127);
    vp8_filter &= mask;

    /* save bottom 3 bits so that we round one side +4 and the other +3
     * if it equals 4 we'll set to adjust by -1 to account for the fact
     * we'd round 3 the other way
     */
    Filter = add_sat((char2)vp8_filter, (char2){4,3});
    Filter.s0 >>= 3;
    Filter.s1 >>= 3;
    
    /* outer tap adjustments */
    vp8_filter = Filter.s0 + 1;
    vp8_filter >>= 1;
    vp8_filter &= ~hev;

    //u.s01 = add_sat(pq.s01, (char2){vp8_filter, Filter.s1});
    //u.s23 = sub_sat(pq.s23, (char2){Filter.s0, vp8_filter});
    u.s0 = clamp(pq.s0 + vp8_filter, -128, 127);
    u.s1 = clamp(pq.s1 + Filter.s1, -128, 127);
    u.s2 = clamp(pq.s2 - Filter.s0, -128, 127);
    u.s3 = clamp(pq.s3 - vp8_filter, -128, 127);

    return convert_uchar4(u ^ 0x80);

}

// Filters horizontal edges of inner blocks in a Macroblock
void vp8_loop_filter_horizontal_edge_worker(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches, /* pitch */
    global loop_filter_info *lfi,
    global int *filters,
    int filter_type,
    int cur_iter
){
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);
    size_t num_planes = 3;
    size_t num_blocks = get_global_size(2);
    int filter_level = filters[block];
    
    if (filters[num_blocks*filter_type + block] > 0){
        if (cur_iter == 0 || plane == 0){
            if (cur_iter > 0){
                num_planes = 1;
            }

            if (filter_level){
                int p = pitches[plane];
                int block_offset = num_blocks*11 + cur_iter*num_blocks*num_planes + block*num_planes+plane;
                int s_off = offsets[block_offset];

                size_t i = get_global_id(0);

                int  hev = 0; /* high edge variance */
                signed char mask = 0;

                global signed char *limit, *flimit, *thresh;
                global loop_filter_info *lf_info;

                if (i < threads[plane]){
                    lf_info = &lfi[filter_level];
                    flimit = lf_info->flim;
                    limit = lf_info->lim;
                    thresh = lf_info->thr;

                    s_off += i;

                    uchar8 data;
                    data.s0 = s_base[s_off-4*p];
                    data.s1 = s_base[s_off-3*p];
                    data.s2 = s_base[s_off-2*p];
                    data.s3 = s_base[s_off-p];
                    data.s4 = s_base[s_off];
                    data.s5 = s_base[s_off+p];
                    data.s6 = s_base[s_off+2*p];
                    data.s7 = s_base[s_off+3*p];

                    mask = vp8_filter_mask(limit[i], flimit[i], data);

                    hev = vp8_hevmask(thresh[i], data.s2345);
                    
                    data.s2345 = vp8_filter(mask, hev, data.s2345);
                    
                    s_base[s_off - 2*p] = data.s2;
                    s_base[s_off - p  ] = data.s3;
                    s_base[s_off      ] = data.s4;
                    s_base[s_off + p  ] = data.s5;

                }
            }
        }
    }
}

void vp8_loop_filter_vertical_edge_worker(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters,
    int filter_type,
    int cur_iter
){
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);
    size_t num_planes = 3;
    size_t num_blocks = get_global_size(2);

    if ((cur_iter ==1 || plane == 0) && filters[num_blocks*filter_type + block] > 0){
        if (cur_iter > 1){
            num_planes = 1;
        }
        
        int filter_level = filters[block];
        if (filter_level){
            int p = pitches[plane];
            int block_offset = cur_iter*num_blocks*num_planes + block*num_planes+plane;
            int s_off = offsets[block_offset];

            int  hev = 0; /* high edge variance */
            signed char mask = 0;
            size_t i = get_global_id(0);

            global signed char *limit, *flimit, *thresh;
            global loop_filter_info *lf_info;

            if (i < threads[plane]){
                lf_info = &lfi[filter_level];
                flimit = lf_info->flim;

                limit = lf_info->lim;
                thresh = lf_info->thr;

                s_off += p * i;
                
                uchar8 data = vload8(0, &s_base[s_off-4]);

                mask = vp8_filter_mask(limit[i], flimit[i], data);

                hev = vp8_hevmask(thresh[i], data.s2345);
                
                data.s2345 = vp8_filter(mask, hev, data.s2345);

                vstore4(data.s2345, 0, &s_base[s_off-2]);
                
            }
        }
    }
}

void vp8_mbloop_filter_horizontal_edge_worker(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters,
    local unsigned char *s_data
){
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);
    size_t num_planes;
    size_t num_blocks;
    num_planes = 3;
    num_blocks = get_global_size(2);

    if (filters[num_blocks*ROWS_LOCATION + block] > 0){
        int filter_level = filters[block];
        if (filter_level){
            int p = pitches[plane];
            int block_offset = 8*num_blocks + block*num_planes+plane;
            int s_off = offsets[block_offset];

            signed char hev = 0; /* high edge variance */
            signed char mask = 0;
            size_t i= get_global_id(0);

            global signed char *limit, *flimit, *thresh;
            global loop_filter_info *lf_info;

            if (i < threads[plane]){
                lf_info = &lfi[filter_level];
                flimit = lf_info->mbflim;

                limit = lf_info->lim;
                thresh = lf_info->thr;

                s_off += i;
                
#if 1
                s_data += plane*128+i*8;

                s_data[0] = s_base[s_off - 4*p];
                s_data[1] = s_base[s_off - 3*p];
                s_data[2] = s_base[s_off - 2*p];
                s_data[3] = s_base[s_off - 1*p];
                s_data[4] = s_base[s_off];
                s_data[5] = s_base[s_off + p];
                s_data[6] = s_base[s_off + 2*p];
                s_data[7] = s_base[s_off + 3*p];
                uchar8 data = vload8(0, s_data);
#else
                uchar8 data;
                data.s0 = s_base[s_off-4*p];
                data.s1 = s_base[s_off-3*p];
                data.s2 = s_base[s_off-2*p];
                data.s3 = s_base[s_off-p];
                data.s4 = s_base[s_off];
                data.s5 = s_base[s_off+p];
                data.s6 = s_base[s_off+2*p];
                data.s7 = s_base[s_off+3*p];
#endif           
                
                mask = vp8_filter_mask(limit[i], flimit[i], data);

                hev = vp8_hevmask(thresh[i], data.s2345);
                
                //TODO: change vp8_mbfilter to use uchar8 instead of local uchar*
                data = vp8_mbfilter(mask, hev, data);

                s_base[s_off - 3*p] = data.s1;
                s_base[s_off - 2*p] = data.s2;
                s_base[s_off - 1*p] = data.s3;
                s_base[s_off      ] = data.s4;
                s_base[s_off + p  ] = data.s5;
                s_base[s_off + 2*p] = data.s6;
            }
        }
    }
}

kernel void vp8_loop_filter_horizontal_edges_kernel(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches, /* pitch */
    global loop_filter_info *lfi,
    global int *filters_in,
    int use_mbflim, //unused for normal filters
    int filter_type,
    int cur_iter,
    int priority_level,
    global int *block_offsets,
    global int *priority_num_blocks
){
    int block_offset = block_offsets[priority_level];
    local unsigned char s_data[16*8*3];
    
    int filter_offset = 4*block_offset;
    int priority_offset = 16*block_offset;
    
    global int *filters = &filters_in[filter_offset];

    vp8_mbloop_filter_horizontal_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, 
            filters, s_data);
    
    //YUV planes, then 2 more passes of Y plane
    vp8_loop_filter_horizontal_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, filters,
            DC_DIFFS_LOCATION, 0);
    vp8_loop_filter_horizontal_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, filters,
            DC_DIFFS_LOCATION, 3);
    vp8_loop_filter_horizontal_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, filters,
            DC_DIFFS_LOCATION, 4);
}

void vp8_mbloop_filter_vertical_edge_worker(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters,
    int filter_type,
    local unsigned char *s_data
){
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);
    size_t num_planes;
    size_t num_blocks;
    num_planes = get_global_size(1);
    num_blocks = get_global_size(2);

    if (filters[num_blocks*filter_type + block] > 0){
        int filter_level = filters[block];
        if (filter_level){
            int p = pitches[plane];
            int block_offset = block*num_planes+plane;
            int s_off = offsets[block_offset];

            signed char hev = 0; /* high edge variance */
            signed char mask = 0;
            size_t i= get_global_id(0);

            global signed char *limit, *flimit, *thresh;
            global loop_filter_info *lf_info;

            if (i < threads[plane]){
                lf_info = &lfi[filter_level];
                flimit = lf_info->mbflim;

                limit = lf_info->lim;
                thresh = lf_info->thr;

                s_off += p * i;
                s_data += plane*128+i*8;
                
                s_data[0] = s_base[s_off - 4];
                s_data[1] = s_base[s_off - 3];
                s_data[2] = s_base[s_off - 2];
                s_data[3] = s_base[s_off - 1];
                s_data[4] = s_base[s_off];
                s_data[5] = s_base[s_off + 1];
                s_data[6] = s_base[s_off + 2];
                s_data[7] = s_base[s_off + 3];
                
                uchar8 data = vload8(0, s_data);
                mask = vp8_filter_mask(limit[i], flimit[i], data);
                
                hev = vp8_hevmask(thresh[i], data.s2345);
                
                data = vp8_mbfilter(mask, hev, data);

                s_base[s_off - 3] = data.s1;
                s_base[s_off - 2] = data.s2;
                s_base[s_off - 1] = data.s3;
                s_base[s_off    ] = data.s4;
                s_base[s_off + 1] = data.s5;
                s_base[s_off + 2] = data.s6;
            }
        }
    }
}

kernel void vp8_loop_filter_all_edges_kernel(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters_in,
    int use_mbflim,
    int filter_type,
    int cur_iter,
    int priority_level,
    global int *block_offsets,
    global int *priority_num_blocks
){
    
    int block_offset = block_offsets[priority_level];
    local unsigned char s_data[16*8*3];
    int filter_offset = 4*block_offset;
    int priority_offset = 16*block_offset;
    
    global int *filters = &filters_in[filter_offset];

    //Prefetch vertical edge source pixels into global cache
    int num_blocks = get_global_size(2);
    for(int plane = 0; plane < 3; plane++){
        int p = pitches[plane];
        int offset = get_global_id(2)*3+plane;
        int s_off = offsets[offset+priority_offset];
        for (int thread = 0; thread < 16; thread++){
            prefetch(&s_base[s_off+p*thread-4], 8);
        }
    }

    vp8_mbloop_filter_vertical_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, filters,
            COLS_LOCATION, s_data);
    
    //YUV planes, then 2 more passes of Y plane
    vp8_loop_filter_vertical_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, filters,
            DC_DIFFS_LOCATION, 1);
    vp8_loop_filter_vertical_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, filters,
            DC_DIFFS_LOCATION, 6);
    vp8_loop_filter_vertical_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, filters,
            DC_DIFFS_LOCATION, 7);

    barrier(CLK_GLOBAL_MEM_FENCE);
    
    //Prefetch horizontal source pixels
#if 0
    for(int plane = 0; plane < 3; plane++){
        int p = pitches[plane];
        int offset = get_global_id(2)*3+plane;
        int s_off = offsets[offset+priority_offset];
        for (int thread = 0; thread < 16; thread++){
            prefetch(&s_base[s_off+thread-4*p], 8*p);
        }
    }
#endif
    
    vp8_mbloop_filter_horizontal_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, 
            filters, s_data);
    
    //YUV planes, then 2 more passes of Y plane
    vp8_loop_filter_horizontal_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, filters,
            DC_DIFFS_LOCATION, 0);
    vp8_loop_filter_horizontal_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, filters,
            DC_DIFFS_LOCATION, 3);
    vp8_loop_filter_horizontal_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, filters,
            DC_DIFFS_LOCATION, 4);
    
}

kernel void vp8_loop_filter_vertical_edges_kernel(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters_in,
    int use_mbflim,
    int filter_type,
    int cur_iter,
    int priority_level,
    global int *block_offsets,
    global int *priority_num_blocks
){
    int block_offset = block_offsets[priority_level];
    local unsigned char s_data[16*8*3];
    int filter_offset = 4*block_offset;
    int priority_offset = 16*block_offset;
    
    global int *filters = &filters_in[filter_offset];

    vp8_mbloop_filter_vertical_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, filters,
            COLS_LOCATION, s_data);
    
    //YUV planes, then 2 more passes of Y plane
    vp8_loop_filter_vertical_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, filters,
            DC_DIFFS_LOCATION, 1);
    vp8_loop_filter_vertical_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, filters,
            DC_DIFFS_LOCATION, 6);
    vp8_loop_filter_vertical_edge_worker(s_base, &offsets[priority_offset], pitches, lfi, filters,
            DC_DIFFS_LOCATION, 7);
    
}

void vp8_loop_filter_simple_horizontal_edge_worker
(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters_in,
    int use_mbflim,
    int filter_type,
    int cur_iter,
    int priority_level,
    global int *block_offsets,
    global int *priority_num_blocks
){
    int block_offset = block_offsets[priority_level];
    int filter_offset = 4*block_offset;
    int priority_offset = 8*block_offset;
    private size_t plane = get_global_id(1);
    private size_t block = get_global_id(2);
    size_t num_planes = get_global_size(1);
    size_t num_blocks = get_global_size(2);

    global int *filters = &filters_in[filter_offset];

    if (filters[num_blocks*filter_type + block] > 0){
        int filter_level = filters[block];
        if (filter_level){
            int p = pitches[plane];
            int block_offset = cur_iter*num_blocks*num_planes + block*num_planes+plane;
            int s_off = offsets[block_offset+priority_offset];


            signed char mask = 0;
            size_t i= get_global_id(0);

            global signed char *limit, *flimit;
            global loop_filter_info *lf_info;

            if (i < threads[plane]){
                lf_info = &lfi[filter_level];
                if (use_mbflim == 0){
                    flimit = lf_info->flim;
                } else {
                    flimit = lf_info->mbflim;
                }

                limit = lf_info->lim;

                s_off += i;
                mask = vp8_simple_filter_mask(limit[i], flimit[i], s_base[s_off-2*p], s_base[s_off-p], s_base[s_off], s_base[s_off+p]);
                vp8_simple_filter(mask, s_base, s_off - 2 * p, s_off - 1 * p, s_off, s_off + 1 * p);
            }
        }
    }
}

void vp8_loop_filter_simple_vertical_edge_worker(
    global unsigned char *s_base,
    global int *offsets, /* Y or YUV offsets for EACH block being processed*/
    global int *pitches, /* 1 or 3 values for Y or YUV pitches*/
    global loop_filter_info *lfi, /* Single struct for the frame */
    global int *filters_in, /* Filters for each block being processed */
    int use_mbflim, /* Use lfi->flim or lfi->mbflim, need once per kernel call */
    int filter_type, /* Should dc_diffs, rows, or cols be used?*/
    int cur_iter,
    int priority_level,
    global int *block_offsets,
    global int *priority_num_blocks
){
    int block_offset = block_offsets[priority_level];
    int filter_offset = 4*block_offset;
    int priority_offset = 8*block_offset;
    private size_t block = get_global_id(2);
    size_t num_blocks = get_global_size(2);

    global int *filters = &filters_in[filter_offset];

    if (filters[filter_type * num_blocks + block] > 0){
        int filter_level = filters[block];
        if (filter_level){
            int p = pitches[0];
            int block_offset = cur_iter*num_blocks + block;
            int s_off = offsets[block_offset+priority_offset];

            signed char mask = 0;
            size_t i= get_global_id(0);

            global signed char *limit, *flimit;
            global loop_filter_info *lf_info;

            if (i < threads[0]){
                lf_info = &lfi[filter_level];
                if (use_mbflim == 0){
                    flimit = lf_info->flim;
                } else {
                    flimit = lf_info->mbflim;
                }

                limit = lf_info->lim;

                s_off += p * i;
                mask = vp8_simple_filter_mask(limit[i], flimit[i], s_base[s_off-2], s_base[s_off-1], s_base[s_off], s_base[s_off+1]);
                vp8_simple_filter(mask, s_base, s_off - 2, s_off - 1, s_off, s_off + 1);
            }
        }
    }
}

kernel void vp8_loop_filter_simple_vertical_edges_kernel
(
    global unsigned char *s_base,
    global int *offsets, /* Y or YUV offsets for EACH block being processed*/
    global int *pitches, /* 1 or 3 values for Y or YUV pitches*/
    global loop_filter_info *lfi, /* Single struct for the frame */
    global int *filters_in, /* Filters for each block being processed */
    int use_mbflim, /* Use lfi->flim or lfi->mbflim, need once per kernel call */
    int filter_type, /* Should dc_diffs, rows, or cols be used?*/
    int cur_iter,
    int priority_level,
    global int *block_offsets,
    global int *priority_num_blocks
){    
    vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
            lfi, filters_in, 1, COLS_LOCATION, 0, priority_level,
            block_offsets, priority_num_blocks
    );

    //3 Y plane iterations
    vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
            lfi, filters_in, 0, DC_DIFFS_LOCATION, 1, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
            lfi, filters_in, 0, DC_DIFFS_LOCATION, 2, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
            lfi, filters_in, 0, DC_DIFFS_LOCATION, 3, priority_level,
            block_offsets, priority_num_blocks
    );
}

kernel void vp8_loop_filter_simple_horizontal_edges_kernel
(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters_in,
    int use_mbflim,
    int filter_type,
    int cur_iter,
    int priority_level,
    global int *block_offsets,
    global int *priority_num_blocks
){
    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, lfi,
            filters_in, 1, ROWS_LOCATION, 4, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, lfi,
            filters_in, 0, DC_DIFFS_LOCATION, 5, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, lfi,
            filters_in, 0, DC_DIFFS_LOCATION, 6, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, lfi,
            filters_in, 0, DC_DIFFS_LOCATION, 7, priority_level,
            block_offsets, priority_num_blocks
    );
}

kernel void vp8_loop_filter_simple_all_edges_kernel
(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters_in,
    int use_mbflim,
    int filter_type,
    int cur_iter,
    int priority_level,
    global int *block_offsets,
    global int *priority_num_blocks
)
{

    vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
            lfi, filters_in, 1, COLS_LOCATION, 0, priority_level,
            block_offsets, priority_num_blocks
    );

    //3 Y plane iterations
    vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
            lfi, filters_in, 0, DC_DIFFS_LOCATION, 1, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
            lfi, filters_in, 0, DC_DIFFS_LOCATION, 2, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
            lfi, filters_in, 0, DC_DIFFS_LOCATION, 3, priority_level,
            block_offsets, priority_num_blocks
    );
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, lfi,
            filters_in, 1, ROWS_LOCATION, 4, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, lfi,
            filters_in, 0, DC_DIFFS_LOCATION, 5, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, lfi,
            filters_in, 0, DC_DIFFS_LOCATION, 6, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, lfi,
            filters_in, 0, DC_DIFFS_LOCATION, 7, priority_level,
            block_offsets, priority_num_blocks
    );
}

//Inline and non-kernel functions follow.

__inline uchar8 vp8_mbfilter(
    signed char mask,
    signed char hev,
    uchar8 base
)
{
    signed char s, u;
    signed char vp8_filter;

    char2 filter;

    //char8 pq = vload8(0, (local char *)base);
    char8 pq = convert_char8(base);
    pq ^= (char8)0x80;
    
    /* add outer taps if we have high edge variance */
    vp8_filter = clamp(pq.s2 - pq.s5, -128, 127);
    vp8_filter = clamp(vp8_filter + 3 * (pq.s4 - pq.s3), -128, 127);
    vp8_filter &= mask;

    filter.s1 = vp8_filter;
    filter.s1 &= hev;
    filter.s0 = filter.s1;

    /* save bottom 3 bits so that we round one side +4 and the other +3 */
    filter.s0 = clamp(filter.s0 + 4, -128, 127);
    filter.s1 = clamp(filter.s1 + 3, -128, 127);
    filter.s0 >>= 3;
    filter.s1 >>= 3;

    pq.s4 = clamp(pq.s4 - filter.s0, -128, 127);
    pq.s3 = clamp(pq.s3 + filter.s1, -128, 127);

    /* only apply wider filter if not high edge variance */
    vp8_filter &= ~hev;
    filter.s1 = vp8_filter;

    /* roughly 3/7th difference across boundary */
    u = clamp((63 + filter.s1 * 27) >> 7, -128, 127);
    s = clamp(pq.s4 - u, -128, 127);
    pq.s4 = s ^ 0x80;
    s = clamp(pq.s3 + u, -128, 127);
    pq.s3 = s ^ 0x80;

    /* roughly 2/7th difference across boundary */
    u = clamp((63 + filter.s1 * 18) >> 7, -128, 127);
    s = clamp(pq.s5 - u, -128, 127);
    pq.s5 = s ^ 0x80;
    s = clamp(pq.s2 + u, -128, 127);
    pq.s2 = s ^ 0x80;

    /* roughly 1/7th difference across boundary */
    u = clamp((63 + filter.s1 * 9) >> 7, -128, 127);
    s = clamp(pq.s6 - u, -128, 127);
    pq.s6 = s ^ 0x80;
    s = clamp(pq.s1 + u, -128, 127);
    pq.s1 = s ^ 0x80;
    
    return convert_uchar8(pq);
}

/* is there high variance internal edge ( 11111111 yes, 00000000 no) */
__inline signed char vp8_hevmask(signed char thresh, uchar4 pq)
{
    signed char hev = 0;
    hev  |= (abs(pq.s0 - pq.s1) > thresh) * -1;
    hev  |= (abs(pq.s3 - pq.s2) > thresh) * -1;
    return hev;
}


/* should we apply any filter at all ( 11111111 yes, 00000000 no) */
__inline signed char vp8_filter_mask( signed char limit, signed char flimit,
        uchar8 pq)
{
    signed char mask = 0;

#if 1
    mask |= (abs(pq.s0 - pq.s1) > limit) * -1;
    mask |= (abs(pq.s1 - pq.s2) > limit) * -1;
    mask |= (abs(pq.s2 - pq.s3) > limit) * -1;
    mask |= (abs(pq.s5 - pq.s4) > limit) * -1;
    mask |= (abs(pq.s6 - pq.s5) > limit) * -1;
    mask |= (abs(pq.s7 - pq.s6) > limit) * -1;
    mask |= (abs(pq.s3 - pq.s4) * 2 + abs(pq.s2 - pq.s5) / 2  > flimit * 2 + limit) * -1;
    mask = ~mask;
    return mask;
#else
    //Only apply the filter if the difference is LESS than 'limit'
    mask |= (abs(pq.s0 - pq.s1) > limit);
    mask |= (abs(pq.s1 - pq.s2) > limit);
    mask |= (abs(pq.s2 - pq.s3) > limit);
    mask |= (abs(pq.s5 - pq.s4) > limit);
    mask |= (abs(pq.s6 - pq.s5) > limit);
    mask |= (abs(pq.s7 - pq.s6) > limit);
    mask |= (abs(pq.s3 - pq.s4) * 2 + abs(pq.s2 - pq.s5) / 2  > flimit * 2 + limit);
    
    return (mask != 0 ? 0 : -1);
#endif
    
}

/* should we apply any filter at all ( 11111111 yes, 00000000 no) */
__inline signed char vp8_simple_filter_mask(
    signed char limit,
    signed char flimit,
    uc p1,
    uc p0,
    uc q0,
    uc q1
)
{
    signed char mask = (abs(p0 - q0) * 2 + abs(p1 - q1) / 2  <= flimit * 2 + limit) * -1;
    return mask;
}

void vp8_simple_filter(
    signed char mask,
    global uc *base,
    int op1_off,
    int op0_off,
    int oq0_off,
    int oq1_off
)
{

    global uc *op1 = base + op1_off;
    global uc *op0 = base + op0_off;
    global uc *oq0 = base + oq0_off;
    global uc *oq1 = base + oq1_off;

    signed char vp8_filter, Filter1, Filter2;
    signed char p1 = (signed char) * op1 ^ 0x80;
    signed char p0 = (signed char) * op0 ^ 0x80;
    signed char q0 = (signed char) * oq0 ^ 0x80;
    signed char q1 = (signed char) * oq1 ^ 0x80;
    signed char u;

    vp8_filter = clamp(p1 - q1, -128, 127);
    vp8_filter = clamp(vp8_filter + 3 * (q0 - p0), -128, 127);
    vp8_filter &= mask;

    /* save bottom 3 bits so that we round one side +4 and the other +3 */
    Filter1 = clamp(vp8_filter + 4, -128, 127);
    Filter1 >>= 3;
    u = clamp(q0 - Filter1, -128, 127);
    *oq0  = u ^ 0x80;

    Filter2 = clamp(vp8_filter + 3, -128, 127);
    Filter2 >>= 3;
    u = clamp(p0 + Filter2, -128, 127);
    *op0 = u ^ 0x80;
}
