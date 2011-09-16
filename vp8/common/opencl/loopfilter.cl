#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

typedef unsigned char uc;
typedef signed char sc;

__inline signed char vp8_filter_mask(sc, sc, uchar8);
__inline signed char vp8_simple_filter_mask(signed char, signed char, uc, uc, uc, uc);
__inline signed char vp8_hevmask(signed char, uchar4);

__inline uchar8 vp8_mbfilter(signed char mask,signed char hev,uchar8);

__inline void vp8_simple_filter(signed char mask,global uc *base, int op1_off,int op0_off,int oq0_off,int oq1_off);

constant int threads[3] = {16, 8, 8};

#ifndef __CL_VERSION_1_0__
#define __CL_VERSION_1_0__ 100
#endif 

#if __OPENCL_VERSION__ == __CL_VERSION_1_0__
#define clamp(x,y,z) vp8_char_clamp(x)
char vp8_char_clamp(int in){
    return max(min(in, 127), -128);
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

//OpenCL built-in functions (
size_t get_global_id(unsigned int);
size_t get_global_size(unsigned int);

__inline uchar4 vp8_filter(
    signed char mask,
    signed char hev,
    uchar4 base
)
{
    char4 pq = as_char4(base) ^ (char4)0x80;

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

    u.s0 = add_sat(pq.s0, vp8_filter);
    u.s1 = add_sat(pq.s1, Filter.s1);
    u.s2 = sub_sat(pq.s2, Filter.s0);
    u.s3 = sub_sat(pq.s3, vp8_filter);

    return as_uchar4(u ^ (char4)0x80);

}

// Filters horizontal edges of inner blocks in a Macroblock
__inline void vp8_loop_filter_horizontal_edge_worker(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches, /* pitch */
    global loop_filter_info *lfi,
    global int *filters,
    int filter_type,
    int filter_level,
    int cur_iter,
    size_t num_blocks,
    size_t num_planes,
    size_t plane,
    size_t block
){
    size_t thread = get_global_id(0);
    
    if (filters[num_blocks*filter_type + block] > 0){
        int p = pitches[plane];
        int block_offset = num_blocks*11 + cur_iter*num_blocks*num_planes + block*num_planes+plane;
        int s_off = offsets[block_offset];

        s_off += thread;
        //s_off += thread, so maybe it's possible to use a strided copy with local memory to load into the vector

        uchar8 data;
        data.s0 = s_base[s_off-4*p];
        data.s1 = s_base[s_off-3*p];
        data.s2 = s_base[s_off-2*p];
        data.s3 = s_base[s_off-p];
        data.s4 = s_base[s_off];
        data.s5 = s_base[s_off+p];
        data.s6 = s_base[s_off+2*p];
        data.s7 = s_base[s_off+3*p];

        char mask = vp8_filter_mask(lfi->lim[thread], lfi->flim[thread], data);

        int hev = vp8_hevmask(lfi->thr[thread], data.s2345);

        data.s2345 = vp8_filter(mask, hev, data.s2345);

        s_base[s_off - 2*p] = data.s2;
        s_base[s_off - p  ] = data.s3;
        s_base[s_off      ] = data.s4;
        s_base[s_off + p  ] = data.s5;
    }
}

__inline void vp8_loop_filter_vertical_edge_worker(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters,
    int filter_type,
    int filter_level,
    int cur_iter,
    size_t num_blocks,
    size_t num_planes,
    size_t plane,
    size_t block
){
    if (filters[num_blocks*filter_type + block] > 0){
        int p = pitches[plane];
        int block_offset = cur_iter*num_blocks*num_planes + block*num_planes+plane;
        int s_off = offsets[block_offset];

        size_t thread = get_global_id(0);
        s_off += p * thread;

        uchar8 data = vload8(0, &s_base[s_off-4]);

        char mask = vp8_filter_mask(lfi->lim[thread], lfi->flim[thread], data);

        int hev = vp8_hevmask(lfi->thr[thread], data.s2345);

        data.s2345 = vp8_filter(mask, hev, data.s2345);

        vstore4(data.s2345, 0, &s_base[s_off-2]);
    }
}

__inline void vp8_mbloop_filter_horizontal_edge_worker(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters,
    int filter_type,
    int filter_level,
    size_t num_blocks,
    size_t num_planes,
    size_t plane,
    size_t block
){
    
    int p = pitches[plane];
    size_t thread = get_global_id(0);
    int s_off = offsets[8*num_blocks + block*num_planes+plane] + thread;

    uchar8 data;
    data.s0 = s_base[s_off-4*p];
    data.s1 = s_base[s_off-3*p];
    data.s2 = s_base[s_off-2*p];
    data.s3 = s_base[s_off-p];
    data.s4 = s_base[s_off];
    data.s5 = s_base[s_off+p];
    data.s6 = s_base[s_off+2*p];
    data.s7 = s_base[s_off+3*p];

    char mask = vp8_filter_mask(lfi->lim[thread], lfi->mbflim[thread], data);

    char hev = vp8_hevmask(lfi->thr[thread], data.s2345);

    data = vp8_mbfilter(mask, hev, data);

    s_base[s_off - 3*p] = data.s1;
    s_base[s_off - 2*p] = data.s2;
    s_base[s_off - 1*p] = data.s3;
    s_base[s_off      ] = data.s4;
    s_base[s_off + p  ] = data.s5;
    s_base[s_off + 2*p] = data.s6;

}

__inline void vp8_mbloop_filter_vertical_edge_worker(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters,
    int filter_type,
    int filter_level,
    size_t num_blocks,
    size_t num_planes,
    size_t plane,
    size_t block
){

    int p = pitches[plane];
    int block_offset = block*num_planes+plane;
    int s_off = offsets[block_offset];
    size_t thread = get_global_id(0);
    
    s_off += p * thread;

    uchar8 data = vload8(0, &s_base[s_off-4]);

    char mask = vp8_filter_mask(lfi->lim[thread], lfi->mbflim[thread], data);

    int hev = vp8_hevmask(lfi->thr[thread], data.s2345);

    data = vp8_mbfilter(mask, hev, data);

    vstore8(data, 0, &s_base[s_off-4]);
}

kernel void vp8_loop_filter_all_edges_kernel(
    global unsigned char *s_base,
    global int *offsets_in,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters_in,
    int priority_level,
    int num_levels,
    global int *block_offsets,
    global int *priority_num_blocks
){
    size_t thread = get_global_id(0);
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);

    for (int i = 0; i < num_levels; i++){
        int block_offset = block_offsets[priority_level];

        global int *offsets = &offsets_in[16*block_offset];
        global int *filters = &filters_in[4*block_offset];
        size_t num_blocks = priority_num_blocks[priority_level];
        int filter_level = filters[block];
        global loop_filter_info *lf_info = &lfi[filter_level];    
        int thread_level_filter = (thread<threads[plane]) & (filter_level!=0);

        if (block < num_blocks){
            //Prefetch vertical edge source pixels into global cache (horizontal isn't worth it)
            for(int pln = 0; pln < 3; pln++){
                int p = pitches[pln];
                int offset = block*3+pln;
                int s_off = offsets[offset] - 4;
                for (int thread = 0; thread < 16; thread++){
                    prefetch(&s_base[s_off+p*thread], 8);
                }
            }

            if (thread_level_filter & (filters[num_blocks*COLS_LOCATION + block] > 0)){
                vp8_mbloop_filter_vertical_edge_worker(s_base, offsets, pitches, lf_info, filters,
                        COLS_LOCATION, filter_level, num_blocks, 3,  plane, block);
            }

            //YUV planes, then 2 more passes of Y plane
            if (thread_level_filter){
                vp8_loop_filter_vertical_edge_worker(s_base, offsets, pitches, lf_info, filters,
                        DC_DIFFS_LOCATION, filter_level, 1, num_blocks, 3, plane, block);
                if (plane == 0){
                    vp8_loop_filter_vertical_edge_worker(s_base, offsets, pitches, lf_info, filters,
                            DC_DIFFS_LOCATION, filter_level, 6, num_blocks, 1, plane, block);
                    vp8_loop_filter_vertical_edge_worker(s_base, offsets, pitches, lf_info, filters,
                            DC_DIFFS_LOCATION, filter_level, 7, num_blocks, 1,  plane, block);
                }
            }
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (block < num_blocks){

            if (thread_level_filter & (filters[num_blocks*ROWS_LOCATION + block] > 0)){
                vp8_mbloop_filter_horizontal_edge_worker(s_base, offsets, pitches, lf_info, 
                    filters, ROWS_LOCATION, filter_level, num_blocks, 3, plane, block);
            }

            //YUV planes, then 2 more passes of Y plane
            if (thread_level_filter){
                vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, lf_info, filters,
                        DC_DIFFS_LOCATION, filter_level, 0, num_blocks, 3, plane, block);
                if (plane == 0){
                    vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, lf_info, filters,
                            DC_DIFFS_LOCATION, filter_level, 3, num_blocks, 1, plane, block);
                    vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, lf_info, filters,
                            DC_DIFFS_LOCATION, filter_level, 4, num_blocks, 1, plane, block);
                }
            }
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
        priority_level++;
    }
    
}

kernel void vp8_loop_filter_horizontal_edges_kernel(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches, /* pitch */
    global loop_filter_info *lfi,
    global int *filters,
    int priority_level,
    int num_levels,
    global int *block_offsets,
    global int *priority_num_blocks
){
    size_t thread = get_global_id(0);
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);
    size_t num_blocks = get_global_size(2);
    
    int block_offset = block_offsets[priority_level];

    filters = &filters[4*block_offset];
    offsets = &offsets[16*block_offset];
    int filter_level = filters[block];
    global loop_filter_info *lf_info = &lfi[filter_level];
    
    int thread_level_filter = (thread<threads[plane]) & (filter_level!=0);
    int do_filter = filters[num_blocks*ROWS_LOCATION + block] > 0;
    do_filter &= thread_level_filter;
    if (do_filter){
        vp8_mbloop_filter_horizontal_edge_worker(s_base, offsets, pitches, lf_info, 
            filters, ROWS_LOCATION, filter_level, num_blocks, 3, plane, block);
    }
    
    //YUV planes, then 2 more passes of Y plane
    if (thread_level_filter){
        vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, lf_info, filters,
                DC_DIFFS_LOCATION, filter_level, 0, num_blocks, 3, plane, block);
        if (plane == 0){
            vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 3, num_blocks, 1, plane, block);
            vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 4, num_blocks, 1, plane, block);
        }
    }
}

kernel void vp8_loop_filter_vertical_edges_kernel(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters,
    int priority_level,
    int num_levels,
    global int *block_offsets,
    global int *priority_num_blocks
){
    size_t thread = get_global_id(0);
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);
    size_t num_blocks = get_global_size(2);
    
    int block_offset = block_offsets[priority_level];
    
    filters = &filters[4*block_offset];
    offsets = &offsets[16*block_offset];
    int filter_level = filters[block];
    global loop_filter_info *lf_info = &lfi[filter_level];

    int thread_level_filter = (thread<threads[plane]) & (filter_level!=0);
    int do_filter = filters[num_blocks*COLS_LOCATION + block] > 0;
    do_filter &= thread_level_filter;
    if (do_filter){
        vp8_mbloop_filter_vertical_edge_worker(s_base, offsets, pitches, lf_info, filters,
            COLS_LOCATION, filter_level, num_blocks, 3, plane, block);
    }
    
    //YUV planes, then 2 more passes of Y plane
    if (thread_level_filter){
        vp8_loop_filter_vertical_edge_worker(s_base, offsets, pitches, lf_info, filters,
                DC_DIFFS_LOCATION, filter_level, 1, num_blocks, 3, plane, block);
        if (plane == 0){
            vp8_loop_filter_vertical_edge_worker(s_base, offsets, pitches, lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 6, num_blocks, 1, plane, block);
            vp8_loop_filter_vertical_edge_worker(s_base, offsets, pitches, lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 7, num_blocks, 1, plane, block);
        }
    }
    
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
    size_t num_blocks = priority_num_blocks[priority_level];

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
    size_t num_blocks = priority_num_blocks[priority_level];

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
    int filter_type, /* Should dc_diffs, rows, or cols be used?*/
    int priority_level,
    int num_levels,
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
    int priority_level,
    int num_levels,
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
    int priority_level,
    int num_levels,
    global int *block_offsets,
    global int *priority_num_blocks
)
{
    
    for (int i = 0; i < num_levels; i++){
        if (get_global_id(2) < priority_num_blocks[priority_level]){
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

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (get_global_id(2) < priority_num_blocks[priority_level]){
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
        
        barrier(CLK_GLOBAL_MEM_FENCE);
        priority_level++;
    }
}

//Inline and non-kernel functions follow.
__inline uchar8 vp8_mbfilter(
    signed char mask,
    signed char hev,
    uchar8 base
)
{
    char4 u;

    char8 pq = as_char8(base);
    pq ^= (char8){0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0};
    
    /* add outer taps if we have high edge variance */
    char vp8_filter = sub_sat(pq.s2, pq.s5);
    vp8_filter = clamp(vp8_filter + 3 * (pq.s4 - pq.s3), -128, 127);
    vp8_filter &= mask;

    char2 filter = (char2)vp8_filter;
    filter &= hev;

    /* save bottom 3 bits so that we round one side +4 and the other +3 */
    filter = add_sat(filter, (char2){4,3});
    filter.s0 >>= 3;
    filter.s1 >>= 3;
    
    pq.s4 = sub_sat(pq.s4, filter.s0);
    pq.s3 = add_sat(pq.s3, filter.s1);

    /* only apply wider filter if not high edge variance */
    vp8_filter &= ~hev;
    filter.s1 = vp8_filter;

    /* roughly 3/7th, 2/7th, and 1/7th difference across boundary */
#if 1
    u = convert_char4(clamp(((short4)63 + (short4)filter.s1 * (short4){27, 18, 9, 0}) >> 7, -128, 127) );
#else
    u.s0 = clamp((63 + filter.s1 * 27) >> 7, -128, 127);
    u.s1 = clamp((63 + filter.s1 * 18) >> 7, -128, 127);
    u.s2 = clamp((63 + filter.s1 * 9) >> 7, -128, 127);
    u.s3 = 0;
#endif
    
#if 0
    char4 s;
    s = sub_sat(pq.s4567, u);
    pq.s4567 = s ^ (char4){0x80, 0x80, 0x80, 0};
    s.s3210 = add_sat(pq.s3210, u);
    pq.s0123 = s ^ (char4){0, 0x80, 0x80, 0x80};
#else
    pq.s4567 = sub_sat(pq.s4567, u.s0123);
    pq.s3210 = add_sat(pq.s3210, u.s0123);
    pq ^= (char8){0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0};
#endif
    
    return as_uchar8(pq);
}

/* is there high variance internal edge ( 11111111 yes, 00000000 no) */
__inline signed char vp8_hevmask(signed char thresh, uchar4 pq)
{
    return any(abs_diff(pq.s03, pq.s12) > thresh) * -1;
}


/* should we apply any filter at all ( 11111111 yes, 00000000 no) */
__inline signed char vp8_filter_mask( signed char limit, signed char flimit,
        uchar8 pq)
{
    //Only apply the filter if the difference is LESS than 'limit'
#if 1
    char mask = (abs_diff(pq.s0, pq.s1) > limit);
    mask |= (abs_diff(pq.s1, pq.s2) > limit);
    mask |= (abs_diff(pq.s2, pq.s3) > limit);
    mask |= (abs_diff(pq.s5, pq.s4) > limit);
    mask |= (abs_diff(pq.s6, pq.s5) > limit);
    mask |= (abs_diff(pq.s7, pq.s6) > limit);
    mask |= (abs_diff(pq.s3, pq.s4) * 2 + abs_diff(pq.s2, pq.s5) / 2  > flimit * 2 + limit);
    return mask - 1;
#else
    short8 mask8 = convert_short8(abs_diff(pq.s01256732, pq.s12345645));
    mask8.s6 = mask8.s6 * 2 + mask8.s7 / 2;
    
    short8 limits = (short8)limit;
    limits.s6 += flimit * 2;
    mask8 = mask8 > limits;
    mask8.s7 = 0;
    
    return any(mask8) - 1;
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
    signed char mask = (abs_diff(p0, q0) * 2 + abs_diff(p1, q1) / 2  <= flimit * 2 + limit) * -1;
    return mask;
}

__inline void vp8_simple_filter(
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

    signed char vp8_filter;
    char2 filter;
    
    char4 pq = (char4){*op1, *op0, *oq0, *oq1};
    pq ^= (char4)0x80;

    signed char u;

    vp8_filter = sub_sat(pq.s0, pq.s3);
    vp8_filter = clamp(vp8_filter + 3 * (pq.s2 - pq.s1), -128, 127);
    vp8_filter &= mask;

    /* save bottom 3 bits so that we round one side +4 and the other +3 */
    filter = add_sat((char2)vp8_filter, (char2){4,3});
    filter.s0 >>= 3;
    filter.s1 >>= 3;

    u = sub_sat(pq.s2, filter.s0);
    *oq0  = u ^ 0x80;

    u = add_sat(pq.s1, filter.s1);
    *op0 = u ^ 0x80;
}
