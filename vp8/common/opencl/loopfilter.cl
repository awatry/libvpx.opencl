#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

typedef unsigned char uc;
typedef signed char sc;

__inline signed char vp8_filter_mask_mem(uc, uc, uchar*);
__inline signed char vp8_filter_mask(uc, uc, uchar8);

__inline char vp8_hevmask_mem(uchar, uchar*);
__inline char vp8_hevmask(uchar, uchar4);

__inline void vp8_filter_mem( signed char mask, uchar hev, uchar *base);

__inline signed char vp8_simple_filter_mask(uc, uc, uc, uc, uc);

__inline uchar8 vp8_mbfilter(signed char mask, uchar hev, uchar8);

__inline void vp8_simple_filter(signed char mask,global uc *base, int op1_off,int op0_off,int oq0_off,int oq1_off);

constant size_t threads[3] = {16, 8, 8};

#ifndef __CL_VERSION_1_0__
#define __CL_VERSION_1_0__ 100
#endif 

#if !defined(__OPENCL_VERSION__) || (__OPENCL_VERSION__ == __CL_VERSION_1_0__)
#define clamp(x,y,z) vp8_char_clamp(x)
char vp8_char_clamp(int in){
    return max(min(in, 127), -128);
}
#endif

typedef struct
{
    unsigned char mblim[MAX_LOOP_FILTER + 1][SIMD_WIDTH];
    unsigned char blim[MAX_LOOP_FILTER + 1][SIMD_WIDTH];
    unsigned char lim[MAX_LOOP_FILTER + 1][SIMD_WIDTH];
    unsigned char hev_thr[4][SIMD_WIDTH];
    unsigned char lvl[4][4][4];
    unsigned char hev_thr_lut[2][MAX_LOOP_FILTER + 1];
    unsigned char mode_lf_lut[10];
} loop_filter_info_n __attribute__ ((aligned(SIMD_WIDTH)));

typedef struct
{
    global unsigned char *mblim;
    global unsigned char *blim;
    global unsigned char *lim;
    global unsigned char *hev_thr;
} loop_filter_info;

__inline void vp8_filter_mem(
    signed char mask,
    uchar hev,
    uchar *base
)
{
    
    char *pq = (char*)base;
    pq[0] ^= 0x80;
    pq[1] ^= 0x80;
    pq[2] ^= 0x80;
    pq[3] ^= 0x80;

    char vp8_filter;
    char2 Filter;
    char4 u;

    /* add outer taps if we have high edge variance */
    vp8_filter = sub_sat(pq[0], pq[3]);
    vp8_filter &= hev;

    /* inner taps */
    vp8_filter = clamp(vp8_filter + 3 * (pq[2] - pq[1]), -128, 127);
    vp8_filter &= mask;

    /* save bottom 3 bits so that we round one side +4 and the other +3
     * if it equals 4 we'll set to adjust by -1 to account for the fact
     * we'd round 3 the other way
     */
    char2 rounding = {4,3};
    Filter = add_sat((char2)vp8_filter, rounding);
    Filter.s0 >>= 3;
    Filter.s1 >>= 3;
    
    /* outer tap adjustments */
    vp8_filter = Filter.s0 + 1;
    vp8_filter >>= 1;
    vp8_filter &= ~hev;

    u.s0 = add_sat(pq[0], vp8_filter);
    u.s1 = add_sat(pq[1], Filter.s1);
    u.s2 = sub_sat(pq[2], Filter.s0);
    u.s3 = sub_sat(pq[3], vp8_filter);

    pq[0] = u.s0 ^ 0x80;
    pq[1] = u.s1 ^ 0x80;
    pq[2] = u.s2 ^ 0x80;
    pq[3] = u.s3 ^ 0x80;
    
}

__inline uchar4 vp8_filter(
    signed char mask,
    uchar hev,
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
    char2 rounding = {4,3};
    Filter = add_sat((char2)vp8_filter, rounding);
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

__inline uchar8 load8(global unsigned char *s_base, int s_off, int p){
    uchar8 data;
    data.s0 = s_base[s_off-4*p];
    data.s1 = s_base[s_off-3*p];
    data.s2 = s_base[s_off-2*p];
    data.s3 = s_base[s_off-p];
    data.s4 = s_base[s_off];
    data.s5 = s_base[s_off+p];
    data.s6 = s_base[s_off+2*p];
    data.s7 = s_base[s_off+3*p];
    return data;
}

__inline void save4(global unsigned char *s_base, int s_off, int p, uchar8 data){
    s_base[s_off - 2*p] = data.s2;
    s_base[s_off - p  ] = data.s3;
    s_base[s_off      ] = data.s4;
    s_base[s_off + p  ] = data.s5;
}

__inline void save6(global unsigned char *s_base, int s_off, int p, uchar8 data){
    s_base[s_off - 3*p] = data.s1;
    s_base[s_off - 2*p] = data.s2;
    s_base[s_off - p  ] = data.s3;
    s_base[s_off      ] = data.s4;
    s_base[s_off + p  ] = data.s5;
    s_base[s_off + 2*p  ] = data.s6;
}

__inline uchar16 load16(global unsigned char *s_base, int s_off, int p){
    uchar16 data;
    data.s01234567 = load8(s_base, s_off, p);
    data.s89abcdef = load8(s_base, s_off+8*p, p);
    return data;
}

__inline void private_load16(private uchar *dest, global unsigned char *s_base, int s_off, int p){
    dest[0] = s_base[s_off-4*p];
    dest[1] = s_base[s_off-3*p];
    dest[2] = s_base[s_off-2*p];
    dest[3] = s_base[s_off-1*p];
    dest[4] = s_base[s_off-0*p];
    dest[5] = s_base[s_off+1*p];
    dest[6] = s_base[s_off+2*p];
    dest[7] = s_base[s_off+3*p];
    dest[8] = s_base[s_off+4*p];
    dest[9] = s_base[s_off+5*p];
    dest[10] = s_base[s_off+6*p];
    dest[11] = s_base[s_off+7*p];
    dest[12] = s_base[s_off+8*p];
    dest[13] = s_base[s_off+9*p];
    dest[14] = s_base[s_off+10*p];
    dest[15] = s_base[s_off+11*p];
}

__inline void private_save12(global uchar *s_base, int s_off, int p, private uchar *data){
    s_base[s_off - 2*p] = data[2];
    s_base[s_off - p  ] = data[3];
    s_base[s_off      ] = data[4];
    s_base[s_off + p  ] = data[5];
    s_base[s_off + 2*p] = data[6];
    s_base[s_off + 3*p] = data[7];
    s_base[s_off + 4*p] = data[8];
    s_base[s_off + 5*p] = data[9];
    s_base[s_off + 6*p] = data[10];
    s_base[s_off + 7*p] = data[11];
    s_base[s_off + 8*p] = data[12];
    s_base[s_off + 9*p] = data[13];
}

__inline void save12(global unsigned char *s_base, int s_off, int p, uchar16 data){
    s_base[s_off - 2*p] = data.s2;
    s_base[s_off - p  ] = data.s3;
    s_base[s_off      ] = data.s4;
    s_base[s_off + p  ] = data.s5;
    s_base[s_off + 2*p] = data.s6;
    s_base[s_off + 3*p] = data.s7;
    s_base[s_off + 4*p] = data.s8;
    s_base[s_off + 5*p] = data.s9;
    s_base[s_off + 6*p] = data.sa;
    s_base[s_off + 7*p] = data.sb;
    s_base[s_off + 8*p] = data.sc;
    s_base[s_off + 9*p] = data.sd;
}

// Filters horizontal edges of inner blocks in a Macroblock
__inline void vp8_loop_filter_horizontal_edge_worker(
    global uchar *s_base,
    global int *offsets,
    global int *pitches, /* pitch */
    loop_filter_info *lfi,
    global int *filters,
    int filter_type,
    int filter_level,
    int cur_iter,
    size_t num_blocks,
    size_t num_planes,
    size_t plane,
    size_t block,
    int p //pitches[plane]
){
    size_t thread = get_global_id(0);
    uchar8 data;
    if (filters[num_blocks*filter_type + block] > 0){
        int block_offset = num_blocks*11 + cur_iter*num_blocks*num_planes + block*num_planes+plane;
        int s_off = offsets[block_offset];

        s_off += thread;
        //s_off += thread, so maybe it's possible to use a strided copy with local memory to load into the vector

        data = load8(s_base, s_off, p);

        char mask = vp8_filter_mask(lfi->lim[thread], lfi->blim[thread], data);

        char hev = vp8_hevmask(lfi->hev_thr[thread], data.s2345);

        //if (cur_iter == 3 && thread == 3 && plane == 0){
        //    printf("block = %d, thread = %d, plane = %d, hev = %d, mask = %d\n", block, thread, plane, hev, mask);
        //}
        
        data.s2345 = vp8_filter(mask, hev, data.s2345);

        save4(s_base, s_off, p, data);
    }
}

__inline void vp8_loop_filter_vertical_edge_worker(
    private uchar *data,
    global int *offsets,
    global int *pitches,
    loop_filter_info *lfi,
    global int *filters,
    int filter_type,
    int filter_level,
    int cur_iter,
    size_t num_blocks,
    size_t num_planes,
    size_t plane,
    size_t block,
    int p
){
    if (filters[num_blocks*filter_type + block] > 0){
        size_t thread = get_global_id(0);

        char mask = vp8_filter_mask_mem(lfi->lim[thread], lfi->blim[thread], data);

        int hev = vp8_hevmask_mem(lfi->hev_thr[thread], &data[2]);
        
        vp8_filter_mem(mask, hev, &data[2]);
    }
}

__inline void vp8_mbloop_filter_horizontal_edge_worker(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    loop_filter_info *lfi,
    global int *filters,
    int filter_type,
    int filter_level,
    size_t num_blocks,
    size_t num_planes,
    size_t plane,
    size_t block,
    int p //pitches[plane]
){
    
    size_t thread = get_global_id(0);
    int s_off = offsets[8*num_blocks + block*num_planes+plane] + thread;

    uchar8 data = load8(s_base, s_off, p);

    char mask = vp8_filter_mask(lfi->lim[thread], lfi->mblim[thread], data);

    char hev = vp8_hevmask(lfi->hev_thr[thread], data.s2345);

    data = vp8_mbfilter(mask, hev, data);

    save6(s_base, s_off, p, data);

}

__inline void vp8_mbloop_filter_vertical_edge_worker(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    loop_filter_info *lfi,
    global int *filters,
    int filter_type,
    int filter_level,
    size_t plane,
    size_t block,
    int p //pitches[plane]
){

    int block_offset = block*3+plane;
    size_t thread = get_global_id(0);
    int s_off = offsets[block_offset] + p*thread;

    uchar8 data = load8(s_base, s_off, 1);

    char mask = vp8_filter_mask(lfi->lim[thread], lfi->mblim[thread], data);

    int hev = vp8_hevmask(lfi->hev_thr[thread], data.s2345);

    data = vp8_mbfilter(mask, hev, data);

    save6(s_base, s_off, 1, data);
}

__inline void set_lfi(global loop_filter_info_n *lfi_n, loop_filter_info *lfi, int frame_type, int filter_level){
    int hev_index = lfi_n->hev_thr_lut[frame_type][filter_level];
    lfi->mblim = lfi_n->mblim[filter_level];
    lfi->blim = lfi_n->blim[filter_level];
    lfi->lim = lfi_n->lim[filter_level];
    lfi->hev_thr = lfi_n->hev_thr[hev_index];
}

kernel void vp8_loop_filter_all_edges_kernel(
    global unsigned char *s_base,
    global int *offsets_in,
    global int *pitches,
    global loop_filter_info_n *lfi_n,
    global int *filters_in,
    int priority_level,
    int num_levels,
    global int *block_offsets,
    global int *priority_num_blocks,
    int frame_type
){
    size_t thread = get_global_id(0);
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);

    //local char local_data[256]; //Local copy of frame data.
#if VP8_LOOP_FILTER_MULTI_LEVEL
    int i;
    for (i = 0; i < num_levels; i++){
#endif
    int block_offset = block_offsets[priority_level];

    global int *offsets = &offsets_in[16*block_offset];
    global int *filters = &filters_in[4*block_offset];
    size_t num_blocks = priority_num_blocks[priority_level];
    int filter_level = filters[block];
    loop_filter_info lf_info;
    
    int p = pitches[plane];
    int thread_level_filter = (thread<threads[plane]) & (filter_level!=0);

    set_lfi(lfi_n, &lf_info, frame_type, filter_level);

#if VP8_LOOP_FILTER_MULTI_LEVEL
    if (block < priority_num_blocks[priority_level]){
#endif
        //Prefetch vertical edge source pixels into global cache (horizontal isn't worth it)
        for(int pln = 0; pln < 3; pln++){
            int pitch = pitches[pln];
            int offset = block*3+pln;
            int s_off = offsets[offset] - 4;
            for (int thread = 0; thread < 16; thread++){
                prefetch(&s_base[s_off+pitch*thread], 8);
            }
        }

        if (thread_level_filter){
            if ( filters[num_blocks*COLS_LOCATION + block] > 0 ){
                vp8_mbloop_filter_vertical_edge_worker(s_base, offsets, pitches, &lf_info, filters,
                        COLS_LOCATION, filter_level, plane, block, p);
            }

            block_offset = num_blocks*3 + block*3 + plane;
            int s_off = offsets[block_offset] + p * thread;
            private uchar data[16];
            private_load16(data, s_base, s_off, 1);

            //YUV planes, then 2 more passes of Y plane
            vp8_loop_filter_vertical_edge_worker(data, offsets, pitches, &lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 1, num_blocks, 3, plane, block, p);
            if (plane == 0){
                vp8_loop_filter_vertical_edge_worker(&data[4], offsets, pitches, &lf_info, filters,
                        DC_DIFFS_LOCATION, filter_level, 6, num_blocks, 1, plane, block, p);
                vp8_loop_filter_vertical_edge_worker(&data[8], offsets, pitches, &lf_info, filters,
                        DC_DIFFS_LOCATION, filter_level, 7, num_blocks, 1,  plane, block, p);
            }
            
            private_save12(s_base, s_off, 1, data);
        }
#if VP8_LOOP_FILTER_MULTI_LEVEL
    }
#endif

    write_mem_fence(CLK_GLOBAL_MEM_FENCE);

#if VP8_LOOP_FILTER_MULTI_LEVEL
    if (block < priority_num_blocks[priority_level]){
#endif
        if (thread_level_filter){
            if (filters[num_blocks*ROWS_LOCATION + block] > 0){
                vp8_mbloop_filter_horizontal_edge_worker(s_base, offsets, pitches, &lf_info, 
                    filters, ROWS_LOCATION, filter_level, num_blocks, 3, plane, block, p);
            }

            //YUV planes, then 2 more passes of Y plane
            vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, &lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 0, num_blocks, 3, plane, block, p);
            if (plane == 0){
                vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, &lf_info, filters,
                        DC_DIFFS_LOCATION, filter_level, 3, num_blocks, 1, plane, block, p);
                vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, &lf_info, filters,
                        DC_DIFFS_LOCATION, filter_level, 4, num_blocks, 1, plane, block, p);
            }
        }
#if VP8_LOOP_FILTER_MULTI_LEVEL
    }
    
    priority_level++;
    barrier(CLK_GLOBAL_MEM_FENCE);
    } //For
#endif

}

kernel void vp8_loop_filter_horizontal_edges_kernel(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches, /* pitch */
    global loop_filter_info_n *lfi_n,
    global int *filters,
    int priority_level,
    int num_levels,
    global int *block_offsets,
    global int *priority_num_blocks,
    int frame_type
){
    size_t thread = get_global_id(0);
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);
    size_t num_blocks = get_global_size(2);
    
    int block_offset = block_offsets[priority_level];
    filters = &filters[4*block_offset];
    offsets = &offsets[16*block_offset];
    int filter_level = filters[block];
    int p = pitches[plane];
    
    loop_filter_info lf_info;
    set_lfi(lfi_n, &lf_info, frame_type, filter_level);

    int thread_level_filter = (thread<threads[plane]) & (filter_level!=0);
    int do_filter = filters[num_blocks*ROWS_LOCATION + block] > 0;
    do_filter &= thread_level_filter;
    if (do_filter){
        vp8_mbloop_filter_horizontal_edge_worker(s_base, offsets, pitches, &lf_info, 
            filters, ROWS_LOCATION, filter_level, num_blocks, 3, plane, block, p);
    }
    
    //YUV planes, then 2 more passes of Y plane
    if (thread_level_filter){
        vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, &lf_info, filters,
                DC_DIFFS_LOCATION, filter_level, 0, num_blocks, 3, plane, block, p);
        if (plane == 0){
            vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, &lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 3, num_blocks, 1, plane, block, p);
            vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, &lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 4, num_blocks, 1, plane, block, p);
        }
    }
}

kernel void vp8_loop_filter_vertical_edges_kernel(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info_n *lfi_n,
    global int *filters,
    int priority_level,
    int num_levels,
    global int *block_offsets,
    global int *priority_num_blocks,
    int frame_type
){
    size_t thread = get_global_id(0);
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);
    size_t num_blocks = get_global_size(2);
    
    int block_offset = block_offsets[priority_level];
    
    filters = &filters[4*block_offset];
    offsets = &offsets[16*block_offset];
    int filter_level = filters[block];
    
    loop_filter_info lf_info;
    set_lfi(lfi_n, &lf_info, frame_type, filter_level);

    int p = pitches[plane];
    
    int thread_level_filter = (thread<threads[plane]) & (filter_level!=0);
    int do_filter = filters[num_blocks*COLS_LOCATION + block] > 0;
    do_filter &= thread_level_filter;
    if (do_filter){
        vp8_mbloop_filter_vertical_edge_worker(s_base, offsets, pitches, &lf_info, filters,
            COLS_LOCATION, filter_level, plane, block, p);
    }
    
    //YUV planes, then 2 more passes of Y plane
    if (thread_level_filter){
        private uchar data[16];
        block_offset = num_blocks*3 + block*3 + plane;
        int s_off = offsets[block_offset] + p * thread;

        private_load16(data, s_base, s_off, 1);

        //YUV planes, then 2 more passes of Y plane
        vp8_loop_filter_vertical_edge_worker(data, offsets, pitches, &lf_info, filters,
                DC_DIFFS_LOCATION, filter_level, 1, num_blocks, 3, plane, block, p);
        if (plane == 0){
            vp8_loop_filter_vertical_edge_worker(&data[4], offsets, pitches, &lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 6, num_blocks, 1, plane, block, p);
            vp8_loop_filter_vertical_edge_worker(&data[8], offsets, pitches, &lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 7, num_blocks, 1,  plane, block, p);
        }

        private_save12(s_base, s_off, 1, data);
    }
    
}

void vp8_loop_filter_simple_horizontal_edge_worker
(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    loop_filter_info *lfi,
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

            if (i < threads[plane]){
                global uchar *flimit;
                if (use_mbflim == 0){
                    flimit = lfi->blim;
                } else {
                    flimit = lfi->mblim;
                }

                s_off += i;
                mask = vp8_simple_filter_mask(flimit[0], s_base[s_off-2*p], s_base[s_off-p], s_base[s_off], s_base[s_off+p]);
                vp8_simple_filter(mask, s_base, s_off - 2 * p, s_off - 1 * p, s_off, s_off + 1 * p);
            }
        }
    }
}

void vp8_loop_filter_simple_vertical_edge_worker(
    global unsigned char *s_base,
    global int *offsets, /* Y or YUV offsets for EACH block being processed*/
    global int *pitches, /* 1 or 3 values for Y or YUV pitches*/
    loop_filter_info *lfi, /* Single struct for the frame */
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

            if (i < threads[0]){
                global uchar *flimit;
                if (use_mbflim == 0){
                    flimit = lfi->blim;
                } else {
                    flimit = lfi->mblim;
                }

                s_off += p * i;
                mask = vp8_simple_filter_mask(flimit[0], s_base[s_off-2], s_base[s_off-1], s_base[s_off], s_base[s_off+1]);
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
    global loop_filter_info_n *lfi_n, /* Single struct for the frame */
    global int *filters_in, /* Filters for each block being processed */
    int filter_type, /* Should dc_diffs, rows, or cols be used?*/
    int priority_level,
    int num_levels,
    global int *block_offsets,
    global int *priority_num_blocks,
    int frame_type
){
    
    loop_filter_info lfi;
    int block_offset = block_offsets[priority_level];
    int filter_level = filters_in[4*block_offset + get_global_id(2)];
    set_lfi(lfi_n, &lfi, frame_type, filter_level);

    vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
            &lfi, filters_in, 1, COLS_LOCATION, 0, priority_level,
            block_offsets, priority_num_blocks
    );

    //3 Y plane iterations
    vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
            &lfi, filters_in, 0, DC_DIFFS_LOCATION, 1, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
            &lfi, filters_in, 0, DC_DIFFS_LOCATION, 2, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
            &lfi, filters_in, 0, DC_DIFFS_LOCATION, 3, priority_level,
            block_offsets, priority_num_blocks
    );
}

kernel void vp8_loop_filter_simple_horizontal_edges_kernel
(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info_n *lfi_n,
    global int *filters_in,
    int priority_level,
    int num_levels,
    global int *block_offsets,
    global int *priority_num_blocks,
    int frame_type
){

    loop_filter_info lfi;
    int block_offset = block_offsets[priority_level];
    int filter_level = filters_in[4*block_offset + get_global_id(2)];
    set_lfi(lfi_n, &lfi, frame_type, filter_level);
    
    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, &lfi,
            filters_in, 1, ROWS_LOCATION, 4, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, &lfi,
            filters_in, 0, DC_DIFFS_LOCATION, 5, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, &lfi,
            filters_in, 0, DC_DIFFS_LOCATION, 6, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, &lfi,
            filters_in, 0, DC_DIFFS_LOCATION, 7, priority_level,
            block_offsets, priority_num_blocks
    );
}

kernel void vp8_loop_filter_simple_all_edges_kernel
(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info_n *lfi_n,
    global int *filters_in,
    int priority_level,
    int num_levels,
    global int *block_offsets,
    global int *priority_num_blocks,
    int frame_type
)
{

    int block = (int)get_global_id(2);
    loop_filter_info lfi;
    
    for (int i = 0; i < num_levels; i++){
        int block_offset = block_offsets[priority_level];
        int filter_level = filters_in[4*block_offset + block];
        set_lfi(lfi_n, &lfi, frame_type, filter_level);
    
        
        if (block < priority_num_blocks[priority_level]){
            vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
                    &lfi, filters_in, 1, COLS_LOCATION, 0, priority_level,
                    block_offsets, priority_num_blocks
            );

            //3 Y plane iterations
            vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
                    &lfi, filters_in, 0, DC_DIFFS_LOCATION, 1, priority_level,
                    block_offsets, priority_num_blocks
            );
            vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
                    &lfi, filters_in, 0, DC_DIFFS_LOCATION, 2, priority_level,
                    block_offsets, priority_num_blocks
            );
            vp8_loop_filter_simple_vertical_edge_worker(s_base, offsets, pitches,
                    &lfi, filters_in, 0, DC_DIFFS_LOCATION, 3, priority_level,
                    block_offsets, priority_num_blocks
            );
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (block < priority_num_blocks[priority_level]){
            vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, &lfi,
                    filters_in, 1, ROWS_LOCATION, 4, priority_level,
                    block_offsets, priority_num_blocks
            );
            vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, &lfi,
                    filters_in, 0, DC_DIFFS_LOCATION, 5, priority_level,
                    block_offsets, priority_num_blocks
            );
            vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, &lfi,
                    filters_in, 0, DC_DIFFS_LOCATION, 6, priority_level,
                    block_offsets, priority_num_blocks
            );
            vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, &lfi,
                    filters_in, 0, DC_DIFFS_LOCATION, 7, priority_level,
                    block_offsets, priority_num_blocks
            );
        }
        
        barrier(CLK_GLOBAL_MEM_FENCE);
        priority_level++;
    }
}

//Inline and non-kernel functions follow.
__inline void vp8_mbfilter_mem(
    signed char mask,
    uchar hev,
    uchar *base
)
{
    char4 u;

    char *pq = (char*)base;
    pq[0] ^= 0x80;
    pq[1] ^= 0x80;
    pq[2] ^= 0x80;
    pq[3] ^= 0x80;
    pq[4] ^= 0x80;
    pq[5] ^= 0x80;
    
    /* add outer taps if we have high edge variance */
    char vp8_filter = sub_sat(pq[2], pq[5]);
    vp8_filter = clamp(vp8_filter + 3 * (pq[4] - pq[3]), -128, 127);
    vp8_filter &= mask;

    char2 filter = (char2)vp8_filter;
    filter &= (char2)hev;

    /* save bottom 3 bits so that we round one side +4 and the other +3 */
    char2 rounding = {4,3};
    filter = add_sat(filter, rounding);
    filter.s0 >>= 3;
    filter.s1 >>= 3;
    
    pq[4] = sub_sat(pq[4], filter.s0);
    pq[3] = add_sat(pq[3], filter.s1);

    /* only apply wider filter if not high edge variance */
    filter.s1 = vp8_filter & ~hev;

    /* roughly 3/7th, 2/7th, and 1/7th difference across boundary */
    u.s0 = clamp((63 + filter.s1 * 27) >> 7, -128, 127);
    u.s1 = clamp((63 + filter.s1 * 18) >> 7, -128, 127);
    u.s2 = clamp((63 + filter.s1 * 9) >> 7, -128, 127);
    u.s3 = 0;
    
    char4 s;
    s = sub_sat(vload4(0, &pq[4]), u);
    pq[4] = s.s0 ^ 0x80;
    pq[5] = s.s1 ^ 0x80;
    pq[6] = s.s2 ^ 0x80;
    pq[7] = s.s3;
    
    s = add_sat(vload4(0, pq), u.s3210);
    pq[0] = s.s0;
    pq[1] = s.s1 ^ 0x80;
    pq[2] = s.s2 ^ 0x80;
    pq[3] = s.s3 ^ 0x80;
}

__inline uchar8 vp8_mbfilter(
    signed char mask,
    uchar hev,
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
    filter &= (char2)hev;

    /* save bottom 3 bits so that we round one side +4 and the other +3 */
    char2 rounding = {4,3};
    filter = add_sat(filter, rounding);
    filter.s0 >>= 3;
    filter.s1 >>= 3;
    
    pq.s4 = sub_sat(pq.s4, filter.s0);
    pq.s3 = add_sat(pq.s3, filter.s1);

    /* only apply wider filter if not high edge variance */
    filter.s1 = vp8_filter & ~hev;

    /* roughly 3/7th, 2/7th, and 1/7th difference across boundary */
    u.s0 = clamp((63 + filter.s1 * 27) >> 7, -128, 127);
    u.s1 = clamp((63 + filter.s1 * 18) >> 7, -128, 127);
    u.s2 = clamp((63 + filter.s1 * 9) >> 7, -128, 127);
    u.s3 = 0;
    
    char4 s;
    s = sub_sat(pq.s4567, u);
    pq.s4567 = s ^ (char4){0x80, 0x80, 0x80, 0};
    s = add_sat(pq.s0123, u.s3210);
    pq.s0123 = s ^ (char4){0, 0x80, 0x80, 0x80};
    
    return as_uchar8(pq);
}

__inline char vp8_hevmask_mem(uchar thresh, uchar *pq){
#if 0
    signed char hev = 0;
    hev  |= (abs(pq[0] - pq[1]) > thresh) * -1;
    hev  |= (abs(pq[3] - pq[2]) > thresh) * -1;
    return hev;
#else
    uchar mask = abs_diff(pq[0], pq[1]) > thresh;
    mask |= abs_diff(pq[3], pq[2]) > thresh;
    return ~mask + 1;
#endif
}

/* is there high variance internal edge ( 11111111 yes, 00000000 no) */
__inline char vp8_hevmask(uchar thresh, uchar4 pq)
{
    return ~any(abs_diff(pq.s03, pq.s12) > (uchar2)thresh) + 1;
}

__inline signed char vp8_filter_mask_mem(uc limit, uc blimit, uchar *pq)
{
    //Only apply the filter if the difference is LESS than 'limit'
    char mask = (abs_diff(pq[0], pq[1]) > limit);
    mask |= (abs_diff(pq[1], pq[2]) > limit);
    mask |= (abs_diff(pq[2], pq[3]) > limit);
    mask |= (abs_diff(pq[5], pq[4]) > limit);
    mask |= (abs_diff(pq[6], pq[5]) > limit);
    mask |= (abs_diff(pq[7], pq[6]) > limit);
    mask |= (abs_diff(pq[3], pq[4]) * 2 + abs_diff(pq[2], pq[5]) / 2  > blimit);
    return mask - 1;
}

/* should we apply any filter at all ( 11111111 yes, 00000000 no) */
__inline signed char vp8_filter_mask(uc limit, uc blimit, uchar8 pq)
{
#if 1
   //Only apply the filter if the difference is LESS than 'limit'
    signed char mask = (abs_diff(pq.s0, pq.s1) > limit);
    mask |= (abs_diff(pq.s1, pq.s2) > limit);
    mask |= (abs_diff(pq.s2, pq.s3) > limit);
    mask |= (abs_diff(pq.s5, pq.s4) > limit);
    mask |= (abs_diff(pq.s6, pq.s5) > limit);
    mask |= (abs_diff(pq.s7, pq.s6) > limit);
    mask |= (abs_diff(pq.s3, pq.s4) * 2 + abs_diff(pq.s2, pq.s5) / 2  > blimit);
    return mask - 1;
#else
	char8 mask8 = abs_diff(pq.s01256700, pq.s12345600) > limit;
	mask8 |= (char8)(abs_diff(pq.s3, pq.s4) * 2 + abs_diff(pq.s2, pq.s5) / 2 > blimit);
	mask8--;
	return any(mask8) ? -1 : 0;
#endif
}

/* should we apply any filter at all ( 11111111 yes, 00000000 no) */
static __inline signed char vp8_simple_filter_mask(uc blimit, uc p1, uc p0, uc q0, uc q1)
{
    signed char mask = (abs(p0 - q0) * 2 + abs(p1 - q1) / 2  <= blimit) * -1;
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
    char2 rounding = {4,3};
    filter = add_sat((char2)vp8_filter, rounding);
    filter.s0 >>= 3;
    filter.s1 >>= 3;

    u = sub_sat(pq.s2, filter.s0);
    *oq0  = u ^ 0x80;

    u = add_sat(pq.s1, filter.s1);
    *op0 = u ^ 0x80;
}
