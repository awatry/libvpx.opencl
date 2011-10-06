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
    unsigned char mblim;
    unsigned char blim;
    unsigned char lim;
    unsigned char hev_thr;
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

__inline uchar8 load8_local(local unsigned char *s_base, int s_off, int p){
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

__inline void save6_local(local unsigned char *s_base, int s_off, int p, uchar8 data){
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
    int source_offset,
    global int *pitches, /* pitch */
    local loop_filter_info *lfi,
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
        int s_off = source_offset + 4*cur_iter*p; //Move down 4 lines per iter
        s_off += thread; //Move to the right part of the horizontal line

        data = load8(s_base, s_off, p);

        char mask = vp8_filter_mask(lfi->lim, lfi->blim, data);

        char hev = vp8_hevmask(lfi->hev_thr, data.s2345);

        data.s2345 = vp8_filter(mask, hev, data.s2345);

        save4(s_base, s_off, p, data);
    }
}

__inline void vp8_loop_filter_vertical_edge_worker(
    global uchar *s_base,
    int source_offset,
    global int *pitches,
    local loop_filter_info *lfi,
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
    size_t thread = get_global_id(0);
    uchar8 data;
    if (filters[num_blocks*filter_type + block] > 0){
        int s_off = source_offset + 4*cur_iter; //Move right 4 cols per iter
        s_off += thread * p; //Move down to the right part of the vertical line

        data = load8(s_base, s_off, 1);

        char mask = vp8_filter_mask(lfi->lim, lfi->blim, data);

        int hev = vp8_hevmask(lfi->hev_thr, data.s2345);
        
        data.s2345 = vp8_filter(mask, hev, data.s2345);
        save4(s_base, s_off, 1, data);
    }
}

__inline void vp8_mbloop_filter_horizontal_edge_worker(
    global unsigned char *s_base,
    int source_offset,
    global int *pitches,
    local loop_filter_info *lfi,
    global int *filters,
    int filter_type,
    int filter_level,
    size_t plane,
    size_t block,
    int p //pitches[plane]
){
    
    size_t thread = get_global_id(0);
    int s_off = source_offset + thread;

    uchar8 data = load8(s_base, s_off, p);

    char mask = vp8_filter_mask(lfi->lim, lfi->mblim, data);

    char hev = vp8_hevmask(lfi->hev_thr, data.s2345);

    data = vp8_mbfilter(mask, hev, data);

    save6(s_base, s_off, p, data);

}

__inline void vp8_mbloop_filter_vertical_edge_worker(
    global unsigned char *s_base,
    int source_offset,
    global int *pitches,
    local loop_filter_info *lfi,
    global int *filters,
    int filter_type,
    int filter_level,
    size_t plane,
    size_t block,
    int p //pitches[plane]
){

    size_t thread = get_global_id(0);
    int s_off = source_offset + p*thread;

    uchar8 data = load8(s_base, s_off, 1);

    char mask = vp8_filter_mask(lfi->lim, lfi->mblim, data);

    int hev = vp8_hevmask(lfi->hev_thr, data.s2345);

    data = vp8_mbfilter(mask, hev, data);

    save6(s_base, s_off, 1, data);
}

__inline void vp8_mbloop_filter_vertical_edge_worker_local(
    local unsigned char *s_base,
    int source_offset,
    local loop_filter_info *lfi,
    global int *filters,
    int filter_type,
    int filter_level,
    size_t plane,
    size_t block,
    int p //pitches[plane]
){

    size_t thread = get_global_id(0);
    int s_off = source_offset + p*thread;

    uchar8 data = load8_local(s_base, s_off, 1);

    char mask = vp8_filter_mask(lfi->lim, lfi->mblim, data);

    int hev = vp8_hevmask(lfi->hev_thr, data.s2345);

    data = vp8_mbfilter(mask, hev, data);

    save6_local(s_base, s_off, 1, data);
}

__inline void set_lfi(global loop_filter_info_n *lfi_n, local loop_filter_info *lfi, int frame_type, int filter_level){
    if (get_local_id(0) == 0 && get_local_id(1)==0 && get_local_id(2)==0){
        int hev_index = lfi_n->hev_thr_lut[frame_type][filter_level];
        lfi->mblim = lfi_n->mblim[filter_level][0];
        lfi->blim = lfi_n->blim[filter_level][0];
        lfi->lim = lfi_n->lim[filter_level][0];
        lfi->hev_thr = lfi_n->hev_thr[hev_index][0];
    }
}

//Assumes a work group size of 1 plane
__inline void load_mb(int size, local uchar *dst, global uchar *src, int src_off, int src_pitch, int mb_row, int mb_col){
    //Load 4 row top border if row != 0, starting at row 0, col 4
    int dst_pitch = size + 4;
    int thread = get_global_id(0);
    if (mb_row > 0){
        dst[0*dst_pitch + 4 + thread] = src[-4*src_pitch + src_off + thread];
        dst[1*dst_pitch + 4 + thread] = src[-3*src_pitch + src_off + thread];
        dst[2*dst_pitch + 4 + thread] = src[-2*src_pitch + src_off + thread];
        dst[3*dst_pitch + 4 + thread] = src[-1*src_pitch + src_off + thread];
    }
    
    //Load 4 col left border if col != 0, otherwise just the pixels of block data
    if (mb_col > 0){
        for (int i = 0; i < size+4; i++){
            dst[(4+thread)*dst_pitch] = src[thread*src_pitch + src_off-4];
        }
    } else {
        //Load 16x16 or 8x8 pixels of Macroblock data with destination starting at
        //row 4, col 4.
        for (int i = 0; i < size; i++){
            dst[(4+thread)*dst_pitch + 4] = src[thread*src_pitch + src_off];
        }
    }
}

__inline void save_mb(int size, local uchar *src, global uchar *dst, int dst_off, int dst_pitch, int mb_row, int mb_col){
    //Load 4 row top border if row != 0, starting at row 0, col 4
    int src_pitch = size + 4;
    int thread = get_global_id(0);
    if (mb_row > 0){
        dst[-3*dst_pitch + dst_off + thread] = src[1*src_pitch + 4 + thread];
        dst[-2*dst_pitch + dst_off + thread] = src[2*src_pitch + 4 + thread];
        dst[-1*dst_pitch + dst_off + thread] = src[3*src_pitch + 4 + thread];
    }
    
    //Load 4 col left border if col != 0, otherwise just the pixels of block data
    if (mb_col > 0){
        for (int i = 0; i < size+4; i++){
            dst[thread*dst_pitch + dst_off-4] = src[(4+thread)*src_pitch];
        }
    } else {
        //Load 16x16 or 8x8 pixels of Macroblock data with destination starting at
        //row 4, col 4.
        for (int i = 0; i < size; i++){
            dst[thread*dst_pitch + dst_off] = src[(4+thread)*src_pitch + 4];
        }
    }
}

void compare_data(local uchar *mb_data, int local_pitch, global uchar *s_base, int s_pitch, int size){
    for (int row = 0; row < size; row++){
        int s_off = s_pitch*row;
        int l_off = local_pitch * row;
        for (int col = 0; col < size; col++){
            if (mb_data[l_off] != s_base[s_off]){
                printf("Data mismatch at {%d, %d}, %d != %d\n", row, col, mb_data[l_off], s_base[s_off]);
            }
        }
    }
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
    
    int block_offset = block_offsets[priority_level];

    global int *offsets = &offsets_in[3*block_offset];
    global int *filters = &filters_in[4*block_offset];
    size_t num_blocks = priority_num_blocks[priority_level];
    int filter_level = filters[block];
    local loop_filter_info lf_info;

    int source_offset = offsets[block*3 + plane];
    
    int p = pitches[plane];

/*
    local uchar mb_data[400]; //Local copy of frame data for current plane
    int mb_col = filters[num_blocks * COLS_LOCATION + block];
    int mb_row = filters[num_blocks * ROWS_LOCATION + block];
    if (thread < threads[plane]){
        load_mb(threads[plane], mb_data, s_base, source_offset, p, mb_row, mb_col);
    }

    if (get_global_id(0)==0 && get_global_id(1)==0){
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
        //if (mb_row > 0){
        //    printf("{%d, %d}: local = %d, global = %d\n", mb_row, mb_col, mb_data[4 + thread],  s_base[-4*p + source_offset + thread]);
        //}
        int row = 4; int col = 4;
        int local_pitch = threads[plane]+4;
        printf("{%d, %d}: local = %d, global = %d\n", mb_row, mb_col, mb_data[(4+row)*local_pitch + col + 4 + thread],  s_base[col + row*p + source_offset + thread]);
        
        //printf("s_base[%d] = %d, mb_data[%d] = %d\n", source_offset, s_base[source_offset], 4*(threads[plane]+4)+4, mb_data[4*(threads[plane]+4)+4]);
        //compare_data(mb_data+4*(threads[plane]+4)+4, threads[plane]+4, &s_base[source_offset], p, threads[plane]);
    }
    
*/
    int thread_level_filter = (thread<threads[plane]) & (filter_level!=0);

    set_lfi(lfi_n, &lf_info, frame_type, filter_level);

    if (thread_level_filter){
        if ( filters[num_blocks*COLS_LOCATION + block] > 0 ){
#if 1
            vp8_mbloop_filter_vertical_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                    COLS_LOCATION, filter_level, plane, block, p);
#else
            barrier(CLK_LOCAL_MEM_FENCE);
            int local_pitch = pitches[plane] + 4;
            vp8_mbloop_filter_vertical_edge_worker_local(mb_data, 4*local_pitch+4, &lf_info, filters,
                    COLS_LOCATION, filter_level, plane, block, local_pitch);
            write_mem_fence(CLK_LOCAL_MEM_FENCE);
            barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
            //Save local mem to global for now...
            save_mb(threads[plane], mb_data, s_base, source_offset, p, mb_row, mb_col);
            write_mem_fence(CLK_GLOBAL_MEM_FENCE);
            barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
#endif
        }

        //YUV planes, then 2 more passes of Y plane
        vp8_loop_filter_vertical_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                DC_DIFFS_LOCATION, filter_level, 1, num_blocks, 3, plane, block, p);
        if (plane == 0){
            vp8_loop_filter_vertical_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 2, num_blocks, 1, plane, block, p);
            vp8_loop_filter_vertical_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 3, num_blocks, 1,  plane, block, p);
        }

    }

    write_mem_fence(CLK_GLOBAL_MEM_FENCE);

    if (thread_level_filter){
        if (filters[num_blocks*ROWS_LOCATION + block] > 0){
            vp8_mbloop_filter_horizontal_edge_worker(s_base, source_offset, pitches, &lf_info, 
                filters, ROWS_LOCATION, filter_level, plane, block, p);
        }

        //YUV planes, then 2 more passes of Y plane
        vp8_loop_filter_horizontal_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                DC_DIFFS_LOCATION, filter_level, 1, num_blocks, 3, plane, block, p);
        if (plane == 0){
            vp8_loop_filter_horizontal_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 2, num_blocks, 1, plane, block, p);
            vp8_loop_filter_horizontal_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 3, num_blocks, 1, plane, block, p);
        }
    }

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
    offsets = &offsets[3*block_offset];
    int filter_level = filters[block];
    int p = pitches[plane];
    
    int source_offset = offsets[block*3 + plane];
    
    local loop_filter_info lf_info;
    set_lfi(lfi_n, &lf_info, frame_type, filter_level);

    int thread_level_filter = (thread<threads[plane]) & (filter_level!=0);
    int do_filter = filters[num_blocks*ROWS_LOCATION + block] > 0;
    do_filter &= thread_level_filter;
    if (do_filter){
        vp8_mbloop_filter_horizontal_edge_worker(s_base, source_offset, pitches, &lf_info, 
            filters, ROWS_LOCATION, filter_level, plane, block, p);
    }
    
    //YUV planes, then 2 more passes of Y plane
    if (thread_level_filter){
        vp8_loop_filter_horizontal_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                DC_DIFFS_LOCATION, filter_level, 1, num_blocks, 3, plane, block, p);
        if (plane == 0){
            vp8_loop_filter_horizontal_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 2, num_blocks, 1, plane, block, p);
            vp8_loop_filter_horizontal_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 3, num_blocks, 1, plane, block, p);
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
    offsets = &offsets[3*block_offset];
    int filter_level = filters[block];
    
    int source_offset = offsets[block*3 + plane];
    
    local loop_filter_info lf_info;
    set_lfi(lfi_n, &lf_info, frame_type, filter_level);

    int p = pitches[plane];
    
    int thread_level_filter = (thread<threads[plane]) & (filter_level!=0);
    int do_filter = filters[num_blocks*COLS_LOCATION + block] > 0;
    do_filter &= thread_level_filter;
    if (do_filter){
        vp8_mbloop_filter_vertical_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
            COLS_LOCATION, filter_level, plane, block, p);
    }
    
    //YUV planes, then 2 more passes of Y plane
    if (thread_level_filter){
        //YUV planes, then 2 more passes of Y plane
        vp8_loop_filter_vertical_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                DC_DIFFS_LOCATION, filter_level, 1, num_blocks, 3, plane, block, p);
        if (plane == 0){
            vp8_loop_filter_vertical_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 2, num_blocks, 1, plane, block, p);
            vp8_loop_filter_vertical_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                    DC_DIFFS_LOCATION, filter_level, 3, num_blocks, 1,  plane, block, p);
        }
    }
    
}

void vp8_loop_filter_simple_horizontal_edge_worker
(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    local loop_filter_info *lfi,
    global int *filters_in,
    int use_mbflim,
    int filter_type,
    int cur_iter,
    int priority_level,
    global int *block_offsets,
    global int *priority_num_blocks
){
    int filter_offset = 4*block_offsets[priority_level];
    private size_t block = get_global_id(2);
    size_t num_blocks = priority_num_blocks[priority_level];

    global int *filters = &filters_in[filter_offset];

    if (filters[num_blocks*filter_type + block] > 0){
        int filter_level = filters[block];
        if (filter_level){
            size_t i = get_global_id(0);
            int p = pitches[0];

            int s_off = cur_iter*4*p;
            s_off += i;
            
            signed char mask = 0;
            
            if (i < threads[0]){
                uchar flimit;
                if (use_mbflim == 0){
                    flimit = lfi->blim;
                } else {
                    flimit = lfi->mblim;
                }

                mask = vp8_simple_filter_mask(flimit, s_base[s_off-2*p], s_base[s_off-p], s_base[s_off], s_base[s_off+p]);
                vp8_simple_filter(mask, s_base, s_off - 2 * p, s_off - 1 * p, s_off, s_off + 1 * p);
            }
        }
    }
}

void vp8_loop_filter_simple_vertical_edge_worker(
    global unsigned char *s_base,
    global int *offsets, /* Y or YUV offsets for EACH block being processed*/
    global int *pitches, /* 1 or 3 values for Y or YUV pitches*/
    local loop_filter_info *lfi, /* Single struct for the frame */
    global int *filters_in, /* Filters for each block being processed */
    int use_mbflim, /* Use lfi->flim or lfi->mbflim, need once per kernel call */
    int filter_type, /* Should dc_diffs, rows, or cols be used?*/
    int cur_iter,
    int priority_level,
    global int *block_offsets,
    global int *priority_num_blocks
){
    int filter_offset = 4*block_offsets[priority_level];
    private size_t block = get_global_id(2);
    size_t num_blocks = priority_num_blocks[priority_level];

    global int *filters = &filters_in[filter_offset];

    if (filters[filter_type * num_blocks + block] > 0){
        int filter_level = filters[block];
        if (filter_level){
            size_t i = get_global_id(0);
            int p = pitches[0];
            
            int s_off = cur_iter*4;
            s_off += p * i;
            
            signed char mask = 0;

            if (i < threads[0]){
                uchar flimit;
                if (use_mbflim == 0){
                    flimit = lfi->blim;
                } else {
                    flimit = lfi->mblim;
                }

                mask = vp8_simple_filter_mask(flimit, s_base[s_off-2], s_base[s_off-1], s_base[s_off], s_base[s_off+1]);
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
    global int *block_offsets, //Number of previously processed blocks per level
    global int *priority_num_blocks,
    int frame_type
){
    
    local loop_filter_info lfi;
    int block = get_global_id(2);
    int block_offset = block_offsets[priority_level];
    int filter_level = filters_in[4*block_offset + block];
    set_lfi(lfi_n, &lfi, frame_type, filter_level);

    s_base += offsets[block_offsets[priority_level] + block];
    
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

    local loop_filter_info lfi;
    int block = get_global_id(2);
    int block_offset = block_offsets[priority_level];
    int filter_level = filters_in[4*block_offset];
    set_lfi(lfi_n, &lfi, frame_type, filter_level);
    
    s_base += offsets[block_offsets[priority_level] + block];

    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, &lfi,
            filters_in, 1, ROWS_LOCATION, 0, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, &lfi,
            filters_in, 0, DC_DIFFS_LOCATION, 1, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, &lfi,
            filters_in, 0, DC_DIFFS_LOCATION, 2, priority_level,
            block_offsets, priority_num_blocks
    );
    vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, &lfi,
            filters_in, 0, DC_DIFFS_LOCATION, 3, priority_level,
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
    local loop_filter_info lfi;
    
    int block_offset = block_offsets[priority_level];
    int filter_level = filters_in[4*block_offset + block];
    set_lfi(lfi_n, &lfi, frame_type, filter_level);

    s_base += offsets[block_offsets[priority_level] + block];

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
                filters_in, 1, ROWS_LOCATION, 0, priority_level,
                block_offsets, priority_num_blocks
        );
        vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, &lfi,
                filters_in, 0, DC_DIFFS_LOCATION, 1, priority_level,
                block_offsets, priority_num_blocks
        );
        vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, &lfi,
                filters_in, 0, DC_DIFFS_LOCATION, 2, priority_level,
                block_offsets, priority_num_blocks
        );
        vp8_loop_filter_simple_horizontal_edge_worker(s_base, offsets, pitches, &lfi,
                filters_in, 0, DC_DIFFS_LOCATION, 3, priority_level,
                block_offsets, priority_num_blocks
        );
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
__inline signed char vp8_simple_filter_mask(uc blimit, uc p1, uc p0, uc q0, uc q1)
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
