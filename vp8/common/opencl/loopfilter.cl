#pragma OPENCL EXTENSION cl_amd_printf : enable

typedef unsigned char uc;
typedef signed char sc;

int vp8_filter_mask_mem(uint limit, uint blimit, uint pq0, uint pq1,
        uint pq2, uint pq3, uint pq4, uint pq5, uint pq6, uint pq7);
__inline int vp8_filter_mask(uint, uint, uint8);

int vp8_hevmask_mem(uint, uchar, uchar, uchar, uchar);
__inline int vp8_hevmask(uint, uint4);

__inline void vp8_filter_mem( signed char mask, uchar hev, local uchar *, local uchar *, local uchar *, local uchar * );
__inline uint4 vp8_filter( int mask, uint hev, uint4 base);

__inline void vp8_mbfilter_mem(int mask, uint hev, local uchar*, int p);
__inline uint8 vp8_mbfilter(int mask, uint hev, uint8);

__inline int vp8_simple_filter_mask(uint, uint, uint, uint, uint);

__inline void vp8_simple_filter(int mask,global uint *base, int op1_off,int op0_off,int oq0_off,int oq1_off);

constant size_t threads[3] = {16, 8, 8};

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
    unsigned int mblim;
    unsigned int blim;
    unsigned int lim;
    unsigned int hev_thr;
} loop_filter_info;

__inline void set_lfi(global loop_filter_info_n *lfi_n, local loop_filter_info *lfi, int frame_type, int filter_level);


//Load + Store functions
__inline uint8 load8(global uint *s_base, int s_off, int p);
__inline uint8 load8_local(local uint *s_base, int s_off, int p);
__inline uint16 load16(global uint *s_base, int s_off, int p);

__inline void save4(global uint *s_base, int s_off, int p, uint8 data);
__inline void save6(global uint *s_base, int s_off, int p, uint8 data);
__inline void save4_local(local uint *s_base, int s_off, int p, uint8 data);
__inline void save6_local(local uint *s_base, int s_off, int p, uint8 data);
__inline void save12(global uint *s_base, int s_off, int p, uint16 data);

__inline void load_mb(int size, local uint *dst, global uint *src, int src_off, int src_pitch, int mb_row, int mb_col, int dc_diffs, int thread);
__inline void save_mb(int size, local uint *src, global uint *dst, int dst_off, int dst_pitch, int mb_row, int mb_col, int dc_diffs, int thread);

__inline void vp8_filter_mem(
    signed char mask,
    uchar hev,
    local uchar *p1,
    local uchar *p0,
    local uchar *q0,
    local uchar *q1
)
{

    char pq0, pq1, pq2, pq3;
    pq0 = *((local char*)p1) ^ 0x80;
    pq1 = *((local char*)p0) ^ 0x80;
    pq2 = *((local char*)q0) ^ 0x80;
    pq3 = *((local char*)q1) ^ 0x80;
    
    char vp8_filter;
    char2 Filter;
    char4 u;

    /* add outer taps if we have high edge variance */
    vp8_filter = sub_sat(pq0, pq3);
    vp8_filter &= hev;

    /* inner taps */
    vp8_filter = clamp(vp8_filter + 3 * (pq2 - pq1), -128, 127);
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

    u.s0 = add_sat(pq0, vp8_filter);
    u.s1 = add_sat(pq1, Filter.s1);
    u.s2 = sub_sat(pq2, Filter.s0);
    u.s3 = sub_sat(pq3, vp8_filter);

    *p1 = (uchar)(u.s0 ^ 0x80);
    *p0 = (uchar)(u.s1 ^ 0x80);
    *q0 = (uchar)(u.s2 ^ 0x80);
    *q1 = (uchar)(u.s3 ^ 0x80);

}

__inline uint4 vp8_filter(
    int mask,
    uint hev,
    uint4 base
)
{
    //Need to convert the uint4 to int4, but intermediately treat it like the
    //original alrogithm which XOR'd with 0x80 on a uchar.
    int4 pq = convert_int4(convert_char4(base) ^ (char4)0x80);
    
    int vp8_filter;
    int2 Filter;
    int4 u;

    /* add outer taps if we have high edge variance */
    vp8_filter = clamp(pq.s0 - pq.s3, -128, 127);
    vp8_filter &= hev;

    /* inner taps */
    vp8_filter = clamp(vp8_filter + 3 * (pq.s2 - pq.s1), -128, 127);
    vp8_filter &= mask;

    /* save bottom 3 bits so that we round one side +4 and the other +3
     * if it equals 4 we'll set to adjust by -1 to account for the fact
     * we'd round 3 the other way
     */
    int2 rounding = {4,3};
    Filter = clamp((int2)vp8_filter + rounding, -128, 127) >> 3;

    /* outer tap adjustments */
    vp8_filter = Filter.s0 + 1;
    vp8_filter >>= 1;
    vp8_filter &= ~hev;

    u.s0 = pq.s0 + vp8_filter;
    u.s1 = pq.s1 + Filter.s1;
    u.s2 = pq.s2 - Filter.s0;
    u.s3 = pq.s3 - vp8_filter;
    u = clamp(u, -128, 127);

    return convert_uint4(convert_uchar4(u) ^ (uchar4)0x80);

}

__inline uint8 load8(global uint *s_base, int s_off, int p){
    uint8 data;
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

__inline uint16 load16(global uint *s_base, int s_off, int p){
    uint16 data;
    data.s01234567 = load8(s_base, s_off, p);
    data.s89abcdef = load8(s_base, s_off+8*p, p);
    return data;
}


__inline uint8 load8_local(local uint *s_base, int s_off, int p){
    uint8 data;
    data.s0 = s_base[s_off-4*p];
    data.s1 = s_base[s_off-3*p];
    data.s2 = s_base[s_off-2*p];
    data.s3 = s_base[s_off-p];
    data.s4 = s_base[s_off];
    data.s5 = s_base[s_off+p];
    data.s6 = s_base[s_off+2*p];
    data.s7 = s_base[s_off+3*p];
    return data & (uint8)0x000000ff;
}

__inline void save4(global uint *s_base, int s_off, int p, uint8 data){
    s_base[s_off - 2*p] = data.s2;
    s_base[s_off - p  ] = data.s3;
    s_base[s_off      ] = data.s4;
    s_base[s_off + p  ] = data.s5;
}

__inline void save12(global uint *s_base, int s_off, int p, uint16 data){
    save4(s_base, s_off, p, data.s01234567);
    save4(s_base, s_off+4*p, p, data.s456789ab);
    save4(s_base, s_off+8*p, p, data.s89abcdef);
}

__inline void save4_local(local uint *s_base, int s_off, int p, uint8 data){
    data &= (uint8)0x000000ff;
    s_base[s_off - 2*p] = data.s2;
    s_base[s_off - p  ] = data.s3;
    s_base[s_off      ] = data.s4;
    s_base[s_off + p  ] = data.s5;
}

__inline void save6(global uint *s_base, int s_off, int p, uint8 data){
    s_base[s_off - 3*p] = data.s1;
    s_base[s_off - 2*p] = data.s2;
    s_base[s_off - p  ] = data.s3;
    s_base[s_off      ] = data.s4;
    s_base[s_off + p  ] = data.s5;
    s_base[s_off + 2*p  ] = data.s6;
}

__inline void save6_local(local uint *s_base, int s_off, int p, uint8 data){
    data &= (uint8)0x000000ff;
    s_base[s_off - 3*p] = data.s1;
    s_base[s_off - 2*p] = data.s2;
    s_base[s_off - p  ] = data.s3;
    s_base[s_off      ] = data.s4;
    s_base[s_off + p  ] = data.s5;
    s_base[s_off + 2*p  ] = data.s6;
}

// Filters horizontal edges of inner blocks in a Macroblock
__inline void vp8_loop_filter_horizontal_edge_worker(
    global uint *s_base,
    int source_offset,
    global int *pitches, /* pitch */
    local loop_filter_info *lfi,
    global int *filters,
    int dc_diffs,
    int filter_level,
    int cur_iter,
    size_t num_blocks,
    size_t num_planes,
    size_t thread,
    size_t plane,
    size_t block,
    int p //pitches[plane]
){
    if (dc_diffs > 0){
        int s_off = source_offset + 4*cur_iter*p; //Move down 4 lines per iter
        s_off += thread; //Move to the right part of the horizontal line

        uint8 data = load8(s_base, s_off, p);

        int mask = vp8_filter_mask(lfi->lim, lfi->blim, data);

        uint hev = vp8_hevmask(lfi->hev_thr, data.s2345);

        data.s2345 = vp8_filter(mask, hev, data.s2345);

        save4(s_base, s_off, p, data);
    }
}

// Filters horizontal edges of inner blocks in a Macroblock
__inline void vp8_loop_filter_horizontal_edge_worker_local(
    local uint *s_base,
    local loop_filter_info *lfi,
    int dc_diffs,
    int cur_iter,
    size_t thread,
    int p //pitches[plane]
){
    if (dc_diffs > 0){
        int s_off = cur_iter*p + thread; //Move down 4 lines per iter
        //Move to the right part of the horizontal line
        uint8 data = load8_local(s_base, s_off, p);

        int mask = vp8_filter_mask(lfi->lim, lfi->blim, data);

        uint hev = vp8_hevmask(lfi->hev_thr, data.s2345);

        data.s2345 = vp8_filter(mask, hev, data.s2345);

        save4_local(s_base, s_off, p, data);
    }
}

__inline void vp8_loop_filter_vertical_edge_worker(
    global uint *s_base,
    int source_offset,
    local loop_filter_info *lfi,
    int dc_diffs,
    int cur_iter,
    size_t thread,
    int p
){
    if (dc_diffs > 0){
        int s_off = source_offset + 4*cur_iter + thread*p; //Move right 4 cols per iter
        //Move down to the right part of the vertical line

        uint8 data = load8(s_base, s_off, 1);

        int mask = vp8_filter_mask(lfi->lim, lfi->blim, data);

        uint hev = vp8_hevmask(lfi->hev_thr, data.s2345);
        
        data.s2345 = vp8_filter(mask, hev, data.s2345);
        
        save4(s_base, s_off, 1, data);
    }
}

__inline uint8 vp8_loop_filter_vertical_edge_worker_vector(
    uint8 data,
    local loop_filter_info *lfi,
    int dc_diffs
){
    if (dc_diffs > 0){
        int mask = vp8_filter_mask(lfi->lim, lfi->blim, data);
        uint hev = vp8_hevmask(lfi->hev_thr, data.s2345);
        data.s2345 = vp8_filter(mask, hev, data.s2345);
    }
    return data;
}


__inline void vp8_loop_filter_vertical_edge_worker_local(
    local uint *s_base,
    local loop_filter_info *lfi,
    int dc_diffs,
    int cur_iter,
    size_t thread,
    int p
){
    if (dc_diffs > 0){
        int s_off = cur_iter + thread * p; //Move right 4 cols per iter
        //Move down to the right part of the vertical line
#if 1
        uint8 data = load8_local(s_base, s_off, 1);
        int mask = vp8_filter_mask(lfi->lim, lfi->blim, data);
        uint hev = vp8_hevmask(lfi->hev_thr, data.s2345);
        data.s2345 = vp8_filter(mask, hev, data.s2345);
        save4_local(s_base, s_off, 1, data);
#else
        uint sn4 = s_base[s_off-4];
        uint sn3 = s_base[s_off-3];
        uint sn2 = s_base[s_off-2];
        uint sn1 = s_base[s_off-1];
        uint s0 = s_base[s_off];
        uint s1 = s_base[s_off+1];
        uint s2 = s_base[s_off+2];
        uint s3 = s_base[s_off+3];

        int mask = vp8_filter_mask_mem(lfi->lim, lfi->blim, sn4, sn3, sn2, sn1, s0, s1, s2, s3);
        uint hev = vp8_hevmask_mem(lfi->hev_thr, sn2, sn1, s0, s1);
        vp8_filter_mem(mask, hev, s_base+s_off-2, s_base+s_off-1, s_base+s_off, s_base+s_off+1 );
#endif
    }
}

__inline void vp8_mbloop_filter_horizontal_edge_worker(
    global uint *s_base,
    int source_offset,
    local loop_filter_info *lfi,
    size_t thread,
    int p //pitches[plane]
){
    
    int s_off = source_offset + thread;

    uint8 data = load8(s_base, s_off, p);

    int mask = vp8_filter_mask(lfi->lim, lfi->mblim, data);

    uint hev = vp8_hevmask(lfi->hev_thr, data.s2345);

    data = vp8_mbfilter(mask, hev, data);

    save6(s_base, s_off, p, data);

}

__inline void vp8_mbloop_filter_horizontal_edge_worker_local(
    local uint *source,
    local loop_filter_info *lfi,
    size_t thread,
    int p //pitches[plane]
){
    
    uint8 data = load8_local(source, thread, p);
    
    int mask = vp8_filter_mask(lfi->lim, lfi->mblim, data);

    uint hev = vp8_hevmask(lfi->hev_thr, data.s2345);

    data = vp8_mbfilter(mask, hev, data);

    save6_local(source, thread, p, data);

}

__inline void vp8_mbloop_filter_vertical_edge_worker(
    global uint *s_base,
    int source_offset,
    local loop_filter_info *lfi,
    size_t thread,
    int p //pitches[plane]
){

    int s_off = source_offset + p*thread;

    uint8 data = load8(s_base, s_off, 1);

    int mask = vp8_filter_mask(lfi->lim, lfi->mblim, data);

    uint hev = vp8_hevmask(lfi->hev_thr, data.s2345);

    data = vp8_mbfilter(mask, hev, data);

    save6(s_base, s_off, 1, data);
}

__inline void vp8_mbloop_filter_vertical_edge_worker_local(
    local unsigned int *source,
    local loop_filter_info *lfi,
    size_t thread,
    int p //threads[plane]+4
){

    int s_off = p*thread;

    uint8 data = load8_local(source, s_off, 1);

    int mask = vp8_filter_mask(lfi->lim, lfi->mblim, data);

    uint hev = vp8_hevmask(lfi->hev_thr, data.s2345);

    data = vp8_mbfilter(mask, hev, data);

    save6_local(source, s_off, 1, data);
}

//Assumes a work group size of 1 plane
__inline void load_mb(int size, local uint *dst, global uint *src, int src_off, int src_pitch, int mb_row, int mb_col, int dc_diffs, int thread){
    //Load 4 row top border if row != 0, starting at row 0, col 4
    int dst_pitch = size + 4;
    int row_start, row_end, col_start, col_end;

    if (dc_diffs > 0){
        row_end = 0;
        col_end = size;
    } else {
        row_end = 4;
        col_end = 4;
    }

    row_start = (mb_row > 0) * -4;

    //Load 4 col left border if col != 0, otherwise just the pixels of block data
    col_start = (mb_col > 0) * -4;

    for (int i = row_start; i < row_end; i++){
        dst[i*dst_pitch + thread] = src[i*src_pitch + src_off + thread];
    }

    //Load 16x16 or 8x8 pixels of Macroblock data with destination starting at
    //row 4, col 4.
    dst = &dst[thread * dst_pitch];
    src = &src[thread * src_pitch + src_off];
    for (int i = col_start; i < col_end; i++){
        dst[i] = src[i];
    }
}

__inline void save_mb(int size, local uint *src, global uint *dst, int dst_off, int dst_pitch, int mb_row, int mb_col, int dc_diffs, int thread){
    //Load 4 row top border if row != 0, starting at row 0, col 4
    int src_pitch = size + 4;
    int row_end, col_start, col_end;

    if (dc_diffs > 0){
        row_end = 0;
        col_end = size;
    } else {
        row_end = 3;
        col_end = 3;
    }

    //Save 3 col left border if col != 0, otherwise just the pixels of block data
    if (mb_col > 0){
        col_start = -3;
    } else {
        col_start = 0;
    }

    if (mb_row > 0){
        for (int i = -3; i < row_end; i++){
            dst[i*dst_pitch + dst_off + thread] = src[i*src_pitch + thread];
        }
    }

    //Save 16x16 or 8x8 pixels of Macroblock data with destination starting at
    //row 4, col 4.
    for (int i = col_start; i < col_end; i++){
        dst[thread*dst_pitch + dst_off + i] = src[thread*src_pitch + i];
    }
}

kernel void vp8_loop_filter_all_edges_kernel(
    global unsigned int *s_base,
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
    size_t block = get_global_id(2);
    size_t thread = get_global_id(0);
    size_t plane = get_global_id(1);

    int block_offset = block_offsets[priority_level];
    global int *filters = &filters_in[4*block_offset];
    int filter_level = filters[block];
    if ( filter_level <= 0 || thread >= threads[plane])
        return;

    global int *offsets = &offsets_in[3*block_offset];
    size_t num_blocks = priority_num_blocks[priority_level];
    local loop_filter_info lf_info;
    local int mb_row, mb_col, dc_diffs;
    if (get_local_id(0) == 0){ //shared among all local threads, save bandwidth
        set_lfi(lfi_n, &lf_info, frame_type, filter_level);

        mb_col = filters[num_blocks * COLS_LOCATION + block];
        mb_row = filters[num_blocks * ROWS_LOCATION + block];
        dc_diffs = filters[num_blocks * DC_DIFFS_LOCATION + block];
    }
    //barrier(CLK_LOCAL_MEM_FENCE);

    int source_offset = offsets[block*3 + plane];
    
    int p = pitches[plane];    
    
#define USE_LOCAL_MEM_FILTER 1
#if USE_LOCAL_MEM_FILTER
    //At the moment this local memory mechanism only works if local number of
    //threads/plane == global number of threads/plane.
    //This is forced in loop_filter_filters.c
    
    local uint mb_data_actual[400]; //Local copy of frame data for current plane
    int mb_offset, mb_pitch;
    
    int num_threads = threads[plane];
    mb_pitch = num_threads+4;
    mb_offset = 4+4*mb_pitch;
    local uint *mb_data = &mb_data_actual[mb_offset];
    
    load_mb(num_threads, mb_data, s_base, source_offset, p, mb_row, mb_col, dc_diffs, thread);
    //write_mem_fence(CLK_LOCAL_MEM_FENCE);

    if ( mb_col > 0 ){
        vp8_mbloop_filter_vertical_edge_worker_local(mb_data, &lf_info, thread, mb_pitch);
    }

    //YUV planes, then 2 more passes of Y plane
    vp8_loop_filter_vertical_edge_worker_local(mb_data, &lf_info, 
            dc_diffs, 4, thread, mb_pitch);
    if (plane == 0){
        vp8_loop_filter_vertical_edge_worker_local(mb_data, &lf_info,
                dc_diffs, 8, thread, mb_pitch);
        vp8_loop_filter_vertical_edge_worker_local(mb_data, &lf_info,
                dc_diffs, 12, thread, mb_pitch);
    }
    
    if (mb_row > 0){
        vp8_mbloop_filter_horizontal_edge_worker_local(mb_data, &lf_info, thread, mb_pitch);
    }
    //YUV planes, then 2 more passes of Y plane
    vp8_loop_filter_horizontal_edge_worker_local(mb_data, &lf_info, dc_diffs, 4, thread, mb_pitch);
    if (plane == 0){
        vp8_loop_filter_horizontal_edge_worker_local(mb_data, &lf_info, dc_diffs, 8, thread, mb_pitch);
        vp8_loop_filter_horizontal_edge_worker_local(mb_data, &lf_info, dc_diffs, 12, thread, mb_pitch);
    }
    save_mb(num_threads, mb_data, s_base, source_offset, p, mb_row, mb_col, dc_diffs, thread);

#else
    //prefetch(&s_base[source_offset+thread*p], threads[plane]);
    
    //Load/stores directly out of global memory.
    if ( mb_col > 0 ){
        vp8_mbloop_filter_vertical_edge_worker(s_base, source_offset, &lf_info, thread, p);
    }

    //YUV planes, then 2 more passes of Y plane
    vp8_loop_filter_vertical_edge_worker(s_base, source_offset, &lf_info,
            dc_diffs, 1, thread, p);
    if (plane == 0){
        vp8_loop_filter_vertical_edge_worker(s_base, source_offset, &lf_info,
                dc_diffs, 2, thread, p);
        vp8_loop_filter_vertical_edge_worker(s_base, source_offset, &lf_info,
                dc_diffs, 3, thread, p);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (mb_row > 0){
        vp8_mbloop_filter_horizontal_edge_worker(s_base, source_offset, &lf_info, thread, p);
    }
    //YUV planes, then 2 more passes of Y plane
    vp8_loop_filter_horizontal_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
            dc_diffs, filter_level, 1, num_blocks, 3, thread, plane, block, p);
    if (plane == 0){
        vp8_loop_filter_horizontal_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                dc_diffs, filter_level, 2, num_blocks, 1, thread, plane, block, p);
        vp8_loop_filter_horizontal_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                dc_diffs, filter_level, 3, num_blocks, 1, thread, plane, block, p);
    }
#endif
}

kernel void vp8_loop_filter_horizontal_edges_kernel(
    global unsigned int *s_base,
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
    int dc_diffs = filters[num_blocks * DC_DIFFS_LOCATION + block];
    
    int source_offset = offsets[block*3 + plane];
    
    local loop_filter_info lf_info;
    set_lfi(lfi_n, &lf_info, frame_type, filter_level);

    int thread_level_filter = (thread<threads[plane]) & (filter_level!=0);
    int do_filter = filters[num_blocks*ROWS_LOCATION + block] > 0;
    do_filter &= thread_level_filter;
    if (do_filter){
        vp8_mbloop_filter_horizontal_edge_worker(s_base, source_offset, &lf_info, thread, p);
    }

    //YUV planes, then 2 more passes of Y plane
    if (thread_level_filter){
        vp8_loop_filter_horizontal_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                dc_diffs, filter_level, 1, num_blocks, 3, thread, plane, block, p);
        if (plane == 0){
            vp8_loop_filter_horizontal_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                    dc_diffs, filter_level, 2, num_blocks, 1, thread, plane, block, p);
            vp8_loop_filter_horizontal_edge_worker(s_base, source_offset, pitches, &lf_info, filters,
                    dc_diffs, filter_level, 3, num_blocks, 1, thread, plane, block, p);
        }
    }
}

kernel void vp8_loop_filter_vertical_edges_kernel(
    global unsigned int *s_base,
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
    int dc_diffs = filters[num_blocks * DC_DIFFS_LOCATION + block];
    
    int source_offset = offsets[block*3 + plane];
    
    local loop_filter_info lf_info;
    set_lfi(lfi_n, &lf_info, frame_type, filter_level);

    int p = pitches[plane];
    
    int thread_level_filter = (thread<threads[plane]) & (filter_level!=0);
    int do_filter = filters[num_blocks*COLS_LOCATION + block] > 0;
    do_filter &= thread_level_filter;
    if (do_filter){
        vp8_mbloop_filter_vertical_edge_worker(s_base, source_offset, &lf_info, thread, p);
    }
    
    //YUV planes, then 2 more passes of Y plane
    if (thread_level_filter){
        //YUV planes, then 2 more passes of Y plane
        vp8_loop_filter_vertical_edge_worker(s_base, source_offset, &lf_info,
                dc_diffs, 1, thread, p);
        if (plane == 0){
            vp8_loop_filter_vertical_edge_worker(s_base, source_offset, &lf_info,
                    dc_diffs, 2, thread, p);
            vp8_loop_filter_vertical_edge_worker(s_base, source_offset, &lf_info,
                    dc_diffs, 3, thread, p);
        }
    }
    
}

void vp8_loop_filter_simple_horizontal_edge_worker
(
    global unsigned int *s_base,
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
    global unsigned int *s_base,
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
    global unsigned int *s_base,
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
    global unsigned int *s_base,
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
    global unsigned int *s_base,
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
__inline uint8 vp8_mbfilter(
    int mask,
    uint hev,
    uint8 base
)
{
    int4 u;

    int8 pq = convert_int8(convert_char8(base) ^ (char8)0x80);
    
    /* add outer taps if we have high edge variance */
    int vp8_filter = clamp(pq.s2 - pq.s5, -128, 127);
    vp8_filter = clamp(vp8_filter + 3 * (pq.s4 - pq.s3), -128, 127);
    vp8_filter &= mask;

    int2 filter = (int2)vp8_filter;
    filter &= (int2)hev;

    /* save bottom 3 bits so that we round one side +4 and the other +3 */
    int2 rounding = (int2){4,3};
    filter = clamp(filter + rounding, -128, 127);
    filter.s0 >>= 3;
    filter.s1 >>= 3;
    
    pq.s4 = clamp(pq.s4 - filter.s0, -128, 127);
    pq.s3 = clamp(pq.s3 + filter.s1, -128, 127);

    /* only apply wider filter if not high edge variance */
    filter.s1 = vp8_filter & ~hev;

    /* roughly 3/7th, 2/7th, and 1/7th difference across boundary */
    u.s0 = clamp((63 + filter.s1 * 27) >> 7, -128, 127);
    u.s1 = clamp((63 + filter.s1 * 18) >> 7, -128, 127);
    u.s2 = clamp((63 + filter.s1 * 9) >> 7, -128, 127);
    u.s3 = 0;
    
    int8 s;
    s.s0123 = clamp(pq.s0123 + u.s3210, -128, 127);
    s.s4567 = clamp(pq.s4567 - u, -128, 127);

    return convert_uint8(convert_uchar8(s) ^ (uchar8)0x80);
}

/* is there high variance internal edge ( 11111111 yes, 00000000 no) */
__inline int vp8_hevmask(uint thresh, uint4 pq)
{
#if 1
    int mask = abs_diff(pq.s0, pq.s1) > thresh;
    mask |= abs_diff(pq.s3, pq.s2) > thresh;
    return mask * -1;
#else
    return any(abs_diff(pq.s03, pq.s12) > (uint2)thresh) * -1;
#endif
}

int vp8_filter_mask_mem(uint limit, uint blimit, uint pq0, uint pq1,
        uint pq2, uint pq3, uint pq4, uint pq5, uint pq6, uint pq7)
{
    //Only apply the filter if the difference is LESS than 'limit'
    int mask = (abs_diff(pq0, pq1) > limit);
    mask |= (abs_diff(pq1, pq2) > limit);
    mask |= (abs_diff(pq2, pq3) > limit);
    mask |= (abs_diff(pq5, pq4) > limit);
    mask |= (abs_diff(pq6, pq5) > limit);
    mask |= (abs_diff(pq7, pq6) > limit);
    mask |= (abs_diff(pq3, pq4) * 2 + abs_diff(pq2, pq5) / 2  > blimit);
    return mask - 1;
}

/* should we apply any filter at all ( 11111111 yes, 00000000 no) */
__inline int vp8_filter_mask(uint limit, uint blimit, uint8 pq)
{

#if 0
   //Only apply the filter if the difference is LESS than 'limit'
    int mask = (abs_diff(pq.s0, pq.s1) > limit);
    mask |= (abs_diff(pq.s1, pq.s2) > limit);
    mask |= (abs_diff(pq.s2, pq.s3) > limit);
    mask |= (abs_diff(pq.s5, pq.s4) > limit);
    mask |= (abs_diff(pq.s6, pq.s5) > limit);
    mask |= (abs_diff(pq.s7, pq.s6) > limit);
    mask |= (abs_diff(pq.s3, pq.s4) * 2 + abs_diff(pq.s2, pq.s5) / 2  > blimit);
    return mask - 1;
#else
    uint8 diffs = abs_diff(pq.s01256732, pq.s12345645);
    diffs.s6 = diffs.s6 * 2 + diffs.s7 / 2;
    uint8 limits = {limit,limit,limit,limit,limit,limit,blimit, 0xffffffff};
    return any(diffs > limits) - 1;
#endif
}

/* should we apply any filter at all ( 11111111 yes, 00000000 no) */
__inline int vp8_simple_filter_mask(uint blimit, uint p1, uint p0, uint q0, uint q1)
{
    //There's a bug somewhere that is messing up the inputs by casting uc -> sc -> int,
    //which is sign extending the uint values incorrectly.
    p1 &= 0x000000ff;
    p0 &= 0x000000ff;
    q0 &= 0x000000ff;
    q1 &= 0x000000ff;
    int mask = ((abs_diff(p0, q0) * 2 + abs_diff(p1, q1) / 2) <= blimit);
    return mask * -1;
}

__inline void vp8_simple_filter(
    int mask,
    global uint *base,
    int op1_off,
    int op0_off,
    int oq0_off,
    int oq1_off
)
{

    global int *op1 = (global int*)&base[op1_off];
    global int *op0 = (global int*)&base[op0_off];
    global int *oq0 = (global int*)&base[oq0_off];
    global int *oq1 = (global int*)&base[oq1_off];

    int vp8_filter;
    int2 filter;
    int4 pq = (int4){*op1, *op0, *oq0, *oq1};
    int4 sign = pq < 0 ? 1 : -1;
    pq += 0x80 * sign; //Equivalent to (char) ^ 0x80

    int u;

    vp8_filter = clamp(pq.s0 - pq.s3, -128, 127);
    vp8_filter = clamp(vp8_filter + 3 * (pq.s2 - pq.s1), -128, 127);
    vp8_filter &= mask;

    /* save bottom 3 bits so that we round one side +4 and the other +3 */
    int2 rounding = {4,3};
    filter = clamp((int2)vp8_filter + (int2)rounding, (int2)-128, (int2)127);
    filter.s0 >>= 3;
    filter.s1 >>= 3;

    u = clamp(pq.s2 - filter.s0, -128, 127);
    int sign1 = u < 0 ? 1 : -1;
    *oq0  = (u + 128 * sign1);

    u = clamp(pq.s1 + filter.s1, -128, 127);
    sign1 = u < 0 ? 1 : -1;
    *op0 = (u + 128 * sign1);
}


__inline void set_lfi(global loop_filter_info_n *lfi_n, local loop_filter_info *lfi, int frame_type, int filter_level){
    int hev_index = lfi_n->hev_thr_lut[frame_type][filter_level];
    lfi->mblim = lfi_n->mblim[filter_level][0];
    lfi->blim = lfi_n->blim[filter_level][0];
    lfi->lim = lfi_n->lim[filter_level][0];
    lfi->hev_thr = lfi_n->hev_thr[hev_index][0];
}
