#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
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


#define LOCK(a) atom_cmpxchg(a, 0, 1)
#define UNLOCK(a) atom_xchg(a, 0)

//Credit for this lock-based barrier goes to Matthew Scarpino at:
//http://www.openclblog.com/2011/04/eureka.html
void wait_on_siblings(int priority_level, global int *locks)
{

    global int *mutex = locks;
    global int *count = &locks[1];
    
    if(get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0) {
        //printf("Level %d, Group [%d, %d, %d] waiting\n", priority_level, get_group_id(0), get_group_id(1), get_group_id(2));
        /* Increment the count */
#if 0
        while(LOCK(mutex))
            ;
        *count += 1;
        printf("Count is now %d\n", *count);
        UNLOCK(mutex);
#else
        atom_inc(count);
        //printf("level = %d, Count is now %d\n", priority_level, *count);
#endif
        /* Wait for everyone else to increment the count */
        int final_count = (priority_level+1) * get_num_groups(0)*get_num_groups(1)*get_num_groups(2);
        //printf("final count = %d\n", final_count);
        //while(*count < final_count)
        //    ;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}


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

    u.s0 = add_sat(pq.s0, vp8_filter);
    u.s1 = add_sat(pq.s1, Filter.s1);
    u.s2 = sub_sat(pq.s2, Filter.s0);
    u.s3 = sub_sat(pq.s3, vp8_filter);

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
        if ( cur_iter == 0 || plane == 0){
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

    if ((cur_iter == 1 || plane == 0) && filters[num_blocks*filter_type + block] > 0){
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
    global int *filters
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

void vp8_mbloop_filter_vertical_edge_worker(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters,
    int filter_type
){
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);
    size_t num_planes = 3;
    size_t num_blocks = get_global_size(2);

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

                uchar8 data = vload8(0, &s_base[s_off-4]);
                
                mask = vp8_filter_mask(limit[i], flimit[i], data);
                
                hev = vp8_hevmask(thresh[i], data.s2345);
                
                data = vp8_mbfilter(mask, hev, data);

                vstore8(data, 0, &s_base[s_off-4]);
            }
        }
    }
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
    global int *priority_num_blocks,
    global int *locks
){

    for (int i = 0; i < num_levels; i++){
        int block_offset = block_offsets[priority_level];

        global int *offsets = &offsets_in[16*block_offset];
        global int *filters = &filters_in[4*block_offset];
            
        if (get_global_id(2) < priority_num_blocks[priority_level]){
            //if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0)
            //    printf("Priority level is now %d\n", priority_level);

            //Prefetch vertical edge source pixels into global cache (horizontal isn't worth it)
            for(int plane = 0; plane < 3; plane++){
                int p = pitches[plane];
                int offset = get_global_id(2)*3+plane;
                int s_off = offsets[offset];
                for (int thread = 0; thread < 16; thread++){
                    prefetch(&s_base[s_off+p*thread-4], 8);
                }
            }

            vp8_mbloop_filter_vertical_edge_worker(s_base, offsets, pitches, lfi, filters,
                    COLS_LOCATION);

            //YUV planes, then 2 more passes of Y plane
            vp8_loop_filter_vertical_edge_worker(s_base, offsets, pitches, lfi, filters,
                    DC_DIFFS_LOCATION, 1);
            vp8_loop_filter_vertical_edge_worker(s_base, offsets, pitches, lfi, filters,
                    DC_DIFFS_LOCATION, 6);
            vp8_loop_filter_vertical_edge_worker(s_base, offsets, pitches, lfi, filters,
                    DC_DIFFS_LOCATION, 7);

        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (get_global_id(2) < priority_num_blocks[priority_level]){

            vp8_mbloop_filter_horizontal_edge_worker(s_base, offsets, pitches, lfi, 
                    filters);

            //YUV planes, then 2 more passes of Y plane
            vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, lfi, filters,
                    DC_DIFFS_LOCATION, 0);
            vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, lfi, filters,
                    DC_DIFFS_LOCATION, 3);
            vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, lfi, filters,
                    DC_DIFFS_LOCATION, 4);
        }
        
        wait_on_siblings(priority_level, locks);
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
    global int *priority_num_blocks,
    global int *locks
){
    int block_offset = block_offsets[priority_level];

    filters = &filters[4*block_offset];
    offsets = &offsets[16*block_offset];

    vp8_mbloop_filter_horizontal_edge_worker(s_base, offsets, pitches, lfi, 
            filters);
    
    //YUV planes, then 2 more passes of Y plane
    vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, lfi, filters,
            DC_DIFFS_LOCATION, 0);
    vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, lfi, filters,
            DC_DIFFS_LOCATION, 3);
    vp8_loop_filter_horizontal_edge_worker(s_base, offsets, pitches, lfi, filters,
            DC_DIFFS_LOCATION, 4);
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
    global int *priority_num_blocks,
    global int *locks
){
    int block_offset = block_offsets[priority_level];
    
    filters = &filters[4*block_offset];
    offsets = &offsets[16*block_offset];

    vp8_mbloop_filter_vertical_edge_worker(s_base, offsets, pitches, lfi, filters,
            COLS_LOCATION);
    
    //YUV planes, then 2 more passes of Y plane
    vp8_loop_filter_vertical_edge_worker(s_base, offsets, pitches, lfi, filters,
            DC_DIFFS_LOCATION, 1);
    vp8_loop_filter_vertical_edge_worker(s_base, offsets, pitches, lfi, filters,
            DC_DIFFS_LOCATION, 6);
    vp8_loop_filter_vertical_edge_worker(s_base, offsets, pitches, lfi, filters,
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
    int filter_type, /* Should dc_diffs, rows, or cols be used?*/
    int priority_level,
    int num_levels,
    global int *block_offsets,
    global int *priority_num_blocks,
    global int *locks
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
    global int *priority_num_blocks,
    global int *locks
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
    global int *priority_num_blocks,
    global int *locks
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
        
        wait_on_siblings(priority_level, locks);
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
    char4 s;
    char4 u;
    signed char vp8_filter;

    char2 filter;

    char8 pq = convert_char8(base);
    pq ^= (char8){0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0};
    
    /* add outer taps if we have high edge variance */
    vp8_filter = sub_sat(pq.s2, pq.s5);
    vp8_filter = clamp(vp8_filter + 3 * (pq.s4 - pq.s3), -128, 127);
    vp8_filter &= mask;

    filter = (char2)vp8_filter;
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
    u.s0 = clamp((63 + filter.s1 * 27) >> 7, -128, 127);
    u.s1 = clamp((63 + filter.s1 * 18) >> 7, -128, 127);
    u.s2 = clamp((63 + filter.s1 * 9) >> 7, -128, 127);
    
    s.s012 = sub_sat(pq.s456, u.s012);
    pq.s456 = s.s012 ^ 0x80;
    
    s.s012 = add_sat(pq.s321, u.s012);
    pq.s321 = s.s012 ^ 0x80;
    
    return convert_uchar8(pq);
}

/* is there high variance internal edge ( 11111111 yes, 00000000 no) */
__inline signed char vp8_hevmask(signed char thresh, uchar4 pq)
{
    signed char hev;
    hev  = (abs_diff(pq.s0, pq.s1) > thresh) * -1;
    hev  |= (abs_diff(pq.s3, pq.s2) > thresh) * -1;
    return hev;
}


/* should we apply any filter at all ( 11111111 yes, 00000000 no) */
__inline signed char vp8_filter_mask( signed char limit, signed char flimit,
        uchar8 pq)
{
    signed char mask = 0;

    //Only apply the filter if the difference is LESS than 'limit'
    mask |= (abs_diff(pq.s0, pq.s1) > limit);
    mask |= (abs_diff(pq.s1, pq.s2) > limit);
    mask |= (abs_diff(pq.s2, pq.s3) > limit);
    mask |= (abs_diff(pq.s5, pq.s4) > limit);
    mask |= (abs_diff(pq.s6, pq.s5) > limit);
    mask |= (abs_diff(pq.s7, pq.s6) > limit);
    mask |= (abs_diff(pq.s3, pq.s4) * 2 + abs_diff(pq.s2, pq.s5) / 2  > flimit * 2 + limit);
    mask *= -1;
    return ~mask;
    
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
    pq ^= 0x80;

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
