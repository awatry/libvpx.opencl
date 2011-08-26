#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

typedef unsigned char uc;
typedef signed char sc;

__inline signed char vp8_filter_mask(sc, sc, uc, uc, uc, uc, uc, uc, uc, uc);
__inline signed char vp8_simple_filter_mask(signed char, signed char, uc, uc, uc, uc);
__inline signed char vp8_hevmask(signed char, char4);

__inline void vp8_mbfilter(signed char mask,signed char hev,global uc *op2,
    global uc *op1,global uc *op0,global uc *oq0,global uc *oq1,global uc *oq2);

void vp8_simple_filter(signed char mask,global uc *base, int op1_off,int op0_off,int oq0_off,int oq1_off);

constant int threads[3] = {16, 8, 8};

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

//Reduce the number of global reads/writes by converting s_base[s_off +- [012]*[p]] into char4's
//and giving those as arguments to __inline functions.
//Can remove 1/2 - 2/3 of the source/dest buffer reads/writes this way.

//replace op1 with pixels.s0;
//    replace op0 with pixels.s1;
//    replace oq0 with pixels.s2;
//    replace oq1 with pixels.s3;
//    change function to return char4 and argument type to char4 instead of char4

typedef struct
{
    signed char lim[16];
    signed char flim[16];
    signed char thr[16];
    signed char mbflim[16];
} loop_filter_info;

#if __OPENCL_VERSION__ != __CL_VERSION_1_0__
//OpenCL 1.1 and higher give us clamp() using integer vector types
char4 vp8_filter(
    signed char mask,
    signed char hev,
    char4 s
)
{

    char2 filter;
    char2 ps, qs;
    char2 u2;

    signed char vp8_filter;
    signed char u;

    ps = (char2){s.s10} ^ (char2)0x80;
    qs = (char2){s.s23} ^ (char2)0x80;

    /* add outer taps if we have high edge variance */
    vp8_filter = clamp(ps.s1 - qs.s1, -128, 127);
    vp8_filter &= hev;

    /* inner taps */
    vp8_filter = clamp(vp8_filter + 3 * (qs.s0 - ps.s0), -128, 127);
    vp8_filter &= mask;

    /* save bottom 3 bits so that we round one side +4 and the other +3
     * if it equals 4 we'll set to adjust by -1 to account for the fact
     * we'd round 3 the other way
     */
    filter = clamp((char2){vp8_filter+4, vp8_filter+3}, (char2)-128, (char2)127);
    filter >>= (char2)3;

    u = clamp(qs.s0 - filter.s0, -128, 127);
    s.s2 = u ^ 0x80;
    u = clamp(ps.s0 + filter.s1, -128, 127);
    s.s1 = u ^ 0x80;

    /* outer tap adjustments */
    vp8_filter = filter.s0;
    vp8_filter += 1;
    vp8_filter >>= 1;
    vp8_filter &= ~hev;

    u = clamp(qs.s1 - vp8_filter, -128, 127);
    s.s3 = u ^ 0x80;
    u = clamp(ps.s1 + vp8_filter, -128, 127);
    s.s0 = u ^ 0x80;

    return s;
}
#else
char4 vp8_filter(
    signed char mask,
    signed char hev,
    char4 s
)
{
    signed char ps0, qs0;
    signed char ps1, qs1;
    signed char vp8_filter, Filter1, Filter2;
    signed char u;

    ps1 = s.s0 ^ 0x80;
    ps0 = s.s1 ^ 0x80;
    qs0 = s.s2 ^ 0x80;
    qs1 = s.s3 ^ 0x80;

    /* add outer taps if we have high edge variance */
    vp8_filter = clamp(ps1 - qs1, -128, 127);

    vp8_filter &= hev;

    /* inner taps */
    vp8_filter = clamp(vp8_filter + 3 * (qs0 - ps0), -128, 127);
    vp8_filter &= mask;

    /* save bottom 3 bits so that we round one side +4 and the other +3
     * if it equals 4 we'll set to adjust by -1 to account for the fact
     * we'd round 3 the other way
     */
    Filter1 = clamp(vp8_filter + 4, -128, 127);
    Filter2 = clamp(vp8_filter + 3, -128, 127);
    Filter1 >>= 3;
    Filter2 >>= 3;
    u = clamp(qs0 - Filter1, -128, 127);
    s.s2 = u ^ 0x80;
    u = clamp(ps0 + Filter2, -128, 127);
    s.s1 = u ^ 0x80;
    vp8_filter = Filter1;

    /* outer tap adjustments */
    vp8_filter += 1;
    vp8_filter >>= 1;
    vp8_filter &= ~hev;

    u = clamp(qs1 - vp8_filter, -128, 127);
    s.s3 = u ^ 0x80;
    u = clamp(ps1 + vp8_filter, -128, 127);
    s.s0 = u ^ 0x80;

    return s;
}
#endif


kernel void vp8_loop_filter_horizontal_edge_kernel
(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches_in, /* pitch */
    global loop_filter_info *lfi,
    global int *filters,
    int use_mbflim,
    int filter_type,
    int cur_iter,
    int priority_offset
)
{
    private size_t plane = get_global_id(1);
    private size_t block = get_global_id(2);
    local size_t num_planes;
    local size_t num_blocks;
    num_planes = get_global_size(1);
    num_blocks = get_global_size(2);
    local int pitches[3];

    if (block == 0 && get_global_id(0) == 0){
        pitches[plane] = pitches_in[plane];
    }

    if (filters[num_blocks*filter_type + block] > 0){
        int filter_level = filters[block];
        if (filter_level){
            int p = pitches[plane];
            int block_offset = num_blocks*11 + cur_iter*num_blocks*num_planes + block*num_planes+plane;
            int s_off = offsets[block_offset+priority_offset];

            int  hev = 0; /* high edge variance */
            signed char mask = 0;
            size_t i = get_global_id(0);

            global signed char *limit, *flimit, *thresh;
            global loop_filter_info *lf_info;

            if (i < threads[plane]){
                lf_info = &lfi[filter_level];
                if (use_mbflim == 0){
                    flimit = lf_info->flim;
                } else {
                    flimit = lf_info->mbflim;
                }

                limit = lf_info->lim;
                thresh = lf_info->thr;

                s_off += i;

                char4 s = (char4){s_base[s_off-2*p],s_base[s_off-p], s_base[s_off],s_base[s_off+p]};

                mask = vp8_filter_mask(limit[i], flimit[i], s_base[s_off - 4*p],
                        s_base[s_off - 3*p], s_base[s_off - 2*p], s_base[s_off - p],
                        s_base[s_off], s_base[s_off + p], s_base[s_off + 2*p],
                        s_base[s_off + 3*p]);

                hev = vp8_hevmask(thresh[i], s);

                s = vp8_filter(mask, hev, s);

                s_base[s_off-2*p] = s.s0;
                s_base[s_off - p] = s.s1;
                s_base[s_off]     = s.s2;
                s_base[s_off + p] = s.s3;

            }
        }
    }
}


kernel void vp8_loop_filter_vertical_edge_kernel
(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters,
    int use_mbflim,
    int filter_type,
    int cur_iter,
    int priority_offset
)
{
    private size_t plane = get_global_id(1);
    private size_t block = get_global_id(2);
    local size_t num_planes;
    local size_t num_blocks;
    num_planes = get_global_size(1);
    num_blocks = get_global_size(2);
    
    if (filters[num_blocks*filter_type + block] > 0){
        int filter_level = filters[block];
        if (filter_level){
            int p = pitches[plane];
            int block_offset = cur_iter*num_blocks*num_planes + block*num_planes+plane;
            int s_off = offsets[block_offset+priority_offset];

            int  hev = 0; /* high edge variance */
            signed char mask = 0;
            size_t i= get_global_id(0);

            global signed char *limit, *flimit, *thresh;
            global loop_filter_info *lf_info;

            if (i < threads[plane]){
                lf_info = &lfi[filter_level];
                if (use_mbflim == 0){
                    flimit = lf_info->flim;
                } else {
                    flimit = lf_info->mbflim;
                }

                limit = lf_info->lim;
                thresh = lf_info->thr;

                s_off += p * i;

                char4 s = (char4){s_base[s_off-2],s_base[s_off-1], s_base[s_off],s_base[s_off+1]};

                mask = vp8_filter_mask(limit[i], flimit[i],
                        s_base[s_off-4], s_base[s_off-3], s_base[s_off-2],
                        s_base[s_off-1], s_base[s_off], s_base[s_off+1],
                        s_base[s_off+2], s_base[s_off+3]);

                hev = vp8_hevmask(thresh[i], s);

                s = vp8_filter(mask, hev, s);
                
                s_base[s_off-2] = s.s0;
                s_base[s_off-1] = s.s1;
                s_base[s_off]   = s.s2;
                s_base[s_off+1] = s.s3;


            }
        }
    }
}


kernel void vp8_mbloop_filter_horizontal_edge_kernel
(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters,
    int use_mbflim,
    int filter_type,
    int cur_iter,
    int priority_offset
)
{
    private size_t plane = get_global_id(1);
    private size_t block = get_global_id(2);
    local size_t num_planes;
    local size_t num_blocks;
    num_planes = get_global_size(1);
    num_blocks = get_global_size(2);

    if (filters[num_blocks*filter_type + block] > 0){
        int filter_level = filters[block];
        if (filter_level){
            int p = pitches[plane];
            int block_offset = 8*num_blocks + block*num_planes+plane;
            int s_off = offsets[block_offset+priority_offset];

            signed char hev = 0; /* high edge variance */
            signed char mask = 0;
            size_t i= get_global_id(0);

            global signed char *limit, *flimit, *thresh;
            global loop_filter_info *lf_info;

            if (i < threads[plane]){
                lf_info = &lfi[filter_level];
                if (use_mbflim == 0){
                    flimit = lf_info->flim;
                } else {
                    flimit = lf_info->mbflim;
                }

                limit = lf_info->lim;
                thresh = lf_info->thr;

                s_off += i;

                char4 s = (char4){s_base[s_off-2*p], s_base[s_off-1*p],
                        s_base[s_off], s_base[s_off+p]};
                        
                mask = vp8_filter_mask(limit[i], flimit[i],
                                       s_base[s_off-4*p], s_base[s_off-3*p], 
                                       s_base[s_off-2*p], s_base[s_off-1*p],
                                       s_base[s_off], s_base[s_off+p], 
                                       s_base[s_off+2*p], s_base[s_off+3*p]);

                hev = vp8_hevmask(thresh[i], s);

                vp8_mbfilter(mask, hev,
                    s_base + s_off - 3 * p,
                    s_base+s_off - 2 * p,
                    s_base+s_off - 1 * p,
                    s_base+s_off,
                    s_base+s_off + 1 * p,
                    s_base+s_off + 2 * p
                );

            }
        }
    }
}


kernel void vp8_mbloop_filter_vertical_edge_kernel
(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters,
    int use_mbflim,
    int filter_type,
    int cur_iter,
    int priority_offset
)
{
    private size_t plane = get_global_id(1);
    private size_t block = get_global_id(2);
    local size_t num_planes;
    local size_t num_blocks;
    num_planes = get_global_size(1);
    num_blocks = get_global_size(2);

    if (filters[num_blocks*filter_type + block] > 0){
        int filter_level = filters[block];
        if (filter_level){
            int p = pitches[plane];
            int block_offset = cur_iter*num_blocks*num_planes + block*num_planes+plane;
            int s_off = offsets[block_offset+priority_offset];

            global uc *s = s_base + s_off;

            signed char hev = 0; /* high edge variance */
            signed char mask = 0;
            size_t i= get_global_id(0);

            global signed char *limit, *flimit, *thresh;
            global loop_filter_info *lf_info;

            if (i < threads[plane]){
                lf_info = &lfi[filter_level];
                if (use_mbflim == 0){
                    flimit = lf_info->flim;
                } else {
                    flimit = lf_info->mbflim;
                }

                limit = lf_info->lim;
                thresh = lf_info->thr;

                s += p * i;

                mask = vp8_filter_mask(limit[i], flimit[i],
                                       s[-4], s[-3], s[-2], s[-1], s[0], s[1], s[2], s[3]);

                hev = vp8_hevmask(thresh[i], (char4){s[-2], s[-1], s[0], s[1]});

                vp8_mbfilter(mask, hev, s - 3, s - 2, s - 1, s, s + 1, s + 2);

            }
        }
    }
}


kernel void vp8_loop_filter_simple_horizontal_edge_kernel
(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches,
    global loop_filter_info *lfi,
    global int *filters,
    int use_mbflim,
    int filter_type,
    int cur_iter,
    int priority_offset
)
{
    private size_t plane = get_global_id(1);
    private size_t block = get_global_id(2);
    local size_t num_planes;
    local size_t num_blocks;
    num_planes = get_global_size(1);
    num_blocks = get_global_size(2);

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


kernel void vp8_loop_filter_simple_vertical_edge_kernel
(
    global unsigned char *s_base,
    global int *offsets, /* Y or YUV offsets for EACH block being processed*/
    global int *pitches, /* 1 or 3 values for Y or YUV pitches*/
    global loop_filter_info *lfi, /* Single struct for the frame */
    global int *filters, /* Filters for each block being processed */
    int use_mbflim, /* Use lfi->flim or lfi->mbflim, need once per kernel call */
    int filter_type, /* Should dc_diffs, rows, or cols be used?*/
    int cur_iter,
    int priority_offset
)
{
    private size_t plane = get_global_id(1);
    private size_t block = get_global_id(2);
    local size_t num_planes;
    local size_t num_blocks;
    num_planes = get_global_size(1);
    num_blocks = get_global_size(2);

    if (filters[filter_type * num_blocks + block] > 0){
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

                s_off += p * i;
                mask = vp8_simple_filter_mask(limit[i], flimit[i], s_base[s_off-2], s_base[s_off-1], s_base[s_off], s_base[s_off+1]);
                vp8_simple_filter(mask, s_base, s_off - 2, s_off - 1, s_off, s_off + 1);
            }
        }
    }
}

//Inline and non-kernel functions follow.
__inline void vp8_mbfilter(
    signed char mask,
    signed char hev,
    global uc *op2,
    global uc *op1,
    global uc *op0,
    global uc *oq0,
    global uc *oq1,
    global uc *oq2
)
{
    char s, u;
    char vp8_filter;

    char2 filter;

    char4 ps = { *op0, *op1, *op2, 0 };
    ps ^= (char4){0x80, 0x80, 0x80, 0x80};

    char4 qs = { *oq0, *oq1, *oq2, 0 };
    qs ^= (char4){0x80, 0x80, 0x80, 0x80};

    /* add outer taps if we have high edge variance */
    vp8_filter = clamp(ps.s1 - qs.s1, -128, 127);
    vp8_filter = clamp(vp8_filter + 3 * (qs.s0 - ps.s0), -128, 127);
    vp8_filter &= mask;

    filter = (char2)vp8_filter;
    filter &= (char2)hev;

    /* save bottom 3 bits so that we round one side +4 and the other +3 */
    filter = clamp(filter+(char2){4,3}, (char2)-128, (char2)127);
    filter >>= (char2)3;

    qs.s0 = clamp(qs.s0 - filter.s0, -128, 127);
    ps.s0 = clamp(ps.s0 + filter.s1, -128, 127);

    /* only apply wider filter if not high edge variance */
    vp8_filter &= ~hev;
    filter.s1 = vp8_filter;

    /* roughly 3/7th difference across boundary */
    u = clamp((63 + filter.s1 * 27) >> 7, -128, 127);
    s = clamp(qs.s0 - u, -128, 127);
    *oq0 = s ^ 0x80;
    s = clamp(ps.s0 + u, -128, 127);
    *op0 = s ^ 0x80;

    /* roughly 2/7th difference across boundary */
    u = clamp((63 + filter.s1 * 18) >> 7, -128, 127);
    s = clamp(qs.s1 - u, -128, 127);
    *oq1 = s ^ 0x80;
    s = clamp(ps.s1 + u, -128, 127);
    *op1 = s ^ 0x80;

    /* roughly 1/7th difference across boundary */
    u = clamp((63 + filter.s1 * 9) >> 7, -128, 127);
    s = clamp(qs.s2 - u, -128, 127);
    *oq2 = s ^ 0x80;
    s = clamp(ps.s2 + u, -128, 127);
    *op2 = s ^ 0x80;
}

/* is there high variance internal edge ( 11111111 yes, 00000000 no) */
__inline signed char vp8_hevmask(signed char thresh, char4 i)
{
    uchar4 s = convert_uchar4(i);
    signed char hev = 0;
    hev  |= (abs(s.s0 - s.s1) > thresh) * -1;
    hev  |= (abs(s.s3 - s.s2) > thresh) * -1;
    return hev;
}

/* should we apply any filter at all ( 11111111 yes, 00000000 no) */
__inline signed char vp8_filter_mask(
    signed char limit,
    signed char flimit,
     uc p3, uc p2, uc p1, uc p0, uc q0, uc q1, uc q2, uc q3)
{
    signed char mask = 0;
    mask |= (abs(p3 - p2) > limit) * -1;
    mask |= (abs(p2 - p1) > limit) * -1;
    mask |= (abs(p1 - p0) > limit) * -1;
    mask |= (abs(q1 - q0) > limit) * -1;
    mask |= (abs(q2 - q1) > limit) * -1;
    mask |= (abs(q3 - q2) > limit) * -1;
    mask |= (abs(p0 - q0) * 2 + abs(p1 - q1) / 2  > flimit * 2 + limit) * -1;
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
