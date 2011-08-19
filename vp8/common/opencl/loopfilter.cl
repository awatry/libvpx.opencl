#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

typedef unsigned char uc;
typedef signed char sc;

__inline signed char vp8_filter_mask(sc, sc, uc, uc, uc, uc, uc, uc, uc, uc);
__inline signed char vp8_simple_filter_mask(signed char, signed char, uc, uc, uc, uc);
__inline signed char vp8_hevmask(signed char, uc, uc, uc, uc);

__inline void vp8_mbfilter(signed char mask,signed char hev,global uc *op2,
    global uc *op1,global uc *op0,global uc *oq0,global uc *oq1,global uc *oq2);

void vp8_simple_filter(signed char mask,global uc *base, int op1_off,int op0_off,int oq0_off,int oq1_off);


typedef struct
{
    signed char lim[16];
    signed char flim[16];
    signed char thr[16];
    signed char mbflim[16];
} loop_filter_info;




void vp8_filter(
    signed char mask,
    signed char hev,
    global uc *base,
    int op1_off,
    int op0_off,
    int oq0_off,
    int oq1_off
)
{

    global uc *op1 = &base[op1_off];
    global uc *op0 = &base[op0_off];
    global uc *oq0 = &base[oq0_off];
    global uc *oq1 = &base[oq1_off];

    signed char ps0, qs0;
    signed char ps1, qs1;
    signed char vp8_filter, Filter1, Filter2;
    signed char u;

    ps1 = (signed char) * op1 ^ 0x80;
    ps0 = (signed char) * op0 ^ 0x80;
    qs0 = (signed char) * oq0 ^ 0x80;
    qs1 = (signed char) * oq1 ^ 0x80;

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
    *oq0 = u ^ 0x80;
    u = clamp(ps0 + Filter2, -128, 127);
    *op0 = u ^ 0x80;
    vp8_filter = Filter1;

    /* outer tap adjustments */
    vp8_filter += 1;
    vp8_filter >>= 1;
    vp8_filter &= ~hev;

    u = clamp(qs1 - vp8_filter, -128, 127);
    *oq1 = u ^ 0x80;
    u = clamp(ps1 + vp8_filter, -128, 127);
    *op1 = u ^ 0x80;
}


kernel void vp8_loop_filter_horizontal_edge_kernel
(
    global unsigned char *s_base,
    global int *offsets,
    global int *pitches, /* pitch */
    global loop_filter_info *lfi,
    global int *filter_levels,
    int use_mbflim,
    global int *threads,
    global int *apply_filters
)
{
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);

    if (plane < get_global_size(1)){
        if (block < get_global_size(2)){
            if (apply_filters[block] > 0){
                int filter_level = filter_levels[block];
                if (filter_level){
                    int p = pitches[plane];
                    int s_off = offsets[block*get_global_size(1)+plane];
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

                        mask = vp8_filter_mask(limit[i], flimit[i], s_base[s_off - 4*p],
                                s_base[s_off - 3*p], s_base[s_off - 2*p], s_base[s_off - p],
                                s_base[s_off], s_base[s_off + p], s_base[s_off + 2*p],
                                s_base[s_off + 3*p]);

                        hev = vp8_hevmask(thresh[i], s_base[s_off - 2*p], s_base[s_off - p],
                                s_base[s_off], s_base[s_off+p]);

                        vp8_filter(mask, hev, s_base, s_off - 2 * p, s_off - p, s_off,
                                s_off + p);
                    }
                }
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
    global int *filter_levels,
    int use_mbflim,
    global int *threads,
    global int *apply_filters
)
{
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);

    if (plane < get_global_size(1)){
        if (block < get_global_size(2)){
            if (apply_filters[block] > 0){
                int filter_level = filter_levels[block];
                if (filter_level){
                    int p = pitches[plane];
                    int s_off = offsets[block*get_global_size(1)+plane];
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
                        mask = vp8_filter_mask(limit[i], flimit[i],
                                s_base[s_off-4], s_base[s_off-3], s_base[s_off-2],
                                s_base[s_off-1], s_base[s_off], s_base[s_off+1],
                                s_base[s_off+2], s_base[s_off+3]);

                        hev = vp8_hevmask(thresh[i], s_base[s_off-2], s_base[s_off-1],
                                s_base[s_off], s_base[s_off+1]);

                        vp8_filter(mask, hev, s_base, s_off - 2, s_off - 1, s_off, s_off + 1);

                    }
                }
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
    global int *filter_levels,
    int use_mbflim,
    global int *threads,
    global int *apply_filters
)
{
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);

    if (plane < get_global_size(1)){
        if (block < get_global_size(2)){
            if (apply_filters[block] > 0){
                int filter_level = filter_levels[block];
                if (filter_level){
                    int p = pitches[plane];
                    int s_off = offsets[block*get_global_size(1)+plane];

                    global uc *s = s_base+s_off;

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


                        s += i;

                        mask = vp8_filter_mask(limit[i], flimit[i],
                                               s[-4*p], s[-3*p], s[-2*p], s[-1*p],
                                               s[0*p], s[1*p], s[2*p], s[3*p]);

                        hev = vp8_hevmask(thresh[i], s[-2*p], s[-1*p], s[0*p], s[1*p]);

                        vp8_mbfilter(mask, hev, s - 3 * p, s - 2 * p, s - 1 * p, s, s + 1 * p, s + 2 * p);

                    }
                }
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
    global int *filter_levels,
    int use_mbflim,
    global int *threads,
    global int *apply_filters
)
{
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);

    if (plane < get_global_size(1)){
        if (block < get_global_size(2)){
            if (apply_filters[block] > 0){
                int filter_level = filter_levels[block];
                if (filter_level){
                    int p = pitches[plane];
                    int s_off = offsets[block*get_global_size(1)+plane];

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

                        hev = vp8_hevmask(thresh[i], s[-2], s[-1], s[0], s[1]);

                        vp8_mbfilter(mask, hev, s - 3, s - 2, s - 1, s, s + 1, s + 2);

                    }
                }
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
    global int *filter_levels,
    int use_mbflim,
    global int *threads,
    global int *apply_filters
)
{
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);

    if (plane < get_global_size(1)){
        if (block < get_global_size(2)){
            if (apply_filters[block] > 0){
                int filter_level = filter_levels[block];
                if (filter_level){
                    int p = pitches[plane];
                    int s_off = offsets[block*get_global_size(1)+plane];

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
    }
}


kernel void vp8_loop_filter_simple_vertical_edge_kernel
(
    global unsigned char *s_base,
    global int *offsets, /* Y or YUV offsets for EACH block being processed*/
    global int *pitches, /* 1 or 3 values for Y or YUV pitches*/
    global loop_filter_info *lfi, /* Single struct for the frame */
    global int *filter_levels, /* Filter level for each block being processed */
    int use_mbflim, /* Use lfi->flim or lfi->mbflim, need once per kernel call */
    global int *threads, /* Thread counts per plane */
    global int *apply_filters /* Should the filter be applied (per block) */
)
{
    size_t plane = get_global_id(1);
    size_t block = get_global_id(2);

    if (plane < get_global_size(1)){
        if (block < get_global_size(2)){
            if (apply_filters[block] > 0){
                int filter_level = filter_levels[block];
                if (filter_level){
                    int p = pitches[plane];
                    int s_off = offsets[block*get_global_size(1)+plane];

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
    signed char s, u;
    int3 s3, u3;
    signed char vp8_filter;

    char2 filter;

    char3 ps = { *op0, *op1, *op2 };
    ps ^= (char3){0x80, 0x80, 0x80};

    char3 qs = { *oq0, *oq1, *oq2 };
    qs ^= (char3){0x80, 0x80, 0x80};

    /* add outer taps if we have high edge variance */
    vp8_filter = clamp(ps.s1 - qs.s1, -128, 127);
    vp8_filter = clamp(vp8_filter + 3 * (qs.s0 - ps.s0), -128, 127);
    vp8_filter &= mask;

    filter.s1 = vp8_filter;
    filter.s1 &= hev;
    filter.s0 = filter.s1;

    /* save bottom 3 bits so that we round one side +4 and the other +3 */
    filter += (char2){4,3};
    filter = clamp(filter, -128, 127);
    filter.s0 >>= 3;
    filter.s1 >>= 3;
    
    qs.s0 = clamp(qs.s0 - filter.s0, -128, 127);
    ps.s0 = clamp(ps.s0 + filter.s1, -128, 127);

    /* only apply wider filter if not high edge variance */
    vp8_filter &= ~hev;
    filter.s1 = vp8_filter;

    u3 = (int3){filter.s1, filter.s1, filter.s1};
    u3 *= (int3){27, 18, 9};
    u3 += 63;
    u3 >>= 7;
    u3 = clamp(u3, -128, 127);

    /* roughly 3/7th difference across boundary */
    s = clamp(qs.s0 - u3.s0, -128, 127);
    *oq0 = s ^ 0x80;
    s = clamp(ps.s0 + u3.s0, -128, 127);
    *op0 = s ^ 0x80;

    /* roughly 2/7th difference across boundary */
    s = clamp(qs.s1 - u3.s1, -128, 127);
    *oq1 = s ^ 0x80;
    s = clamp(ps.s1 + u3.s1, -128, 127);
    *op1 = s ^ 0x80;

    /* roughly 1/7th difference across boundary */
    s = clamp(qs.s2 - u3.s2, -128, 127);
    *oq2 = s ^ 0x80;
    s = clamp(ps.s2 + u3.s2, -128, 127);
    *op2 = s ^ 0x80;
}

/* is there high variance internal edge ( 11111111 yes, 00000000 no) */
__inline signed char vp8_hevmask(signed char thresh, uc p1, uc p0, uc q0, uc q1)
{
    signed char hev = 0;
    hev  |= (abs(p1 - p0) > thresh) * -1;
    hev  |= (abs(q1 - q0) > thresh) * -1;
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
    mask = ~mask;
    return mask;
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
