#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

typedef unsigned char uc;
typedef signed char sc;

__inline signed char vp8_filter_mask(sc, sc, uc, uc, uc, uc, uc, uc, uc, uc);
__inline signed char vp8_simple_filter_mask(signed char, signed char, uc, uc, uc, uc);
__inline signed char vp8_hevmask(signed char, uc, uc, uc, uc);
__inline signed char vp8_signed_char_clamp(int);




kernel void vp8_filter_kernel(
    signed char mask,
    signed char hev,
    global uc *op1,
    global uc *op0,
    global uc *oq0,
    global uc *oq1)
{
    signed char ps0, qs0;
    signed char ps1, qs1;
    signed char vp8_filter, Filter1, Filter2;
    signed char u;

    ps1 = (signed char) * op1 ^ 0x80;
    ps0 = (signed char) * op0 ^ 0x80;
    qs0 = (signed char) * oq0 ^ 0x80;
    qs1 = (signed char) * oq1 ^ 0x80;

    /* add outer taps if we have high edge variance */
    vp8_filter = vp8_signed_char_clamp(ps1 - qs1);
    vp8_filter &= hev;

    /* inner taps */
    vp8_filter = vp8_signed_char_clamp(vp8_filter + 3 * (qs0 - ps0));
    vp8_filter &= mask;

    /* save bottom 3 bits so that we round one side +4 and the other +3
     * if it equals 4 we'll set to adjust by -1 to account for the fact
     * we'd round 3 the other way
     */
    Filter1 = vp8_signed_char_clamp(vp8_filter + 4);
    Filter2 = vp8_signed_char_clamp(vp8_filter + 3);
    Filter1 >>= 3;
    Filter2 >>= 3;
    u = vp8_signed_char_clamp(qs0 - Filter1);
    *oq0 = u ^ 0x80;
    u = vp8_signed_char_clamp(ps0 + Filter2);
    *op0 = u ^ 0x80;
    vp8_filter = Filter1;

    /* outer tap adjustments */
    vp8_filter += 1;
    vp8_filter >>= 1;
    vp8_filter &= ~hev;

    u = vp8_signed_char_clamp(qs1 - vp8_filter);
    *oq1 = u ^ 0x80;
    u = vp8_signed_char_clamp(ps1 + vp8_filter);
    *op1 = u ^ 0x80;

}









__inline signed char vp8_signed_char_clamp(int t)
{
    t = (t < -128 ? -128 : t);
    t = (t > 127 ? 127 : t);
    return (signed char) t;
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
