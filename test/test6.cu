__global__ void dslash(const LatticeComplex *U, const LatticeComplex *src, LatticeComplex *dest)
{
    __share__ int x = blockIdx.x;
    __share__ int y = blockIdx.y;
    __share__ int z = blockIdx.z;
    __share__ int t = threadIdx.x;
    __share__ const LatticeComplex i(0.0, 1.0);
    __share__ const LatticeComplex zero(0.0, 0.0);
    __share__ LatticeComplex tmp0(0.0, 0.0);
    __share__ LatticeComplex tmp1(0.0, 0.0);
    __share__ LatticeComplex share_src__o(0.0, 0.0);
    __share__ LatticeComplex share_src__o(0.0, 0.0);
    int tmp;

    // mass term and others
    // for (int s = 0; s < lat_s; s++)
    // {
    //     for (int c = 0; c < lat_c0; c++)
    //     {
    //         dest(x, y, z, t, s, c) += src(x, y, z, t, s, c) * 0;
    //     }
    // }
    // backward x
    if (x == 0)
    {
        tmp = lat_x - 1;
    }
    else
    {
        tmp = x - 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (share_src[index_fermi(tmp, y, z, t, 0, c1)] + share_src[index_fermi(tmp, y, z, t, 3, c1)] * i) * share_U[index_guage(tmp, y, z, t, 0, c1, c0)].conj();
            tmp1 += (share_src[index_fermi(tmp, y, z, t, 1, c1)] + share_src[index_fermi(tmp, y, z, t, 2, c1)] * i) * share_U[index_guage(tmp, y, z, t, 0, c1, c0)].conj();
        }
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] -= tmp1 * i;
        dest[index_fermi(x, y, z, t, 3, c0)] -= tmp0 * i;
    }
    // forward x
    if (x == lat_x - 1)
    {
        tmp = 0;
    }
    else
    {
        tmp = x + 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (share_src[index_fermi(tmp, y, z, t, 0, c1)] - share_src[index_fermi(tmp, y, z, t, 3, c1)] * i) * share_U[index_guage(x, y, z, t, 0, c0, c1)];
            tmp1 += (share_src[index_fermi(tmp, y, z, t, 1, c1)] - share_src[index_fermi(tmp, y, z, t, 2, c1)] * i) * share_U[index_guage(x, y, z, t, 0, c0, c1)];
        }
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] += tmp1 * i;
        dest[index_fermi(x, y, z, t, 3, c0)] += tmp0 * i;
    }
    // backward y
    if (y == 0)
    {
        tmp = lat_y - 1;
    }
    else
    {
        tmp = y - 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (share_src[index_fermi(x, tmp, z, t, 0, c1)] - share_src[index_fermi(x, tmp, z, t, 3, c1)]) * share_U[index_guage(x, tmp, z, t, 1, c1, c0)].conj();
            tmp1 += (share_src[index_fermi(x, tmp, z, t, 1, c1)] + share_src[index_fermi(x, tmp, z, t, 2, c1)]) * share_U[index_guage(x, tmp, z, t, 1, c1, c0)].conj();
        }
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 3, c0)] -= tmp0;
    }
    // forward y
    if (y == lat_y - 1)
    {
        tmp = 0;
    }
    else
    {
        tmp = y + 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (share_src[index_fermi(x, tmp, z, t, 0, c1)] + share_src[index_fermi(x, tmp, z, t, 3, c1)]) * share_U[index_guage(x, y, z, t, 1, c0, c1)];
            tmp1 += (share_src[index_fermi(x, tmp, z, t, 1, c1)] - share_src[index_fermi(x, tmp, z, t, 2, c1)]) * share_U[index_guage(x, y, z, t, 1, c0, c1)];
        }
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] -= tmp1;
        dest[index_fermi(x, y, z, t, 3, c0)] += tmp0;
    }
    // backward z
    if (z == 0)
    {
        tmp = lat_z - 1;
    }
    else
    {
        tmp = z - 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (share_src[index_fermi(x, y, tmp, t, 0, c1)] + share_src[index_fermi(x, y, tmp, t, 2, c1)] * i) * share_U[index_guage(x, y, tmp, t, 2, c1, c0)].conj();
            tmp1 += (share_src[index_fermi(x, y, tmp, t, 1, c1)] - share_src[index_fermi(x, y, tmp, t, 3, c1)] * i) * share_U[index_guage(x, y, tmp, t, 2, c1, c0)].conj();
        }
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] -= tmp0 * i;
        dest[index_fermi(x, y, z, t, 3, c0)] += tmp1 * i;
    }
    // forward z
    if (z == lat_z - 1)
    {
        tmp = 0;
    }
    else
    {
        tmp = z + 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (share_src[index_fermi(x, y, tmp, t, 0, c1)] - share_src[index_fermi(x, y, tmp, t, 2, c1)] * i) * share_U[index_guage(x, y, z, t, 2, c0, c1)];
            tmp1 += (share_src[index_fermi(x, y, tmp, t, 1, c1)] + share_src[index_fermi(x, y, tmp, t, 3, c1)] * i) * share_U[index_guage(x, y, z, t, 2, c0, c1)];
        }
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] += tmp0 * i;
        dest[index_fermi(x, y, z, t, 3, c0)] -= tmp1 * i;
    }
    // backward t
    if (t == 0)
    {
        tmp = lat_t - 1;
    }
    else
    {
        tmp = t - 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (share_src[index_fermi(x, y, z, tmp, 0, c1)] + share_src[index_fermi(x, y, z, tmp, 2, c1)]) * share_U[index_guage(x, y, z, tmp, 3, c1, c0)].conj();
            tmp1 += (share_src[index_fermi(x, y, z, tmp, 1, c1)] + share_src[index_fermi(x, y, z, tmp, 3, c1)]) * share_U[index_guage(x, y, z, tmp, 3, c1, c0)].conj();
        }
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 3, c0)] += tmp1;
    }
    // forward t
    if (t == lat_t - 1)
    {
        tmp = 0;
    }
    else
    {
        tmp = t + 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (share_src[index_fermi(x, y, z, tmp, 0, c1)] - share_src[index_fermi(x, y, z, tmp, 2, c1)]) * share_U[index_guage(x, y, z, t, 3, c0, c1)];
            tmp1 += (share_src[index_fermi(x, y, z, tmp, 1, c1)] - share_src[index_fermi(x, y, z, tmp, 3, c1)]) * share_U[index_guage(x, y, z, t, 3, c0, c1)];
        }
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] -= tmp0;
        dest[index_fermi(x, y, z, t, 3, c0)] -= tmp1;
    }
}