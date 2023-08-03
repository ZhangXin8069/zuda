__global__ void dslash(const LatticeComplex *U, const LatticeComplex *src, LatticeComplex *dest)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;
    int t = threadIdx.x;
    const LatticeComplex i(0.0, 1.0);
    const LatticeComplex zero(0.0, 0.0);
    int tmp;
    LatticeComplex tmp0(0.0, 0.0);
    LatticeComplex tmp1(0.0, 0.0);
    __shared__ LatticeComplex share_dest[12];
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        share_dest[c0 * lat_s + 0] = dest[index_fermi(x, y, z, t, 0, c0)];
        share_dest[c0 * lat_s + 1] = dest[index_fermi(x, y, z, t, 1, c0)];
        share_dest[c0 * lat_s + 2] = dest[index_fermi(x, y, z, t, 2, c0)];
        share_dest[c0 * lat_s + 3] = dest[index_fermi(x, y, z, t, 3, c0)];
    }

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
            tmp0 += (src[index_fermi(tmp, y, z, t, 0, c1)] + src[index_fermi(tmp, y, z, t, 3, c1)] * i) * U[index_guage(tmp, y, z, t, 0, c1, c0)].conj();
            tmp1 += (src[index_fermi(tmp, y, z, t, 1, c1)] + src[index_fermi(tmp, y, z, t, 2, c1)] * i) * U[index_guage(tmp, y, z, t, 0, c1, c0)].conj();
        }
        share_dest[c0 * lat_s + 0] += tmp0;
        share_dest[c0 * lat_s + 1] += tmp1;
        share_dest[c0 * lat_s + 2] -= tmp1 * i;
        share_dest[c0 * lat_s + 3] -= tmp0 * i;
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
            tmp0 += (src[index_fermi(tmp, y, z, t, 0, c1)] - src[index_fermi(tmp, y, z, t, 3, c1)] * i) * U[index_guage(x, y, z, t, 0, c0, c1)];
            tmp1 += (src[index_fermi(tmp, y, z, t, 1, c1)] - src[index_fermi(tmp, y, z, t, 2, c1)] * i) * U[index_guage(x, y, z, t, 0, c0, c1)];
        }
        share_dest[c0 * lat_s + 0] += tmp0;
        share_dest[c0 * lat_s + 1] += tmp1;
        share_dest[c0 * lat_s + 2] += tmp1 * i;
        share_dest[c0 * lat_s + 3] += tmp0 * i;
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
            tmp0 += (src[index_fermi(x, tmp, z, t, 0, c1)] - src[index_fermi(x, tmp, z, t, 3, c1)]) * U[index_guage(x, tmp, z, t, 1, c1, c0)].conj();
            tmp1 += (src[index_fermi(x, tmp, z, t, 1, c1)] + src[index_fermi(x, tmp, z, t, 2, c1)]) * U[index_guage(x, tmp, z, t, 1, c1, c0)].conj();
        }
        share_dest[c0 * lat_s + 0] += tmp0;
        share_dest[c0 * lat_s + 1] += tmp1;
        share_dest[c0 * lat_s + 2] += tmp1;
        share_dest[c0 * lat_s + 3] -= tmp0;
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
            tmp0 += (src[index_fermi(x, tmp, z, t, 0, c1)] + src[index_fermi(x, tmp, z, t, 3, c1)]) * U[index_guage(x, y, z, t, 1, c0, c1)];
            tmp1 += (src[index_fermi(x, tmp, z, t, 1, c1)] - src[index_fermi(x, tmp, z, t, 2, c1)]) * U[index_guage(x, y, z, t, 1, c0, c1)];
        }
        share_dest[c0 * lat_s + 0] += tmp0;
        share_dest[c0 * lat_s + 1] += tmp1;
        share_dest[c0 * lat_s + 2] -= tmp1;
        share_dest[c0 * lat_s + 3] += tmp0;
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
            tmp0 += (src[index_fermi(x, y, tmp, t, 0, c1)] + src[index_fermi(x, y, tmp, t, 2, c1)] * i) * U[index_guage(x, y, tmp, t, 2, c1, c0)].conj();
            tmp1 += (src[index_fermi(x, y, tmp, t, 1, c1)] - src[index_fermi(x, y, tmp, t, 3, c1)] * i) * U[index_guage(x, y, tmp, t, 2, c1, c0)].conj();
        }
        share_dest[c0 * lat_s + 0] += tmp0;
        share_dest[c0 * lat_s + 1] += tmp1;
        share_dest[c0 * lat_s + 2] -= tmp0 * i;
        share_dest[c0 * lat_s + 3] += tmp1 * i;
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
            tmp0 += (src[index_fermi(x, y, tmp, t, 0, c1)] - src[index_fermi(x, y, tmp, t, 2, c1)] * i) * U[index_guage(x, y, z, t, 2, c0, c1)];
            tmp1 += (src[index_fermi(x, y, tmp, t, 1, c1)] + src[index_fermi(x, y, tmp, t, 3, c1)] * i) * U[index_guage(x, y, z, t, 2, c0, c1)];
        }
        share_dest[c0 * lat_s + 0] += tmp0;
        share_dest[c0 * lat_s + 1] += tmp1;
        share_dest[c0 * lat_s + 2] += tmp0 * i;
        share_dest[c0 * lat_s + 3] -= tmp1 * i;
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
            tmp0 += (src[index_fermi(x, y, z, tmp, 0, c1)] + src[index_fermi(x, y, z, tmp, 2, c1)]) * U[index_guage(x, y, z, tmp, 3, c1, c0)].conj();
            tmp1 += (src[index_fermi(x, y, z, tmp, 1, c1)] + src[index_fermi(x, y, z, tmp, 3, c1)]) * U[index_guage(x, y, z, tmp, 3, c1, c0)].conj();
        }
        share_dest[c0 * lat_s + 0] += tmp0;
        share_dest[c0 * lat_s + 1] += tmp1;
        share_dest[c0 * lat_s + 2] += tmp0;
        share_dest[c0 * lat_s + 3] += tmp1;
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
            tmp0 += (src[index_fermi(x, y, z, tmp, 0, c1)] - src[index_fermi(x, y, z, tmp, 2, c1)]) * U[index_guage(x, y, z, t, 3, c0, c1)];
            tmp1 += (src[index_fermi(x, y, z, tmp, 1, c1)] - src[index_fermi(x, y, z, tmp, 3, c1)]) * U[index_guage(x, y, z, t, 3, c0, c1)];
        }
        share_dest[c0 * lat_s + 0] += tmp0;
        share_dest[c0 * lat_s + 1] += tmp1;
        share_dest[c0 * lat_s + 2] -= tmp0;
        share_dest[c0 * lat_s + 3] -= tmp1;
    }

    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        dest[index_fermi(x, y, z, t, 0, c0)] = share_dest[c0 * lat_s + 0];
        dest[index_fermi(x, y, z, t, 1, c0)] = share_dest[c0 * lat_s + 1];
        dest[index_fermi(x, y, z, t, 2, c0)] = share_dest[c0 * lat_s + 2];
        dest[index_fermi(x, y, z, t, 3, c0)] = share_dest[c0 * lat_s + 3];
    }
}
