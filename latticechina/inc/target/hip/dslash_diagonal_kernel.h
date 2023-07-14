//
// Created by louis on 2022/2/20.
//

#ifndef LATTICE_DSLASH_KERNEL_H
#define LATTICE_DSLASH_KERNEL_H

#include "hip/hip_runtime.h"

static __global__ void Dslash_d(double *src, double *dest,
                                const int s_x, const int s_y, const int s_z, const int s_t,
                                const double mass, const int cb) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    const double a = 4.0;
    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int s_x_cb = s_x >> 1;

    for (int t = 0; t < s_t; t++) {

        double *src_d = src + (s_x_cb * s_y * s_z * t +
                               s_x_cb * s_y * z +
                               s_x_cb * y +
                               x + cb * subgrid_vol_cb) * 12 * 2;

        double *dest_d = dest + (s_x_cb * s_y * s_z * t +
                                 s_x_cb * s_y * z +
                                 s_x_cb * y +
                                 x + cb * subgrid_vol_cb) * 12 * 2;

        for (int i = 0; i < 24; i++) {
            dest_d[i] = (a + mass) * src_d[i];
        }
    }
}

#endif //LATTICE_DSLASH_KERNEL_H
