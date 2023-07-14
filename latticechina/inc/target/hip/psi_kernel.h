//
// Created by louis on 2022/2/20.
//

#ifndef LATTICE_PSI_KERNEL_H
#define LATTICE_PSI_KERNEL_H

#include "hip/hip_runtime.h"

static __global__ void psi_g5(double *src, const int s_x, const int s_y, const int s_z, const int s_t) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    for (int t = 0; t < s_t; t++) {
        double *src_d = src + (s_x * s_y * s_z * t +
                               s_x * s_y * z +
                               s_x * y +
                               x) * 12 * 2;


        for (int i = 0; i < 3; i++) {
//            src_d[0 * 3 * 2 + i * 2 + 0] = src_d[0 * 3 * 2 + i * 2 + 0];
//            src_d[0 * 3 * 2 + i * 2 + 1] = src_d[0 * 3 * 2 + i * 2 + 1];
//            src_d[1 * 3 * 2 + i * 2 + 0] = src_d[1 * 3 * 2 + i * 2 + 0];
//            src_d[1 * 3 * 2 + i * 2 + 1] = src_d[1 * 3 * 2 + i * 2 + 1];
            src_d[2 * 3 * 2 + i * 2 + 0] = -src_d[2 * 3 * 2 + i * 2 + 0];
            src_d[2 * 3 * 2 + i * 2 + 1] = -src_d[2 * 3 * 2 + i * 2 + 1];
            src_d[3 * 3 * 2 + i * 2 + 0] = -src_d[3 * 3 * 2 + i * 2 + 0];
            src_d[3 * 3 * 2 + i * 2 + 1] = -src_d[3 * 3 * 2 + i * 2 + 1];
        }
    }
}

__device__ void psi_g5_d(double *src) {

    for (int i = 0; i < 3; i++) {
        src[2 * 3 * 2 + i * 2 + 0] = -src[2 * 3 * 2 + i * 2 + 0];
        src[2 * 3 * 2 + i * 2 + 1] = -src[2 * 3 * 2 + i * 2 + 1];
        src[3 * 3 * 2 + i * 2 + 0] = -src[3 * 3 * 2 + i * 2 + 0];
        src[3 * 3 * 2 + i * 2 + 1] = -src[3 * 3 * 2 + i * 2 + 1];
    }
}

#endif //LATTICE_PSI_KERNEL_H
