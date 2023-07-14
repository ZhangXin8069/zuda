//
// Created by louis on 2022/2/20.
//

#ifndef LATTICE_OPERATOR_KERNEL_H
#define LATTICE_OPERATOR_KERNEL_H

#include "hip/hip_runtime.h"

static __global__ void equal(double *out, double *in,
                             const int s_x, const int s_y, const int s_z, const int s_t) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    for (int t = 0; t < s_t; t++) {
        int pos = s_x * s_y * s_z * t + s_x * s_y * z + s_x * y + x;
        for (int i = 0; i < 24; i++) {
            out[pos * 24 + i] = in[pos * 24 + i];
        }
    }
}

static __global__ void sign(int *out, double *in) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    out[x] = (in[x] > 0) ? 1 : -1;
}

static __global__ void mult(double *out, int *in) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    out[x] *= in[x / 2];
}

#endif //LATTICE_OPERATOR_KERNEL_H
