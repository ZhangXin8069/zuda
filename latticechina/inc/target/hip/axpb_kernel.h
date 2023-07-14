//
// Created by louis on 2022/2/20.
//

#ifndef LATTICE_AXPB_KERNEL_H
#define LATTICE_AXPB_KERNEL_H

#include "hip/hip_runtime.h"

static __global__ void axpbyz_g(const double a, double *x_i, const double b, double *y_i, double *z_i,
                         const int s_x, const int s_y, const int s_z, const int s_t) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    for (int t = 0; t < s_t; t++) {
        int pos = s_x * s_y * s_z * t + s_x * s_y * z + s_x * y + x;
        double *x_t = x_i + pos * 12 * 2;
        double *y_t = y_i + pos * 12 * 2;
        double *z_t = z_i + pos * 12 * 2;
        for (int i = 0; i < 12; i++) {
            z_t[i * 2 + 0] = a * x_t[i * 2 + 0] + b * y_t[i * 2 + 0];
            z_t[i * 2 + 1] = a * x_t[i * 2 + 1] + b * y_t[i * 2 + 1];
        }
    }
}

static __global__ void axpbyz_g2(const double a, double *x, const double b, double *y, double *z) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    z[i] = a * x[i] + b * y[i];
}

static __global__ void axpbyczw_g2(const double a, double *x, const double b, double *y, const double c, double *z, double *w) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    w[i] = a * x[i] + b * y[i] + c * z[i];
}


static __device__ void axpbyz_g_d(const double a, double *x, const double b, double *y, double *z) {

    for (int i = 0; i < 12; i++) {
        z[i * 2 + 0] = a * x[i * 2 + 0] + b * y[i * 2 + 0];
        z[i * 2 + 1] = a * x[i * 2 + 1] + b * y[i * 2 + 1];
    }
}

#endif //LATTICE_AXPB_KERNEL_H
