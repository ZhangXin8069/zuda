//
// Created by louis on 2022/2/20.
//

#ifndef LATTICE_GHOST_OVERLAP_KERNEL_H
#define LATTICE_GHOST_OVERLAP_KERNEL_H

#include "hip/hip_runtime.h"
#include "hip/hip_complex.h"

/////////////////    for overlap    /////////////////////////////////
//
//                sink * gamma5 * b
//
/////////////////////////////////////////////////////////////////

static __global__ void ghost_x_f_abp5(double *src_f, double *dest, double *U,
                                      const int v_x, const int v_y, const int v_z, const int v_t,
                                      const int s_x, const int s_y, const int s_z, const int s_t,
                                      const int rank, const int cb, const int flag, const double b) {

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int N_sub[4] = {v_x / s_x, v_y / s_y, v_z / s_z, v_t / s_t};

    const double half = 0.5;

    const int x_p = ((rank / N_sub[0]) % N_sub[1]) * s_y +
                    ((rank / (N_sub[1] * N_sub[0])) % N_sub[2]) * s_z +
                    (rank / (N_sub[2] * N_sub[1] * N_sub[0])) * s_t;

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    int tp = xyz / (s_z * s_y);
    int z = (xyz / s_y) % s_z;
    int y = xyz % s_y;

    int t = (y + z + 2 * tp + x_p) % 2 == cb ? 2 * tp + 1 : 2 * tp;
    if (t >= s_t) { return; }

    const int s_x_cb = s_x >> 1;

    int cont = s_y * s_z * tp + s_y * z + y;

    int x = s_x_cb - 1;

    double tmp[2];
    double destE[24];
    double srcO[12];
    double AE[18];

    for (int i = 0; i < 24; i++) {
        destE[i] = 0;
    }

    for (int i = 0; i < 12; i++) {
        srcO[i] = src_f[cont * 6 * 2 + i];
    }

    for (int i = 0; i < 18; i++) {
        AE[i] = U[(s_x_cb * s_y * s_z * t +
                   s_x_cb * s_y * z +
                   s_x_cb * y +
                   x + cb * subgrid_vol_cb) * 9 * 2 + i];
    }

    for (int c1 = 0; c1 < 3; c1++) {
        for (int c2 = 0; c2 < 3; c2++) {

            tmp[0] = srcO[(0 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 0]
                     - srcO[(0 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 1];
            tmp[1] = srcO[(0 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 1]
                     + srcO[(0 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 0];

            destE[(0 * 3 + c1) * 2 + 0] += tmp[0] * b;
            destE[(0 * 3 + c1) * 2 + 1] += tmp[1] * b;
            destE[(3 * 3 + c1) * 2 + 0] += flag * tmp[1] * b;
            destE[(3 * 3 + c1) * 2 + 1] += -flag * tmp[0] * b;

            tmp[0] = srcO[(1 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 0]
                     - srcO[(1 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 1];
            tmp[1] = srcO[(1 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 1]
                     + srcO[(1 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 0];

            destE[(1 * 3 + c1) * 2 + 0] += tmp[0] * b;
            destE[(1 * 3 + c1) * 2 + 1] += tmp[1] * b;
            destE[(2 * 3 + c1) * 2 + 0] += flag * tmp[1] * b;
            destE[(2 * 3 + c1) * 2 + 1] += -flag * tmp[0] * b;

        }
    }

    for (int i = 0; i < 24; i++) {
        dest[(s_x_cb * s_y * s_z * t +
              s_x_cb * s_y * z +
              s_x_cb * y +
              x + cb * subgrid_vol_cb) * 12 * 2 + i] += destE[i];
    }

}


static __global__ void ghost_x_b_abp5(double *src_b, double *dest, double *U,
                                      const int v_x, const int v_y, const int v_z, const int v_t,
                                      const int s_x, const int s_y, const int s_z, const int s_t,
                                      const int rank, const int cb, const int flag, const double b) {

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int N_sub[4] = {v_x / s_x, v_y / s_y, v_z / s_z, v_t / s_t};

    const double half = 0.5;

    const int x_p = ((rank / N_sub[0]) % N_sub[1]) * s_y +
                    ((rank / (N_sub[1] * N_sub[0])) % N_sub[2]) * s_z +
                    (rank / (N_sub[2] * N_sub[1] * N_sub[0])) * s_t;

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    int tp = xyz / (s_z * s_y);
    int z = (xyz / s_y) % s_z;
    int y = xyz % s_y;

    int t = (y + z + 2 * tp + x_p) % 2 != cb ? 2 * tp + 1 : 2 * tp;
    if (t >= s_t) { return; }

    const int s_x_cb = s_x >> 1;

    int cont = s_y * s_z * tp + s_y * z + y;

    int x = 0;

    double srcO[12];

    for (int i = 0; i < 12; i++) {
        srcO[i] = src_b[cont * 6 * 2 + i];
    }

    double *destE = dest + (s_x_cb * s_y * s_z * t +
                            s_x_cb * s_y * z +
                            s_x_cb * y +
                            x + cb * subgrid_vol_cb) * 12 * 2;

    for (int c1 = 0; c1 < 3; c1++) {

        destE[(0 * 3 + c1) * 2 + 0] += srcO[(0 * 3 + c1) * 2 + 0] * b;
        destE[(0 * 3 + c1) * 2 + 1] += srcO[(0 * 3 + c1) * 2 + 1] * b;
        destE[(3 * 3 + c1) * 2 + 0] += -flag * srcO[(0 * 3 + c1) * 2 + 1] * b;
        destE[(3 * 3 + c1) * 2 + 1] += flag * srcO[(0 * 3 + c1) * 2 + 0] * b;
        destE[(1 * 3 + c1) * 2 + 0] += srcO[(1 * 3 + c1) * 2 + 0] * b;
        destE[(1 * 3 + c1) * 2 + 1] += srcO[(1 * 3 + c1) * 2 + 1] * b;
        destE[(2 * 3 + c1) * 2 + 0] += -flag * srcO[(1 * 3 + c1) * 2 + 1] * b;
        destE[(2 * 3 + c1) * 2 + 1] += flag * srcO[(1 * 3 + c1) * 2 + 0] * b;
    }
}

static __global__ void ghost_y_f_abp5(double *src_f, double *dest, double *U,
                                      const int v_x, const int v_y, const int v_z, const int v_t,
                                      const int s_x, const int s_y, const int s_z, const int s_t,
                                      const int rank, const int cb, const int flag, const double b) {

    const int s_x_cb = s_x >> 1;

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    const double half = 0.5;

    int y = s_y - 1;

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    int t = xyz / (s_z * s_x_cb);
    int z = (xyz / s_x_cb) % s_z;
    int x = xyz % s_x_cb;

    double tmp[2];
    double destE[24];
    double srcO[12];
    double AE[18];

    int cont = s_x_cb * s_z * t + s_x_cb * z + x;

    for (int i = 0; i < 24; i++) {
        destE[i] = 0;
    }

    for (int i = 0; i < 12; i++) {
        srcO[i] = src_f[cont * 6 * 2 + i];
    }

    for (int i = 0; i < 18; i++) {
        AE[i] = U[(s_x_cb * s_y * s_z * t +
                   s_x_cb * s_y * z +
                   s_x_cb * y +
                   x + cb * subgrid_vol_cb) * 9 * 2 + i];
    }

    for (int c1 = 0; c1 < 3; c1++) {
        for (int c2 = 0; c2 < 3; c2++) {

            tmp[0] = +srcO[(0 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 0]
                     - srcO[(0 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 1];

            tmp[1] = +srcO[(0 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 1]
                     + srcO[(0 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 0];

            destE[(0 * 3 + c1) * 2 + 0] += tmp[0] * b;
            destE[(0 * 3 + c1) * 2 + 1] += tmp[1] * b;
            destE[(3 * 3 + c1) * 2 + 0] += -flag * tmp[0] * b;
            destE[(3 * 3 + c1) * 2 + 1] += -flag * tmp[1] * b;

            tmp[0] = +srcO[(1 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 0]
                     - srcO[(1 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 1];

            tmp[1] = +srcO[(1 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 1]
                     + srcO[(1 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 0];

            destE[(1 * 3 + c1) * 2 + 0] += tmp[0] * b;
            destE[(1 * 3 + c1) * 2 + 1] += tmp[1] * b;
            destE[(2 * 3 + c1) * 2 + 0] += flag * tmp[0] * b;
            destE[(2 * 3 + c1) * 2 + 1] += flag * tmp[1] * b;

        }
    }

    for (int i = 0; i < 24; i++) {
        dest[(s_x_cb * s_y * s_z * t +
              s_x_cb * s_y * z +
              s_x_cb * y +
              x + cb * subgrid_vol_cb) * 12 * 2 + i] += destE[i];
    }

}

static __global__ void ghost_y_b_abp5(double *src_b, double *dest, double *U,
                                      const int v_x, const int v_y, const int v_z, const int v_t,
                                      const int s_x, const int s_y, const int s_z, const int s_t,
                                      const int rank, const int cb, const int flag, const double b) {

    const int s_x_cb = s_x >> 1;

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    const double half = 0.5;

    int y = 0;

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    int t = xyz / (s_z * s_x_cb);
    int z = (xyz / s_x_cb) % s_z;
    int x = xyz % s_x_cb;

    int cont = s_x_cb * s_z * t + s_x_cb * z + x;

    double srcO[12];

    for (int i = 0; i < 12; i++) {
        srcO[i] = src_b[cont * 6 * 2 + i];
    }

    double *destE = dest + (s_x_cb * s_y * s_z * t +
                            s_x_cb * s_y * z +
                            s_x_cb * y +
                            x + cb * subgrid_vol_cb) * 12 * 2;

    for (int c1 = 0; c1 < 3; c1++) {

        destE[(0 * 3 + c1) * 2 + 0] += srcO[(0 * 3 + c1) * 2 + 0] * b;
        destE[(0 * 3 + c1) * 2 + 1] += srcO[(0 * 3 + c1) * 2 + 1] * b;
        destE[(3 * 3 + c1) * 2 + 0] += flag * srcO[(0 * 3 + c1) * 2 + 0] * b;
        destE[(3 * 3 + c1) * 2 + 1] += flag * srcO[(0 * 3 + c1) * 2 + 1] * b;
        destE[(1 * 3 + c1) * 2 + 0] += srcO[(1 * 3 + c1) * 2 + 0] * b;
        destE[(1 * 3 + c1) * 2 + 1] += srcO[(1 * 3 + c1) * 2 + 1] * b;
        destE[(2 * 3 + c1) * 2 + 0] += -flag * srcO[(1 * 3 + c1) * 2 + 0] * b;
        destE[(2 * 3 + c1) * 2 + 1] += -flag * srcO[(1 * 3 + c1) * 2 + 1] * b;
    }
}

static __global__ void ghost_z_f_abp5(double *src_f, double *dest, double *U,
                                      const int v_x, const int v_y, const int v_z, const int v_t,
                                      const int s_x, const int s_y, const int s_z, const int s_t,
                                      const int rank, const int cb, const int flag, const double b) {

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    const int s_x_cb = s_x >> 1;

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    const double half = 0.5;

    int z = s_z - 1;

    int t = xyz / (s_y * s_x_cb);
    int y = (xyz / s_x_cb) % s_y;
    int x = xyz % s_x_cb;

    int cont = s_x_cb * s_y * t + s_x_cb * y + x;

    double tmp[2];
    double destE[24];
    double srcO[12];
    double AE[18];

    for (int i = 0; i < 24; i++) {
        destE[i] = 0;
    }

    for (int i = 0; i < 12; i++) {
        srcO[i] = src_f[cont * 6 * 2 + i];
    }

    for (int i = 0; i < 18; i++) {
        AE[i] = U[(s_x_cb * s_y * s_z * t +
                   s_x_cb * s_y * z +
                   s_x_cb * y +
                   x + cb * subgrid_vol_cb) * 9 * 2 + i];
    }

    for (int c1 = 0; c1 < 3; c1++) {
        for (int c2 = 0; c2 < 3; c2++) {

            tmp[0] = +srcO[(0 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 0]
                     - srcO[(0 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 1];

            tmp[1] = +srcO[(0 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 1]
                     + srcO[(0 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 0];

            destE[(0 * 3 + c1) * 2 + 0] += tmp[0] * b;
            destE[(0 * 3 + c1) * 2 + 1] += tmp[1] * b;
            destE[(2 * 3 + c1) * 2 + 0] += flag * tmp[1] * b;
            destE[(2 * 3 + c1) * 2 + 1] += -flag * tmp[0] * b;

            tmp[0] = +srcO[(1 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 0]
                     - srcO[(1 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 1];

            tmp[1] = + srcO[(1 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 1]
                     + srcO[(1 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 0];

            destE[(1 * 3 + c1) * 2 + 0] += tmp[0] * b;
            destE[(1 * 3 + c1) * 2 + 1] += tmp[1] * b;
            destE[(3 * 3 + c1) * 2 + 0] +=-flag * tmp[1] * b;
            destE[(3 * 3 + c1) * 2 + 1] += flag * tmp[0] * b;

        }
    }

    for (int i = 0; i < 24; i++) {
        dest[(s_x_cb * s_y * s_z * t +
              s_x_cb * s_y * z +
              s_x_cb * y +
              x + cb * subgrid_vol_cb) * 12 * 2 + i] += destE[i];
    }

}

static __global__ void ghost_z_b_abp5(double *src_b, double *dest, double *U,
                                      const int v_x, const int v_y, const int v_z, const int v_t,
                                      const int s_x, const int s_y, const int s_z, const int s_t,
                                      const int rank, const int cb, const int flag, const double b) {

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    const int s_x_cb = s_x >> 1;

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    const double half = 0.5;

    int z = 0;

    int t = xyz / (s_y * s_x_cb);
    int y = (xyz / s_x_cb) % s_y;
    int x = xyz % s_x_cb;

    int cont = s_x_cb * s_y * t + s_x_cb * y + x;

    double srcO[12];

    for (int i = 0; i < 12; i++) {
        srcO[i] = src_b[cont * 6 * 2 + i];
    }

    double *destE = dest + (s_x_cb * s_y * s_z * t +
                            s_x_cb * s_y * z +
                            s_x_cb * y +
                            x + cb * subgrid_vol_cb) * 12 * 2;

    for (int c1 = 0; c1 < 3; c1++) {
        destE[(0 * 3 + c1) * 2 + 0] += srcO[(0 * 3 + c1) * 2 + 0] * b;
        destE[(0 * 3 + c1) * 2 + 1] += srcO[(0 * 3 + c1) * 2 + 1] * b;

        destE[(2 * 3 + c1) * 2 + 0] += -flag * srcO[(0 * 3 + c1) * 2 + 1] * b;
        destE[(2 * 3 + c1) * 2 + 1] += flag * srcO[(0 * 3 + c1) * 2 + 0] * b;

        destE[(1 * 3 + c1) * 2 + 0] += srcO[(1 * 3 + c1) * 2 + 0] * b;
        destE[(1 * 3 + c1) * 2 + 1] += srcO[(1 * 3 + c1) * 2 + 1] * b;

        destE[(3 * 3 + c1) * 2 + 0] += flag * srcO[(1 * 3 + c1) * 2 + 1] * b;
        destE[(3 * 3 + c1) * 2 + 1] += -flag * srcO[(1 * 3 + c1) * 2 + 0] * b;

    }
}

static __global__ void ghost_t_f_abp5(double *src_f, double *dest, double *U,
                                      const int v_x, const int v_y, const int v_z, const int v_t,
                                      const int s_x, const int s_y, const int s_z, const int s_t,
                                      const int rank, const int cb, const int flag, const double b) {

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    const int s_x_cb = s_x >> 1;

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int t = s_t - 1;

    int z = xyz / (s_y * s_x_cb);
    int y = (xyz / s_x_cb) % s_y;
    int x = xyz % s_x_cb;

    double tmp[2];
    double destE[24];
    double srcO[12];
    double AE[18];

    int cont = s_x_cb * s_y * z + s_x_cb * y + x;

    for (int i = 0; i < 24; i++) {
        destE[i] = 0;
    }

    for (int i = 0; i < 12; i++) {
        srcO[i] = src_f[cont * 6 * 2 + i];
    }

    for (int i = 0; i < 18; i++) {
        AE[i] = U[(s_x_cb * s_y * s_z * t +
                   s_x_cb * s_y * z +
                   s_x_cb * y +
                   x + cb * subgrid_vol_cb) * 9 * 2 + i];
    }

    for (int c1 = 0; c1 < 3; c1++) {
        for (int c2 = 0; c2 < 3; c2++) {

            tmp[0] = +srcO[(0 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 0]
                     - srcO[(0 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 1];

            tmp[1] = +srcO[(0 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 1]
                     + srcO[(0 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 0];

            destE[(0 * 3 + c1) * 2 + 0] += tmp[0] * b;
            destE[(0 * 3 + c1) * 2 + 1] += tmp[1] * b;
            destE[(2 * 3 + c1) * 2 + 0] += flag * tmp[0] * b;
            destE[(2 * 3 + c1) * 2 + 1] += flag * tmp[1] * b;

            tmp[0] = +srcO[(1 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 0]
                     - srcO[(1 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 1];

            tmp[1] = +srcO[(1 * 3 + c2) * 2 + 0] * AE[(c1 * 3 + c2) * 2 + 1]
                     + srcO[(1 * 3 + c2) * 2 + 1] * AE[(c1 * 3 + c2) * 2 + 0];

            destE[(1 * 3 + c1) * 2 + 0] += tmp[0] * b;
            destE[(1 * 3 + c1) * 2 + 1] += tmp[1] * b;
            destE[(3 * 3 + c1) * 2 + 0] += flag * tmp[0] * b;
            destE[(3 * 3 + c1) * 2 + 1] += flag * tmp[1] * b;

        }
    }
/*
    for (int i = 0; i < 3; i++) {
        destE[0 * 3 * 2 + i * 2 + 0] = destE[0 * 3 * 2 + i * 2 + 0] * b;
        destE[0 * 3 * 2 + i * 2 + 1] = destE[0 * 3 * 2 + i * 2 + 1] * b;
        destE[1 * 3 * 2 + i * 2 + 0] = destE[1 * 3 * 2 + i * 2 + 0] * b;
        destE[1 * 3 * 2 + i * 2 + 1] = destE[1 * 3 * 2 + i * 2 + 1] * b;

        destE[2 * 3 * 2 + i * 2 + 0] = -destE[2 * 3 * 2 + i * 2 + 0] * b;
        destE[2 * 3 * 2 + i * 2 + 1] = -destE[2 * 3 * 2 + i * 2 + 1] * b;
        destE[3 * 3 * 2 + i * 2 + 0] = -destE[3 * 3 * 2 + i * 2 + 0] * b;
        destE[3 * 3 * 2 + i * 2 + 1] = -destE[3 * 3 * 2 + i * 2 + 1] * b;
    }
*/
    for (int i = 0; i < 24; i++) {
        dest[(s_x_cb * s_y * s_z * t +
              s_x_cb * s_y * z +
              s_x_cb * y +
              x + cb * subgrid_vol_cb) * 12 * 2 + i] += destE[i];
    }
}

static __global__ void ghost_t_b_abp5(double *src_b, double *dest, double *U,
                                      const int v_x, const int v_y, const int v_z, const int v_t,
                                      const int s_x, const int s_y, const int s_z, const int s_t,
                                      const int rank, const int cb, const int flag, const double b) {

    const int s_x_cb = s_x >> 1;

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int t = 0;

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    int z = xyz / (s_y * s_x_cb);
    int y = (xyz / s_x_cb) % s_y;
    int x = xyz % s_x_cb;

    int cont = s_x_cb * s_y * z + s_x_cb * y + x;

    double srcO[12];

    for (int i = 0; i < 12; i++) {
        srcO[i] = src_b[cont * 6 * 2 + i];
    }

    double *destE = dest + (s_x_cb * s_y * s_z * t +
                            s_x_cb * s_y * z +
                            s_x_cb * y +
                            x + cb * subgrid_vol_cb) * 12 * 2;

    for (int c1 = 0; c1 < 3; c1++) {
        destE[(0 * 3 + c1) * 2 + 0] += srcO[(0 * 3 + c1) * 2 + 0] * b;
        destE[(0 * 3 + c1) * 2 + 1] += srcO[(0 * 3 + c1) * 2 + 1] * b;
        destE[(2 * 3 + c1) * 2 + 0] += -flag * srcO[(0 * 3 + c1) * 2 + 0] * b;
        destE[(2 * 3 + c1) * 2 + 1] += -flag * srcO[(0 * 3 + c1) * 2 + 1] * b;
        destE[(1 * 3 + c1) * 2 + 0] += srcO[(1 * 3 + c1) * 2 + 0] * b;
        destE[(1 * 3 + c1) * 2 + 1] += srcO[(1 * 3 + c1) * 2 + 1] * b;
        destE[(3 * 3 + c1) * 2 + 0] += -flag * srcO[(1 * 3 + c1) * 2 + 0] * b;
        destE[(3 * 3 + c1) * 2 + 1] += -flag * srcO[(1 * 3 + c1) * 2 + 1] * b;
    }
}

#endif //LATTICE_GHOST_OVERLAP_KERNEL_H
