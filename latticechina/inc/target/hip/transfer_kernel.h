//
// Created by louis on 2022/2/20.
//

#ifndef LATTICE_TRANSFER_KERNEL_H
#define LATTICE_TRANSFER_KERNEL_H

#include "hip/hip_runtime.h"
#include "hip/hip_complex.h"

static __global__ void transfer_x_f(double *src, double *dest_b, double *U,
                                    const int v_x, const int v_y, const int v_z, const int v_t,
                                    const int s_x, const int s_y, const int s_z, const int s_t,
                                    const int rank, const int cb, const int flag) {
/*
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.y * blockIdx.y + threadIdx.y;
    int tp = blockDim.z * blockIdx.z + threadIdx.z;
*/

//    if (y >= s_y || z >= s_z) { return; }

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int N_sub[4] = {
            v_x / s_x,
            v_y / s_y,
            v_z / s_z,
            v_t / s_t
    };

    const double half = 0.5;
//    const hipDoubleComplex I(0, 1);

    const int x_p = ((rank / N_sub[0]) % N_sub[1]) * s_y +
                    ((rank / (N_sub[1] * N_sub[0])) % N_sub[2]) * s_z +
                    (rank / (N_sub[2] * N_sub[1] * N_sub[0])) * s_t;

    const int s_x_cb = s_x >> 1;

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    int tp = xyz / (s_z * s_y);
    int z = (xyz / s_y) % s_z;
    int y = xyz % s_y;

    int t = (y + z + 2 * tp + x_p) % 2 == cb ? 2 * tp + 1 : 2 * tp;
    if (t >= s_t) { return; }

    int cont = s_y * s_z * tp + s_y * z + y;

    int x = 0;

    double tmp[2];

    double *srcO = src + (s_x_cb * s_y * s_z * t +
                          s_x_cb * s_y * z +
                          s_x_cb * y +
                          x + (1 - cb) * subgrid_vol_cb) * 12 * 2;

//    hipDoubleComplex tmp;
/*
    hipDoubleComplex *srcO = (hipDoubleComplex *) (src + (s_x_cb * s_y * s_z * t +
                                                          s_x_cb * s_y * z +
                                                          s_x_cb * y + x +
                                                          (1 - cb) * subgrid_vol_cb) * 12 * 2);
*/
    for (int c2 = 0; c2 < 3; c2++) {
/*
        tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half;
        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 0] = tmp.x;
        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 1] = tmp.y;
        tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half;
        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 0] = tmp.x;
        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 1] = tmp.y;
*/
        tmp[0] = -(srcO[(0 * 3 + c2) * 2 + 0] + flag * srcO[(3 * 3 + c2) * 2 + 1]) * half;
        tmp[1] = -(srcO[(0 * 3 + c2) * 2 + 1] - flag * srcO[(3 * 3 + c2) * 2 + 0]) * half;
        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 0] = tmp[0];
        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 1] = tmp[1];
        tmp[0] = -(srcO[(1 * 3 + c2) * 2 + 0] + flag * srcO[(2 * 3 + c2) * 2 + 1]) * half;
        tmp[1] = -(srcO[(1 * 3 + c2) * 2 + 1] - flag * srcO[(2 * 3 + c2) * 2 + 0]) * half;
        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 0] = tmp[0];
        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 1] = tmp[1];

    }
/*
    tmp = -(srcO[0 * 3 + 0] - flag * I * srcO[3 * 3 + 0]) * half;
    dest_b[cont * 6 * 2 + (0 * 3 + 0) * 2 + 0] = tmp.x;
    dest_b[cont * 6 * 2 + (0 * 3 + 0) * 2 + 1] = tmp.y;
    tmp = -(srcO[1 * 3 + 0] - flag * I * srcO[2 * 3 + 0]) * half;
    dest_b[cont * 6 * 2 + (1 * 3 + 0) * 2 + 0] = tmp.x;
    dest_b[cont * 6 * 2 + (1 * 3 + 0) * 2 + 1] = tmp.y;

    tmp = -(srcO[0 * 3 + 1] - flag * I * srcO[3 * 3 + 1]) * half;
    dest_b[cont * 6 * 2 + (0 * 3 + 1) * 2 + 0] = tmp.x;
    dest_b[cont * 6 * 2 + (0 * 3 + 1) * 2 + 1] = tmp.y;
    tmp = -(srcO[1 * 3 + 1] - flag * I * srcO[2 * 3 + 1]) * half;
    dest_b[cont * 6 * 2 + (1 * 3 + 1) * 2 + 0] = tmp.x;
    dest_b[cont * 6 * 2 + (1 * 3 + 1) * 2 + 1] = tmp.y;

    tmp = -(srcO[0 * 3 + 2] - flag * I * srcO[3 * 3 + 2]) * half;
    dest_b[cont * 6 * 2 + (0 * 3 + 2) * 2 + 0] = tmp.x;
    dest_b[cont * 6 * 2 + (0 * 3 + 2) * 2 + 1] = tmp.y;
    tmp = -(srcO[1 * 3 + 2] - flag * I * srcO[2 * 3 + 2]) * half;
    dest_b[cont * 6 * 2 + (1 * 3 + 2) * 2 + 0] = tmp.x;
    dest_b[cont * 6 * 2 + (1 * 3 + 2) * 2 + 1] = tmp.y;
*/
}

static __global__ void transfer_x_b(double *src, double *dest_f, double *U,
                                    const int v_x, const int v_y, const int v_z, const int v_t,
                                    const int s_x, const int s_y, const int s_z, const int s_t,
                                    const int rank, const int cb, const int flag) {
/*
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.y * blockIdx.y + threadIdx.y;
    int tp = blockDim.z * blockIdx.z + threadIdx.z;
*/

//  if (y >= s_y || z >= s_z) { return; }

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int N_sub[4] = {v_x / s_x, v_y / s_y, v_z / s_z, v_t / s_t};

    const double half = 0.5;
//    const hipDoubleComplex I(0, 1);

    const int x_p = ((rank / N_sub[0]) % N_sub[1]) * s_y +
                    ((rank / (N_sub[1] * N_sub[0])) % N_sub[2]) * s_z +
                    (rank / (N_sub[2] * N_sub[1] * N_sub[0])) * s_t;

    const int s_x_cb = s_x >> 1;

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    int tp = xyz / (s_z * s_y);
    int z = (xyz / s_y) % s_z;
    int y = xyz % s_y;

    int t = (y + z + 2 * tp + x_p) % 2 != cb ? 2 * tp + 1 : 2 * tp;
    if (t >= s_t) { return; }

    int cont = s_y * s_z * tp + s_y * z + y;

    int x = s_x_cb - 1;

/*
    hipDoubleComplex tmp;
    hipDoubleComplex *AO = (hipDoubleComplex *) (U + (s_x_cb * s_y * s_z * t +
                                                      s_x_cb * s_y * z +
                                                      s_x_cb * y +
                                                      x + (1 - cb) * subgrid_vol_cb) * 9 * 2);

    hipDoubleComplex *srcO = (hipDoubleComplex *) (src + (s_x_cb * s_y * s_z * t +
                                                          s_x_cb * s_y * z +
                                                          s_x_cb * y + x +
                                                          (1 - cb) * subgrid_vol_cb) * 12 * 2);
*/

    double tmp[2];
    double destE[12];
    double srcO[24];
    double AO[18];

    for (int i = 0; i < 24; i++) {

        srcO[i] = src[(s_x_cb * s_y * s_z * t +
                       s_x_cb * s_y * z +
                       s_x_cb * y +
                       x + (1 - cb) * subgrid_vol_cb) * 12 * 2 + i];

    }

    for (int i = 0; i < 12; i++) {

        destE[i] = 0;

    }

    for (int i = 0; i < 18; i++) {

        AO[i] = U[(s_x_cb * s_y * s_z * t +
                   s_x_cb * s_y * z +
                   s_x_cb * y +
                   x + (1 - cb) * subgrid_vol_cb) * 9 * 2 + i];

    }

    for (int c1 = 0; c1 < 3; c1++) {
        for (int c2 = 0; c2 < 3; c2++) {
/*
            tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half * hipConj(AO[c2 * 3 + c1]);

            dest_f[cont * 6 * 2 + (0 * 3 + c1) * 2 + 0] += tmp.x;
            dest_f[cont * 6 * 2 + (0 * 3 + c1) * 2 + 1] += tmp.y;

            tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half * hipConj(AO[c2 * 3 + c1]);

            dest_f[cont * 6 * 2 + (1 * 3 + c1) * 2 + 0] += tmp.x;
            dest_f[cont * 6 * 2 + (1 * 3 + c1) * 2 + 1] += tmp.y;
*/
            tmp[0] = -(srcO[(0 * 3 + c2) * 2 + 0] - flag * srcO[(3 * 3 + c2) * 2 + 1]) * half * AO[(c2 * 3 + c1) * 2 + 0]
                     -(srcO[(0 * 3 + c2) * 2 + 1] + flag * srcO[(3 * 3 + c2) * 2 + 0]) * half * AO[(c2 * 3 + c1) * 2 + 1];

            tmp[1] = +(srcO[(0 * 3 + c2) * 2 + 0] - flag * srcO[(3 * 3 + c2) * 2 + 1]) * half * AO[(c2 * 3 + c1) * 2 + 1]
                     -(srcO[(0 * 3 + c2) * 2 + 1] + flag * srcO[(3 * 3 + c2) * 2 + 0]) * half * AO[(c2 * 3 + c1) * 2 + 0];

            destE[(0 * 3 + c1) * 2 + 0] += tmp[0];
            destE[(0 * 3 + c1) * 2 + 1] += tmp[1];

            tmp[0] = -(srcO[(1 * 3 + c2) * 2 + 0] - flag * srcO[(2 * 3 + c2) * 2 + 1]) * half * AO[(c2 * 3 + c1) * 2 + 0]
                     -(srcO[(1 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 0]) * half * AO[(c2 * 3 + c1) * 2 + 1];

            tmp[1] = +(srcO[(1 * 3 + c2) * 2 + 0] - flag * srcO[(2 * 3 + c2) * 2 + 1]) * half * AO[(c2 * 3 + c1) * 2 + 1]
                     -(srcO[(1 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 0]) * half * AO[(c2 * 3 + c1) * 2 + 0];

            destE[(1 * 3 + c1) * 2 + 0] += tmp[0];
            destE[(1 * 3 + c1) * 2 + 1] += tmp[1];

        }
    }

    for (int i = 0; i < 12; i++) {
        dest_f[cont * 6 * 2 + i] = destE[i];
    }
}

static __global__ void transfer_y_f(double *src, double *dest_b, double *U,
                                    const int v_x, const int v_y, const int v_z, const int v_t,
                                    const int s_x, const int s_y, const int s_z, const int s_t,
                                    const int rank, const int cb, const int flag) {
/*
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.y * blockIdx.y + threadIdx.y;
    int t = blockDim.z * blockIdx.z + threadIdx.z;
*/
    const int s_x_cb = s_x >> 1;

//    if (x >= s_x_cb || z >= s_z || t >= s_t) { return; }

    const double half = 0.5;
//    const hipDoubleComplex I(0, 1);

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int y = 0;

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    int t = xyz / (s_z * s_x_cb);
    int z = (xyz / s_x_cb) % s_z;
    int x = xyz % s_x_cb;

    double tmp[2];

/*
    hipDoubleComplex tmp;
    hipDoubleComplex *srcO = (hipDoubleComplex *) (src + (s_x_cb * s_y * s_z * t +
                                                          s_x_cb * s_y * z +
                                                          s_x_cb * y +
                                                          x + (1 - cb) * subgrid_vol_cb) * 12 * 2);
*/

    double *srcO = src + (s_x_cb * s_y * s_z * t +
                          s_x_cb * s_y * z +
                          s_x_cb * y +
                          x + (1 - cb) * subgrid_vol_cb) * 12 * 2;

    int cont = s_x_cb * s_z * t + s_x_cb * z + x;

    for (int c2 = 0; c2 < 3; c2++) {
/*
        tmp = -(srcO[0 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half;
        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 0] = tmp.x;
        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 1] = tmp.y;
        tmp = -(srcO[1 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half;
        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 0] = tmp.x;
        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 1] = tmp.y;
*/

        tmp[0] = -(srcO[(0 * 3 + c2) * 2 + 0] + flag * srcO[(3 * 3 + c2) * 2 + 0]) * half;
        tmp[1] = -(srcO[(0 * 3 + c2) * 2 + 1] + flag * srcO[(3 * 3 + c2) * 2 + 1]) * half;

        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 0] = tmp[0];
        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 1] = tmp[1];

        tmp[0] = -(srcO[(1 * 3 + c2) * 2 + 0] - flag * srcO[(2 * 3 + c2) * 2 + 0]) * half;
        tmp[1] = -(srcO[(1 * 3 + c2) * 2 + 1] - flag * srcO[(2 * 3 + c2) * 2 + 1]) * half;

        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 0] = tmp[0];
        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 1] = tmp[1];

    }
}

static __global__ void transfer_y_b(double *src, double *dest_f, double *U,
                                    const int v_x, const int v_y, const int v_z, const int v_t,
                                    const int s_x, const int s_y, const int s_z, const int s_t,
                                    const int rank, const int cb, const int flag) {
/*
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.y * blockIdx.y + threadIdx.y;
    int t = blockDim.z * blockIdx.z + threadIdx.z;
*/

    const int s_x_cb = s_x >> 1;

//    if (x >= s_x_cb || z >= s_z || t >= s_t) { return; }

    const double half = 0.5;
//    const hipDoubleComplex I(0, 1);

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

//    hipDoubleComplex tmp;

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    int y = s_y - 1;

    int t = xyz / (s_z * s_x_cb);
    int z = (xyz / s_x_cb) % s_z;
    int x = xyz % s_x_cb;
/*
    hipDoubleComplex *srcO = (hipDoubleComplex *) (src + (s_x_cb * s_y * s_z * t +
                                                          s_x_cb * s_y * z +
                                                          s_x_cb * y +
                                                          x + (1 - cb) * subgrid_vol_cb) * 12 * 2);

    hipDoubleComplex *AO = (hipDoubleComplex *) (U + (s_x_cb * s_y * s_z * t +
                                                      s_x_cb * s_y * z +
                                                      s_x_cb * y +
                                                      x + (1 - cb) * subgrid_vol_cb) * 9 * 2);
*/
    double tmp[2];
    double destE[12];
    double srcO[24];
    double AO[18];

    for (int i = 0; i < 24; i++) {

        srcO[i] = src[(s_x_cb * s_y * s_z * t +
                       s_x_cb * s_y * z +
                       s_x_cb * y +
                       x + (1 - cb) * subgrid_vol_cb) * 12 * 2 + i];

    }

    for (int i = 0; i < 12; i++) {

        destE[i] = 0;

    }

    for (int i = 0; i < 18; i++) {

        AO[i] = U[(s_x_cb * s_y * s_z * t +
                   s_x_cb * s_y * z +
                   s_x_cb * y +
                   x + (1 - cb) * subgrid_vol_cb) * 9 * 2 + i];

    }

    int cont = s_x_cb * s_z * t + s_x_cb * z + x;
    for (int c1 = 0; c1 < 3; c1++) {
        for (int c2 = 0; c2 < 3; c2++) {
/*
            tmp = -(srcO[0 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half * hipConj(AO[c2 * 3 + c1]);
            dest_f[cont * 6 * 2 + (0 * 3 + c1) * 2 + 0] += tmp.x;
            dest_f[cont * 6 * 2 + (0 * 3 + c1) * 2 + 1] += tmp.y;
            tmp = -(srcO[1 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half * hipConj(AO[c2 * 3 + c1]);
            dest_f[cont * 6 * 2 + (1 * 3 + c1) * 2 + 0] += tmp.x;
            dest_f[cont * 6 * 2 + (1 * 3 + c1) * 2 + 1] += tmp.y;
*/

            tmp[0] = -(srcO[(0 * 3 + c2) * 2 + 0] - flag * srcO[(3 * 3 + c2) * 2 + 0]) * half * AO[(c2 * 3 + c1) * 2 + 0]
                     -(srcO[(0 * 3 + c2) * 2 + 1] - flag * srcO[(3 * 3 + c2) * 2 + 1]) * half * AO[(c2 * 3 + c1) * 2 + 1];
            tmp[1] = +(srcO[(0 * 3 + c2) * 2 + 0] - flag * srcO[(3 * 3 + c2) * 2 + 0]) * half * AO[(c2 * 3 + c1) * 2 + 1]
                     -(srcO[(0 * 3 + c2) * 2 + 1] - flag * srcO[(3 * 3 + c2) * 2 + 1]) * half * AO[(c2 * 3 + c1) * 2 + 0];

            destE[(0 * 3 + c1) * 2 + 0] += tmp[0];
            destE[(0 * 3 + c1) * 2 + 1] += tmp[1];

            tmp[0] = -(srcO[(1 * 3 + c2) * 2 + 0] + flag * srcO[(2 * 3 + c2) * 2 + 0]) * half * AO[(c2 * 3 + c1) * 2 + 0]
                     -(srcO[(1 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 1]) * half * AO[(c2 * 3 + c1) * 2 + 1];
            tmp[1] = +(srcO[(1 * 3 + c2) * 2 + 0] + flag * srcO[(2 * 3 + c2) * 2 + 0]) * half * AO[(c2 * 3 + c1) * 2 + 1]
                     -(srcO[(1 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 1]) * half * AO[(c2 * 3 + c1) * 2 + 0];

            destE[(1 * 3 + c1) * 2 + 0] += tmp[0];
            destE[(1 * 3 + c1) * 2 + 1] += tmp[1];

        }
    }

    for (int i = 0; i < 12; i++) {
        dest_f[cont * 6 * 2 + i] = destE[i];
    }
}

static __global__ void transfer_z_f(double *src, double *dest_b, double *U,
                                    const int v_x, const int v_y, const int v_z, const int v_t,
                                    const int s_x, const int s_y, const int s_z, const int s_t,
                                    const int rank, const int cb, const int flag) {

//    int x = blockDim.x * blockIdx.x + threadIdx.x;
//    int y = blockDim.y * blockIdx.y + threadIdx.y;
//    int t = blockDim.z * blockIdx.z + threadIdx.z;

    const int s_x_cb = s_x >> 1;

    const double half = 0.5;
//    const hipDoubleComplex I(0, 1);

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    int t = xyz / (s_y * s_x_cb);
    int y = (xyz / s_x_cb) % s_y;
    int x = xyz % s_x_cb;

    int z = 0;
//    hipDoubleComplex tmp;

    double *srcO = src + (s_x_cb * s_y * s_z * t +
                          s_x_cb * s_y * z +
                          s_x_cb * y +
                          x + (1 - cb) * subgrid_vol_cb) * 12 * 2;

    double tmp[2];

    int cont = s_x_cb * s_y * t + s_x_cb * y + x;

    for (int c2 = 0; c2 < 3; c2++) {
/*
        tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half;
        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 0] += tmp.x;
        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 1] += tmp.y;
        tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half;
        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 0] += tmp.x;
        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 1] += tmp.y;
*/
        tmp[0] = -(srcO[(0 * 3 + c2) * 2 + 0] + flag * srcO[(2 * 3 + c2) * 2 + 1]) * half;
        tmp[1] = -(srcO[(0 * 3 + c2) * 2 + 1] - flag * srcO[(2 * 3 + c2) * 2 + 0]) * half;

        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 0] = tmp[0];
        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 1] = tmp[1];

        tmp[0] = -(srcO[(1 * 3 + c2) * 2 + 0] - flag * srcO[(3 * 3 + c2) * 2 + 1]) * half;
        tmp[1] = -(srcO[(1 * 3 + c2) * 2 + 1] + flag * srcO[(3 * 3 + c2) * 2 + 0]) * half;

        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 0] = tmp[0];
        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 1] = tmp[1];
    }
}

static __global__ void transfer_z_b(double *src, double *dest_f, double *U,
                                    const int v_x, const int v_y, const int v_z, const int v_t,
                                    const int s_x, const int s_y, const int s_z, const int s_t,
                                    const int rank, const int cb, const int flag) {

//    int x = blockDim.x * blockIdx.x + threadIdx.x;
//    int y = blockDim.y * blockIdx.y + threadIdx.y;
//    int t = blockDim.z * blockIdx.z + threadIdx.z;

    const int s_x_cb = s_x >> 1;

    const double half = 0.5;
//    const hipDoubleComplex I(0, 1);

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    int t = xyz / (s_y * s_x_cb);
    int y = (xyz / s_x_cb) % s_y;
    int x = xyz % s_x_cb;

//    hipDoubleComplex tmp;

    int z = s_z - 1;
/*
    hipDoubleComplex *srcO = (hipDoubleComplex *) (src + (s_x_cb * s_y * s_z * t +
                                                          s_x_cb * s_y * z +
                                                          s_x_cb * y +
                                                          x + (1 - cb) * subgrid_vol_cb) * 12 * 2);

    hipDoubleComplex *AO = (hipDoubleComplex *) (U + (s_x_cb * s_y * s_z * t +
                                                      s_x_cb * s_y * z +
                                                      s_x_cb * y +
                                                      x + (1 - cb) * subgrid_vol_cb) * 9 * 2);
*/

    double tmp[2];
    double destE[12];
    double srcO[24];
    double AO[18];

    for (int i = 0; i < 24; i++) {

        srcO[i] = src[(s_x_cb * s_y * s_z * t +
                       s_x_cb * s_y * z +
                       s_x_cb * y +
                       x + (1 - cb) * subgrid_vol_cb) * 12 * 2 + i];

    }

    for (int i = 0; i < 12; i++) {

        destE[i] = 0;

    }

    for (int i = 0; i < 18; i++) {

        AO[i] = U[(s_x_cb * s_y * s_z * t +
                   s_x_cb * s_y * z +
                   s_x_cb * y +
                   x + (1 - cb) * subgrid_vol_cb) * 9 * 2 + i];

    }

    int cont = s_x_cb * s_y * t + s_x_cb * y + x;
    for (int c1 = 0; c1 < 3; c1++) {
        for (int c2 = 0; c2 < 3; c2++) {
/*
            tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half * hipConj(AO[c2 * 3 + c1]);
            dest_f[cont * 6 * 2 + (0 * 3 + c1) * 2 + 0] += tmp.x;
            dest_f[cont * 6 * 2 + (0 * 3 + c1) * 2 + 1] += tmp.y;
            tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half * hipConj(AO[c2 * 3 + c1]);
            dest_f[cont * 6 * 2 + (1 * 3 + c1) * 2 + 0] += tmp.x;
            dest_f[cont * 6 * 2 + (1 * 3 + c1) * 2 + 1] += tmp.y;
*/

            tmp[0] = -(srcO[(0 * 3 + c2) * 2 + 0] - flag * srcO[(2 * 3 + c2) * 2 + 1]) * half * AO[(c2 * 3 + c1) * 2 + 0]
                     -(srcO[(0 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 0]) * half * AO[(c2 * 3 + c1) * 2 + 1];
            tmp[1] = +(srcO[(0 * 3 + c2) * 2 + 0] - flag * srcO[(2 * 3 + c2) * 2 + 1]) * half * AO[(c2 * 3 + c1) * 2 + 1]
                     -(srcO[(0 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 0]) * half * AO[(c2 * 3 + c1) * 2 + 0];

            destE[(0 * 3 + c1) * 2 + 0] += tmp[0];
            destE[(0 * 3 + c1) * 2 + 1] += tmp[1];

            tmp[0] = -(srcO[(1 * 3 + c2) * 2 + 0] + flag * srcO[(3 * 3 + c2) * 2 + 1]) * half * AO[(c2 * 3 + c1) * 2 + 0]
                     -(srcO[(1 * 3 + c2) * 2 + 1] - flag * srcO[(3 * 3 + c2) * 2 + 0]) * half * AO[(c2 * 3 + c1) * 2 + 1];
            tmp[1] = +(srcO[(1 * 3 + c2) * 2 + 0] + flag * srcO[(3 * 3 + c2) * 2 + 1]) * half * AO[(c2 * 3 + c1) * 2 + 1]
                     -(srcO[(1 * 3 + c2) * 2 + 1] - flag * srcO[(3 * 3 + c2) * 2 + 0]) * half * AO[(c2 * 3 + c1) * 2 + 0];

            destE[(1 * 3 + c1) * 2 + 0] += tmp[0];
            destE[(1 * 3 + c1) * 2 + 1] += tmp[1];
        }
    }

    for (int i = 0; i < 12; i++) {
        dest_f[cont * 6 * 2 + i] = destE[i];
    }
}

static __global__ void transfer_t_f(double *src, double *dest_b, double *U,
                                    const int v_x, const int v_y, const int v_z, const int v_t,
                                    const int s_x, const int s_y, const int s_z, const int s_t,
                                    const int rank, const int cb, const int flag) {
/*
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;
*/


    const int s_x_cb = s_x >> 1;

    const double half = 0.5;
//    const hipDoubleComplex I(0, 1);

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    int z = xyz / (s_y * s_x_cb);
    int y = (xyz / s_x_cb) % s_y;
    int x = xyz % s_x_cb;


    int t = 0;
/*
    hipDoubleComplex tmp;
    hipDoubleComplex *srcO = (hipDoubleComplex * )(src + (s_x_cb * s_y * s_z * t +
                                                          s_x_cb * s_y * z +
                                                          s_x_cb * y +
                                                          x + (1 - cb) * subgrid_vol_cb) * 12 * 2);
*/

    double *srcO = src + (s_x_cb * s_y * s_z * t +
                          s_x_cb * s_y * z +
                          s_x_cb * y +
                          x + (1 - cb) * subgrid_vol_cb) * 12 * 2;

    double tmp[2];


    int cont = s_x_cb * s_y * z + s_x_cb * y + x;

    for (int c2 = 0; c2 < 3; c2++) {
//        tmp = -(srcO[0 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half;
        tmp[0] = -(srcO[(0 * 3 + c2) * 2 + 0] - flag * srcO[(2 * 3 + c2) * 2 + 0]) * half;
        tmp[1] = -(srcO[(0 * 3 + c2) * 2 + 1] - flag * srcO[(2 * 3 + c2) * 2 + 1]) * half;

//        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 0] += tmp.x;
//        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 1] += tmp.y;

        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 0] = tmp[0];
        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 1] = tmp[1];

//        tmp = -(srcO[1 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half;

        tmp[0] = -(srcO[(1 * 3 + c2) * 2 + 0] - flag * srcO[(3 * 3 + c2) * 2 + 0]) * half;
        tmp[1] = -(srcO[(1 * 3 + c2) * 2 + 1] - flag * srcO[(3 * 3 + c2) * 2 + 1]) * half;

        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 0] = tmp[0];
        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 1] = tmp[1];
    }
}

static __global__ void transfer_t_b(double *src, double *dest_f, double *U,
                                    const int v_x, const int v_y, const int v_z, const int v_t,
                                    const int s_x, const int s_y, const int s_z, const int s_t,
                                    const int rank, const int cb, const int flag) {

//    int x = blockDim.x * blockIdx.x + threadIdx.x;
//    int y = blockDim.y * blockIdx.y + threadIdx.y;
//    int z = blockDim.z * blockIdx.z + threadIdx.z;

    const int s_x_cb = s_x >> 1;

    const double half = 0.5;
//    const hipDoubleComplex I(0, 1);

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

//    hipDoubleComplex tmp;

    int xyz = blockDim.x * blockIdx.x + threadIdx.x;

    int z = xyz / (s_y * s_x_cb);
    int y = (xyz / s_x_cb) % s_y;
    int x = xyz % s_x_cb;

    int t = s_t - 1;

/*
    hipDoubleComplex *srcO = (hipDoubleComplex * )(src + (s_x_cb * s_y * s_z * t +
                                                          s_x_cb * s_y * z +
                                                          s_x_cb * y +
                                                          x + (1 - cb) * subgrid_vol_cb) * 12 * 2);

    hipDoubleComplex *AO = (hipDoubleComplex * )(U + (s_x_cb * s_y * s_z * t +
                                                      s_x_cb * s_y * z +
                                                      s_x_cb * y +
                                                      x + (1 - cb) * subgrid_vol_cb) * 9 * 2);
*/

    double tmp[2];
    double destE[12];
    double srcO[24];
    double AO[18];

    for (int i = 0; i < 24; i++) {
        srcO[i] = src[(s_x_cb * s_y * s_z * t +
                       s_x_cb * s_y * z +
                       s_x_cb * y +
                       x + (1 - cb) * subgrid_vol_cb) * 12 * 2 + i];
    }

    for (int i = 0; i < 12; i++) {
        destE[i] = 0;
    }

    for (int i = 0; i < 18; i++) {

        AO[i] = U[(s_x_cb * s_y * s_z * t +
                   s_x_cb * s_y * z +
                   s_x_cb * y +
                   x + (1 - cb) * subgrid_vol_cb) * 9 * 2 + i];

    }

    int cont = s_x_cb * s_y * z + s_x_cb * y + x;
    for (int c1 = 0; c1 < 3; c1++) {
        for (int c2 = 0; c2 < 3; c2++) {
/*
            tmp = -(srcO[0 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half * hipConj(AO[c2 * 3 + c1]);
            dest_f[cont * 6 * 2 + (0 * 3 + c1) * 2 + 0] += tmp.x;
            dest_f[cont * 6 * 2 + (0 * 3 + c1) * 2 + 1] += tmp.y;
            tmp = -(srcO[1 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half * hipConj(AO[c2 * 3 + c1]);
            dest_f[cont * 6 * 2 + (1 * 3 + c1) * 2 + 0] += tmp.x;
            dest_f[cont * 6 * 2 + (1 * 3 + c1) * 2 + 1] += tmp.y;
*/
            tmp[0] = (-(srcO[(0 * 3 + c2) * 2 + 0] + flag * srcO[(2 * 3 + c2) * 2 + 0]) *
                      half *
                      AO[(c2 * 3 + c1) * 2 + 0]) -
                     ((srcO[(0 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 1]) *
                      half *
                      AO[(c2 * 3 + c1) * 2 + 1]);

            tmp[1] = (+(srcO[(0 * 3 + c2) * 2 + 0] + flag * srcO[(2 * 3 + c2) * 2 + 0]) *
                      half *
                      AO[(c2 * 3 + c1) * 2 + 1]) -
                     ((srcO[(0 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 1]) *
                      half *
                      AO[(c2 * 3 + c1) * 2 + 0]);

            destE[(0 * 3 + c1) * 2 + 0] += tmp[0];
            destE[(0 * 3 + c1) * 2 + 1] += tmp[1];

            tmp[0] = (-(srcO[(1 * 3 + c2) * 2 + 0] + flag * srcO[(3 * 3 + c2) * 2 + 0]) *
                      half *
                      AO[(c2 * 3 + c1) * 2 + 0]) -
                     ((srcO[(1 * 3 + c2) * 2 + 1] + flag * srcO[(3 * 3 + c2) * 2 + 1]) *
                      half *
                      AO[(c2 * 3 + c1) * 2 + 1]);

            tmp[1] = (+(srcO[(1 * 3 + c2) * 2 + 0] + flag * srcO[(3 * 3 + c2) * 2 + 0]) *
                      half *
                      AO[(c2 * 3 + c1) * 2 + 1])
                     - ((srcO[(1 * 3 + c2) * 2 + 1] + flag * srcO[(3 * 3 + c2) * 2 + 1]) *
                        half *
                        AO[(c2 * 3 + c1) * 2 + 0]);

            destE[(1 * 3 + c1) * 2 + 0] += tmp[0];
            destE[(1 * 3 + c1) * 2 + 1] += tmp[1];

        }
    }

    for (int i = 0; i < 12; i++) {
        dest_f[cont * 6 * 2 + i] = destE[i];
    }

}

#endif //LATTICE_TRANSFER_KERNEL_H
