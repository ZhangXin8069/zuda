//
// Created by louis on 2022/2/20.
//

#ifndef LATTICE_MAIN_XYZT_ABP5_H
#define LATTICE_MAIN_XYZT_ABP5_H

#include "hip/hip_runtime.h"
#include "hip/hip_complex.h"

static __global__ void main_xyzt(double *src, double *dest,
                                 double *U_x, double *U_y, double *U_z, double *U_t,
                                 const int v_x, const int v_y, const int v_z, const int v_t,
                                 const int s_x, const int s_y, const int s_z, const int s_t, // s_x s_y s_z s_t  from 16 to 24
                                 const int rank, const int cb, const int flag) {


    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int N_sub[4] = {v_x / s_x, v_y / s_y, v_z / s_z, v_t / s_t};

    const double half = 0.5;

    const int x_p = ((rank / N_sub[0]) % N_sub[1]) * s_y +
                    ((rank / (N_sub[1] * N_sub[0])) % N_sub[2]) * s_z +
                    (rank / (N_sub[2] * N_sub[1] * N_sub[0])) * s_t;

    const int s_x_cb = s_x >> 1;

    int xyzt = blockDim.x * blockIdx.x + threadIdx.x;

    int t = xyzt / (s_z * s_y * s_x_cb);
    int z = (xyzt / (s_y * s_x_cb)) % s_z;
    int y = (xyzt / s_x_cb) % s_y;
    int x = xyzt % s_x_cb;

    double tmp[2];

    double destE[24];
    double srcO[24];
    double AE[18];

    for (int i = 0; i < 24; i++) {
        destE[i] = 0;
    }

    int x_u = ((y + z + t + x_p) % 2 == cb || N_sub[0] == 1) ? s_x_cb : s_x_cb - 1;
    if (x < x_u) {

        int f_x = ((y + z + t + x_p) % 2 == cb) ? x : (x + 1) % s_x_cb;

        for (int i = 0; i < 24; i++) {

            srcO[i] = src[(s_x_cb * s_y * s_z * t +
                           s_x_cb * s_y * z +
                           s_x_cb * y +
                           f_x + (1 - cb) * subgrid_vol_cb) * 12 * 2 + i];
        }

        for (int i = 0; i < 18; i++) {

            AE[i] = U_x[(s_x_cb * s_y * s_z * t +
                         s_x_cb * s_y * z +
                         s_x_cb * y +
                         x + cb * subgrid_vol_cb) * 9 * 2 + i];

        }


        for (int c1 = 0; c1 < 3; c1++) {
            for (int c2 = 0; c2 < 3; c2++) {

                tmp[0] = -(srcO[(0 * 3 + c2) * 2 + 0] + flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                         AE[(c1 * 3 + c2) * 2 + 0]
                         + (srcO[(0 * 3 + c2) * 2 + 1] - flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                           AE[(c1 * 3 + c2) * 2 + 1];
                tmp[1] = -(srcO[(0 * 3 + c2) * 2 + 0] + flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                         AE[(c1 * 3 + c2) * 2 + 1]
                         - (srcO[(0 * 3 + c2) * 2 + 1] - flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                           AE[(c1 * 3 + c2) * 2 + 0];

                destE[(0 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(0 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(3 * 3 + c1) * 2 + 0] += -flag * tmp[1];
                destE[(3 * 3 + c1) * 2 + 1] += flag * tmp[0];

                tmp[0] = -(srcO[(1 * 3 + c2) * 2 + 0] + flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                         AE[(c1 * 3 + c2) * 2 + 0]
                         + (srcO[(1 * 3 + c2) * 2 + 1] - flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                           AE[(c1 * 3 + c2) * 2 + 1];
                tmp[1] = -(srcO[(1 * 3 + c2) * 2 + 0] + flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                         AE[(c1 * 3 + c2) * 2 + 1]
                         - (srcO[(1 * 3 + c2) * 2 + 1] - flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                           AE[(c1 * 3 + c2) * 2 + 0];

                destE[(1 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(1 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(2 * 3 + c1) * 2 + 0] += -flag * tmp[1];
                destE[(2 * 3 + c1) * 2 + 1] += flag * tmp[0];
            }
        }
    }

    int x_d = (((y + z + t + x_p) % 2) != cb || N_sub[0] == 1) ? 0 : 1;
    if (x >= x_d) {

        int b_x = ((t + z + y + x_p) % 2 == cb) ? (x - 1 + s_x_cb) % s_x_cb : x;

        for (int i = 0; i < 24; i++) {

            srcO[i] = src[(s_x_cb * s_y * s_z * t +
                           s_x_cb * s_y * z +
                           s_x_cb * y +
                           b_x + (1 - cb) * subgrid_vol_cb) * 12 * 2 + i];

        }

        for (int i = 0; i < 18; i++) {

            AE[i] = U_x[(s_x_cb * s_y * s_z * t +
                         s_x_cb * s_y * z +
                         s_x_cb * y +
                         b_x + (1 - cb) * subgrid_vol_cb) * 9 * 2 + i];

        }

        for (int c1 = 0; c1 < 3; c1++) {
            for (int c2 = 0; c2 < 3; c2++) {

                tmp[0] = -(srcO[(0 * 3 + c2) * 2 + 0] - flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                         AE[(c2 * 3 + c1) * 2 + 0]
                         - (srcO[(0 * 3 + c2) * 2 + 1] + flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                           AE[(c2 * 3 + c1) * 2 + 1];
                tmp[1] = +(srcO[(0 * 3 + c2) * 2 + 0] - flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                         AE[(c2 * 3 + c1) * 2 + 1]
                         - (srcO[(0 * 3 + c2) * 2 + 1] + flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                           AE[(c2 * 3 + c1) * 2 + 0];

                destE[(0 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(0 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(3 * 3 + c1) * 2 + 0] += flag * tmp[1];
                destE[(3 * 3 + c1) * 2 + 1] += -flag * tmp[0];

                tmp[0] = -(srcO[(1 * 3 + c2) * 2 + 0] - flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                         AE[(c2 * 3 + c1) * 2 + 0]
                         - (srcO[(1 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                           AE[(c2 * 3 + c1) * 2 + 1];
                tmp[1] = +(srcO[(1 * 3 + c2) * 2 + 0] - flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                         AE[(c2 * 3 + c1) * 2 + 1]
                         - (srcO[(1 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                           AE[(c2 * 3 + c1) * 2 + 0];

                destE[(1 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(1 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(2 * 3 + c1) * 2 + 0] += flag * tmp[1];
                destE[(2 * 3 + c1) * 2 + 1] += -flag * tmp[0];
            }
        }
    }

    int y_u = (N_sub[1] == 1) ? s_y : s_y - 1;
    if (y < y_u) {

        int f_y = (y + 1) % s_y;

        for (int i = 0; i < 24; i++) {

            srcO[i] = src[(s_x_cb * s_y * s_z * t +
                           s_x_cb * s_y * z +
                           s_x_cb * f_y +
                           x + (1 - cb) * subgrid_vol_cb) * 12 * 2 + i];

        }

        for (int i = 0; i < 18; i++) {

            AE[i] = U_y[(s_x_cb * s_y * s_z * t +
                         s_x_cb * s_y * z +
                         s_x_cb * y +
                         x + cb * subgrid_vol_cb) * 9 * 2 + i];

        }

        for (int c1 = 0; c1 < 3; c1++) {
            for (int c2 = 0; c2 < 3; c2++) {

                tmp[0] = -(srcO[(0 * 3 + c2) * 2 + 0] + flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                         AE[(c1 * 3 + c2) * 2 + 0]
                         + (srcO[(0 * 3 + c2) * 2 + 1] + flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                           AE[(c1 * 3 + c2) * 2 + 1];
                tmp[1] = -(srcO[(0 * 3 + c2) * 2 + 0] + flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                         AE[(c1 * 3 + c2) * 2 + 1]
                         - (srcO[(0 * 3 + c2) * 2 + 1] + flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                           AE[(c1 * 3 + c2) * 2 + 0];

                destE[(0 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(0 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(3 * 3 + c1) * 2 + 0] += flag * tmp[0];
                destE[(3 * 3 + c1) * 2 + 1] += flag * tmp[1];

                tmp[0] = -(srcO[(1 * 3 + c2) * 2 + 0] - flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                         AE[(c1 * 3 + c2) * 2 + 0]
                         + (srcO[(1 * 3 + c2) * 2 + 1] - flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                           AE[(c1 * 3 + c2) * 2 + 1];
                tmp[1] = -(srcO[(1 * 3 + c2) * 2 + 0] - flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                         AE[(c1 * 3 + c2) * 2 + 1]
                         - (srcO[(1 * 3 + c2) * 2 + 1] - flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                           AE[(c1 * 3 + c2) * 2 + 0];

                destE[(1 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(1 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(2 * 3 + c1) * 2 + 0] -= flag * tmp[0];
                destE[(2 * 3 + c1) * 2 + 1] -= flag * tmp[1];


            }
        }
    }

    int y_d = (N_sub[1] == 1) ? 0 : 1;

    if (y >= y_d) {

        int b_y = (y - 1 + s_y) % s_y;

        for (int i = 0; i < 24; i++) {

            srcO[i] = src[(s_x_cb * s_y * s_z * t +
                           s_x_cb * s_y * z +
                           s_x_cb * b_y +
                           x + (1 - cb) * subgrid_vol_cb) * 12 * 2 + i];

        }

        for (int i = 0; i < 18; i++) {

            AE[i] = U_y[(s_x_cb * s_y * s_z * t +
                         s_x_cb * s_y * z +
                         s_x_cb * b_y +
                         x + (1 - cb) * subgrid_vol_cb) * 9 * 2 + i];

        }

        for (int c1 = 0; c1 < 3; c1++) {
            for (int c2 = 0; c2 < 3; c2++) {

                tmp[0] = -(srcO[(0 * 3 + c2) * 2 + 0] - flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                         AE[(c2 * 3 + c1) * 2 + 0]
                         - (srcO[(0 * 3 + c2) * 2 + 1] - flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                           AE[(c2 * 3 + c1) * 2 + 1];
                tmp[1] = +(srcO[(0 * 3 + c2) * 2 + 0] - flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                         AE[(c2 * 3 + c1) * 2 + 1]
                         - (srcO[(0 * 3 + c2) * 2 + 1] - flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                           AE[(c2 * 3 + c1) * 2 + 0];

                destE[(0 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(0 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(3 * 3 + c1) * 2 + 0] -= flag * tmp[0];
                destE[(3 * 3 + c1) * 2 + 1] -= flag * tmp[1];

                tmp[0] = -(srcO[(1 * 3 + c2) * 2 + 0] + flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                         AE[(c2 * 3 + c1) * 2 + 0]
                         - (srcO[(1 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                           AE[(c2 * 3 + c1) * 2 + 1];
                tmp[1] = +(srcO[(1 * 3 + c2) * 2 + 0] + flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                         AE[(c2 * 3 + c1) * 2 + 1]
                         - (srcO[(1 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                           AE[(c2 * 3 + c1) * 2 + 0];

                destE[(1 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(1 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(2 * 3 + c1) * 2 + 0] += flag * tmp[0];
                destE[(2 * 3 + c1) * 2 + 1] += flag * tmp[1];

            }
        }
    }


    int z_u = (N_sub[2] == 1) ? s_z : s_z - 1;
    if (z < z_u) {
        int f_z = (z + 1) % s_z;

        for (int i = 0; i < 24; i++) {

            srcO[i] = src[(s_x_cb * s_y * s_z * t +
                           s_x_cb * s_y * f_z +
                           s_x_cb * y +
                           x + (1 - cb) * subgrid_vol_cb) * 12 * 2 + i];
        }

        for (int i = 0; i < 18; i++) {

            AE[i] = U_z[(s_x_cb * s_y * s_z * t +
                         s_x_cb * s_y * z +
                         s_x_cb * y +
                         x + cb * subgrid_vol_cb) * 9 * 2 + i];

        }

        for (int c1 = 0; c1 < 3; c1++) {
            for (int c2 = 0; c2 < 3; c2++) {

                tmp[0] = -(srcO[(0 * 3 + c2) * 2 + 0] + flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                         AE[(c1 * 3 + c2) * 2 + 0]
                         + (srcO[(0 * 3 + c2) * 2 + 1] - flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                           AE[(c1 * 3 + c2) * 2 + 1];
                tmp[1] = -(srcO[(0 * 3 + c2) * 2 + 0] + flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                         AE[(c1 * 3 + c2) * 2 + 1]
                         - (srcO[(0 * 3 + c2) * 2 + 1] - flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                           AE[(c1 * 3 + c2) * 2 + 0];

                destE[(0 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(0 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(2 * 3 + c1) * 2 + 0] += -flag * tmp[1];
                destE[(2 * 3 + c1) * 2 + 1] += flag * tmp[0];

                tmp[0] = -(srcO[(1 * 3 + c2) * 2 + 0] - flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                         AE[(c1 * 3 + c2) * 2 + 0]
                         + (srcO[(1 * 3 + c2) * 2 + 1] + flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                           AE[(c1 * 3 + c2) * 2 + 1];
                tmp[1] = -(srcO[(1 * 3 + c2) * 2 + 0] - flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                         AE[(c1 * 3 + c2) * 2 + 1]
                         - (srcO[(1 * 3 + c2) * 2 + 1] + flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                           AE[(c1 * 3 + c2) * 2 + 0];

                destE[(1 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(1 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(3 * 3 + c1) * 2 + 0] += flag * tmp[1];
                destE[(3 * 3 + c1) * 2 + 1] += -flag * tmp[0];
            }
        }
    }


    int z_d = (N_sub[2] == 1) ? 0 : 1;
    if (z >= z_d) {

        int b_z = (z - 1 + s_z) % s_z;

        for (int i = 0; i < 24; i++) {

            srcO[i] = src[(s_x_cb * s_y * s_z * t +
                           s_x_cb * s_y * b_z +
                           s_x_cb * y +
                           x + (1 - cb) * subgrid_vol_cb) * 12 * 2 + i];

        }

        for (int i = 0; i < 18; i++) {

            AE[i] = U_z[(s_x_cb * s_y * s_z * t +
                         s_x_cb * s_y * b_z +
                         s_x_cb * y +
                         x + (1 - cb) * subgrid_vol_cb) * 9 * 2 + i];
        }


        for (int c1 = 0; c1 < 3; c1++) {
            for (int c2 = 0; c2 < 3; c2++) {

                tmp[0] = -(srcO[(0 * 3 + c2) * 2 + 0] - flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                         AE[(c2 * 3 + c1) * 2 + 0]
                         - (srcO[(0 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                           AE[(c2 * 3 + c1) * 2 + 1];
                tmp[1] = +(srcO[(0 * 3 + c2) * 2 + 0] - flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                         AE[(c2 * 3 + c1) * 2 + 1]
                         - (srcO[(0 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                           AE[(c2 * 3 + c1) * 2 + 0];

                destE[(0 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(0 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(2 * 3 + c1) * 2 + 0] += flag * tmp[1];
                destE[(2 * 3 + c1) * 2 + 1] += -flag * tmp[0];

                tmp[0] = -(srcO[(1 * 3 + c2) * 2 + 0] + flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                         AE[(c2 * 3 + c1) * 2 + 0]
                         - (srcO[(1 * 3 + c2) * 2 + 1] - flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                           AE[(c2 * 3 + c1) * 2 + 1];
                tmp[1] = +(srcO[(1 * 3 + c2) * 2 + 0] + flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                         AE[(c2 * 3 + c1) * 2 + 1]
                         - (srcO[(1 * 3 + c2) * 2 + 1] - flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                           AE[(c2 * 3 + c1) * 2 + 0];

                destE[(1 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(1 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(3 * 3 + c1) * 2 + 0] += -flag * tmp[1];
                destE[(3 * 3 + c1) * 2 + 1] += flag * tmp[0];
            }
        }
    }

    int t_u = (N_sub[3] == 1) ? s_t : s_t - 1;
    if (t < t_u) {

        int f_t = (t + 1) % s_t;

        for (int i = 0; i < 24; i++) {

            srcO[i] = src[(s_x_cb * s_y * s_z * f_t +
                           s_x_cb * s_y * z +
                           s_x_cb * y +
                           x + (1 - cb) * subgrid_vol_cb) * 12 * 2 + i];

        }

        for (int i = 0; i < 18; i++) {

            AE[i] = U_t[(s_x_cb * s_y * s_z * t +
                         s_x_cb * s_y * z +
                         s_x_cb * y +
                         x + cb * subgrid_vol_cb) * 9 * 2 + i];

        }

        for (int c1 = 0; c1 < 3; c1++) {
            for (int c2 = 0; c2 < 3; c2++) {

                tmp[0] = -(srcO[(0 * 3 + c2) * 2 + 0] - flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                         AE[(c1 * 3 + c2) * 2 + 0]
                         + (srcO[(0 * 3 + c2) * 2 + 1] - flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                           AE[(c1 * 3 + c2) * 2 + 1];
                tmp[1] = -(srcO[(0 * 3 + c2) * 2 + 0] - flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                         AE[(c1 * 3 + c2) * 2 + 1]
                         - (srcO[(0 * 3 + c2) * 2 + 1] - flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                           AE[(c1 * 3 + c2) * 2 + 0];

                destE[(0 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(0 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(2 * 3 + c1) * 2 + 0] -= flag * tmp[0];
                destE[(2 * 3 + c1) * 2 + 1] -= flag * tmp[1];

                tmp[0] = -(srcO[(1 * 3 + c2) * 2 + 0] - flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                         AE[(c1 * 3 + c2) * 2 + 0]
                         + (srcO[(1 * 3 + c2) * 2 + 1] - flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                           AE[(c1 * 3 + c2) * 2 + 1];
                tmp[1] = -(srcO[(1 * 3 + c2) * 2 + 0] - flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                         AE[(c1 * 3 + c2) * 2 + 1]
                         - (srcO[(1 * 3 + c2) * 2 + 1] - flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                           AE[(c1 * 3 + c2) * 2 + 0];

                destE[(1 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(1 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(3 * 3 + c1) * 2 + 0] -= flag * tmp[0];
                destE[(3 * 3 + c1) * 2 + 1] -= flag * tmp[1];
            }
        }
    }

    int t_d = (N_sub[3] == 1) ? 0 : 1;
    if (t >= t_d) {

        int b_t = (t - 1 + s_t) % s_t;

        for (int i = 0; i < 24; i++) {

            srcO[i] = src[(s_x_cb * s_y * s_z * b_t +
                           s_x_cb * s_y * z +
                           s_x_cb * y +
                           x + (1 - cb) * subgrid_vol_cb) * 12 * 2 + i];

        }

        for (int i = 0; i < 18; i++) {

            AE[i] = U_t[(s_x_cb * s_y * s_z * b_t +
                         s_x_cb * s_y * z +
                         s_x_cb * y +
                         x + (1 - cb) * subgrid_vol_cb) * 9 * 2 + i];

        }

        for (int c1 = 0; c1 < 3; c1++) {
            for (int c2 = 0; c2 < 3; c2++) {

                tmp[0] = -(srcO[(0 * 3 + c2) * 2 + 0] + flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                         AE[(c2 * 3 + c1) * 2 + 0]
                         - (srcO[(0 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                           AE[(c2 * 3 + c1) * 2 + 1];
                tmp[1] = +(srcO[(0 * 3 + c2) * 2 + 0] + flag * srcO[(2 * 3 + c2) * 2 + 0]) * half *
                         AE[(c2 * 3 + c1) * 2 + 1]
                         - (srcO[(0 * 3 + c2) * 2 + 1] + flag * srcO[(2 * 3 + c2) * 2 + 1]) * half *
                           AE[(c2 * 3 + c1) * 2 + 0];

                destE[(0 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(0 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(2 * 3 + c1) * 2 + 0] += flag * tmp[0];
                destE[(2 * 3 + c1) * 2 + 1] += flag * tmp[1];

                tmp[0] = -(srcO[(1 * 3 + c2) * 2 + 0] + flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                         AE[(c2 * 3 + c1) * 2 + 0]
                         - (srcO[(1 * 3 + c2) * 2 + 1] + flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                           AE[(c2 * 3 + c1) * 2 + 1];
                tmp[1] = +(srcO[(1 * 3 + c2) * 2 + 0] + flag * srcO[(3 * 3 + c2) * 2 + 0]) * half *
                         AE[(c2 * 3 + c1) * 2 + 1]
                         - (srcO[(1 * 3 + c2) * 2 + 1] + flag * srcO[(3 * 3 + c2) * 2 + 1]) * half *
                           AE[(c2 * 3 + c1) * 2 + 0];

                destE[(1 * 3 + c1) * 2 + 0] += tmp[0];
                destE[(1 * 3 + c1) * 2 + 1] += tmp[1];
                destE[(3 * 3 + c1) * 2 + 0] += flag * tmp[0];
                destE[(3 * 3 + c1) * 2 + 1] += flag * tmp[1];

            }
        }
    }

    for (int i = 0; i < 24; i++) {

        dest[(s_x_cb * s_y * s_z * t +
              s_x_cb * s_y * z +
              s_x_cb * y +
              x + cb * subgrid_vol_cb) * 12 * 2 + i] = destE[i];
    }
}

#endif //LATTICE_MAIN_XYZT_ABP5_H
