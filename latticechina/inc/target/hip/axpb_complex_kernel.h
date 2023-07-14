//
// Created by louis on 2022/2/20.
//

#ifndef LATTICE_AXPB_COMPLEX_KERNEL_H
#define LATTICE_AXPB_COMPLEX_KERNEL_H

#include <complex>

#include "hip/hip_runtime.h"
#include "hip/hip_complex.h"

static __global__ void caxpbyz_g(double *a, double *x_i, double *b, double *y_i, double *z_i,
                                 const int s_x, const int s_y, const int s_z, const int s_t) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    hipDoubleComplex *a_t = (hipDoubleComplex *) a;
    hipDoubleComplex *b_t = (hipDoubleComplex *) b;

    for (int t = 0; t < s_t; t++) {

        int pos = s_x * s_y * s_z * t + s_x * s_y * z + s_x * y + x;

        hipDoubleComplex *x_t = (hipDoubleComplex *) (x_i + pos * 12 * 2);
        hipDoubleComplex *y_t = (hipDoubleComplex *) (y_i + pos * 12 * 2);
        hipDoubleComplex *z_t = (hipDoubleComplex *) (z_i + pos * 12 * 2);

        for (int i = 0; i < 12; i++) {
            z_t[i] = (*a_t) * x_t[i] + (*b_t) * y_t[i];
        }
    }
}

static __global__ void caxmbyz_g(double *a, double *x_i, double *b, double *y_i, double *z_i,
                                 const int s_x, const int s_y, const int s_z, const int s_t) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    hipDoubleComplex *a_t = (hipDoubleComplex *) a;
    hipDoubleComplex *b_t = (hipDoubleComplex *) b;

    for (int t = 0; t < s_t; t++) {

        int pos = s_x * s_y * s_z * t + s_x * s_y * z + s_x * y + x;

        hipDoubleComplex *x_t = (hipDoubleComplex *) (x_i + pos * 12 * 2);
        hipDoubleComplex *y_t = (hipDoubleComplex *) (y_i + pos * 12 * 2);
        hipDoubleComplex *z_t = (hipDoubleComplex *) (z_i + pos * 12 * 2);

        for (int i = 0; i < 12; i++) {
            z_t[i] = (*a_t) * x_t[i] - (*b_t) * y_t[i];
        }
    }
}

static __global__ void cxpbyz_g(double *x_i, double *b, double *y_i, double *z_i,
                                const int s_x, const int s_y, const int s_z, const int s_t) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    hipDoubleComplex *b_t = (hipDoubleComplex *) b;

    for (int t = 0; t < s_t; t++) {

        int pos = s_x * s_y * s_z * t + s_x * s_y * z + s_x * y + x;

        hipDoubleComplex *x_t = (hipDoubleComplex *) (x_i + pos * 12 * 2);
        hipDoubleComplex *y_t = (hipDoubleComplex *) (y_i + pos * 12 * 2);
        hipDoubleComplex *z_t = (hipDoubleComplex *) (z_i + pos * 12 * 2);

        for (int i = 0; i < 12; i++) {
            z_t[i] = x_t[i] + (*b_t) * y_t[i];
        }
    }
}

static __global__ void cxpbyz_g2(double *x_i, double *b, double *y_i, double *z_i) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;

    z_i[2 * x + 0] = x_i[2 * x + 0] + b[0] * y_i[2 * x + 0] - b[1] * y_i[2 * x + 1];
    z_i[2 * x + 1] = x_i[2 * x + 1] + b[0] * y_i[2 * x + 1] + b[1] * y_i[2 * x + 0];

}

static __global__ void cxpbyx_v1_g2(double *x_i, double *b, double *y_i, const int size, const int N) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;

    double x_t_r = x_i[2 * x + 0];
    double x_t_i = x_i[2 * x + 1];


    for (int i = 0; i < N; i++) {

        double b_t_r = b[i * 2 + 0];
        double b_t_i = b[i * 2 + 1];
        double y_t_r = y_i[i * size * 2 + x * 2 + 0];
        double y_t_i = y_i[i * size * 2 + x * 2 + 1];

        x_t_r += +b_t_r * y_t_r - b_t_i * y_t_i;
        x_t_i += +b_t_r * y_t_i + b_t_i * y_t_r;
    }

    x_i[2 * x + 0] = x_t_r;
    x_i[2 * x + 1] = x_t_i;

}


static __global__ void cxmbyz_g(double *x_i, double *b, double *y_i, double *z_i,
                                const int s_x, const int s_y, const int s_z, const int s_t) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;


    hipDoubleComplex *b_t = (hipDoubleComplex *) b;

    for (int t = 0; t < s_t; t++) {

        int pos = s_x * s_y * s_z * t + s_x * s_y * z + s_x * y + x;

        hipDoubleComplex *x_t = (hipDoubleComplex *) (x_i + pos * 12 * 2);
        hipDoubleComplex *y_t = (hipDoubleComplex *) (y_i + pos * 12 * 2);
        hipDoubleComplex *z_t = (hipDoubleComplex *) (z_i + pos * 12 * 2);

        for (int i = 0; i < 12; i++) {
            z_t[i] = x_t[i] - (*b_t) * y_t[i];
        }
    }
}

static __global__ void cxmbyx_v1_g2(double *x_i, double *b, double *y_i, const int size, const int N) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;

    double x_t_r = x_i[2 * x + 0];
    double x_t_i = x_i[2 * x + 1];


    for (int i = 0; i < N; i++) {

        double b_t_r = b[i * 2 + 0];
        double b_t_i = b[i * 2 + 1];
        double y_t_r = y_i[i * size * 2 + x * 2 + 0];
        double y_t_i = y_i[i * size * 2 + x * 2 + 1];

        x_t_r += -b_t_r * y_t_r + b_t_i * y_t_i;
        x_t_i += -b_t_r * y_t_i - b_t_i * y_t_r;
    }

    x_i[2 * x + 0] = x_t_r;
    x_i[2 * x + 1] = x_t_i;

}

static __global__ void cxmbyz_g2(double *x_i, double *b, double *y_i, double *z_i) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;

    z_i[2 * x + 0] = x_i[2 * x + 0] - b[0] * y_i[2 * x + 0] + b[1] * y_i[2 * x + 1];
    z_i[2 * x + 1] = x_i[2 * x + 1] - b[0] * y_i[2 * x + 1] - b[1] * y_i[2 * x + 0];

}


static void caxpbyz(std::complex<double> a, std::complex<double> *x_i,
                    std::complex<double> b, std::complex<double> *y_i,
                    std::complex<double> *z_i,
                    const int s_x, const int s_y, const int s_z, const int s_t) {

    for (int t = 0; t < s_t; t++) {
        for (int z = 0; z < s_z; z++) {
            for (int y = 0; y < s_y; y++) {
                for (int x = 0; x < s_x; x++) {

                    int pos = s_x * s_y * s_z * t + s_x * s_y * z + s_x * y + x;

                    std::complex<double> *x_t = x_i + pos * 12;
                    std::complex<double> *y_t = y_i + pos * 12;
                    std::complex<double> *z_t = z_i + pos * 12;

                    for (int i = 0; i < 12; i++) {
                        z_t[i] = a * x_t[i] + b * y_t[i];
                    }
                }
            }
        }
    }
}

void caxz(std::complex<double> a, std::complex<double> *x_i, std::complex<double> *z_i,
          const int s_x, const int s_y, const int s_z, const int s_t) {

    for (int t = 0; t < s_t; t++) {
        for (int z = 0; z < s_z; z++) {
            for (int y = 0; y < s_y; y++) {
                for (int x = 0; x < s_x; x++) {

                    int pos = s_x * s_y * s_z * t + s_x * s_y * z + s_x * y + x;
                    std::complex<double> *x_t = x_i + pos * 12;
                    std::complex<double> *z_t = z_i + pos * 12;
                    for (int i = 0; i < 12; i++) {
                        z_t[i] = a * x_t[i];
                    }
                }
            }
        }
    }
}


#endif //LATTICE_AXPB_COMPLEX_KERNEL_H
