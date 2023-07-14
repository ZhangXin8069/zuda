//
// Created by louis on 2022/2/20.
//

#ifndef LATTICE_DOT_PRODUCT_KERNEL_H
#define LATTICE_DOT_PRODUCT_KERNEL_H

#include <complex>
#include "hip/hip_runtime.h"
#include "hip/hip_complex.h"

static __global__ void cDotProduct_g(double *result, double *x_i, double *y_i,
                                     const int s_x, const int s_y, const int s_z, const int s_t) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    __shared__ double r_t[8000];

    const int pos_loc_xyz = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    const int vol_loc = blockDim.x * blockDim.y * blockDim.z;

    r_t[pos_loc_xyz * 2 + 0] = 0;
    r_t[pos_loc_xyz * 2 + 1] = 0;

    for (int t = 0; t < s_t; t++) {
        int pos = s_x * s_y * s_z * t + s_x * s_y * z + s_x * y + x;
        hipDoubleComplex *x_t = (hipDoubleComplex *) (x_i + pos * 24);
        hipDoubleComplex *y_t = (hipDoubleComplex *) (y_i + pos * 24);
        for (int j = 0; j < 12; j++) {

            r_t[pos_loc_xyz * 2 + 0] += (y_t[j] * hipConj(x_t[j])).x;
            r_t[pos_loc_xyz * 2 + 1] += (y_t[j] * hipConj(x_t[j])).y;

        }
    }

    __syncthreads();

    for (int stride = 1; stride < vol_loc; stride *= 2) {

        if (pos_loc_xyz % (2 * stride) == 0 && pos_loc_xyz + stride < vol_loc) {

            r_t[pos_loc_xyz * 2 + 0] += r_t[(pos_loc_xyz + stride) * 2 + 0];
            r_t[pos_loc_xyz * 2 + 1] += r_t[(pos_loc_xyz + stride) * 2 + 1];
        }

        __syncthreads();

    }

    double real = r_t[0];
    double imag = r_t[1];


    if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
        result[0] += real;
        result[1] += imag;
    }
    __syncthreads();
}

static __global__ void cDotProduct_g2(double *result, double *x_i, double *y_i,
                                      const int s_x, const int s_y, const int s_z, const int s_t) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;

    int thread = blockDim.x;

    __shared__ double r_t[8000];

    int tot = s_x * s_y * s_z * s_t * 12;

    int sub = tot / thread;


    r_t[x * 2 + 0] = 0;
    r_t[x * 2 + 1] = 0;

    for (int i = 0; i < sub; i++) {

        double *x_t = x_i + (i * thread + x) * 2;
        double *y_t = y_i + (i * thread + x) * 2;

        r_t[x * 2 + 0] += y_t[0] * x_t[0] + y_t[1] * x_t[1];
        r_t[x * 2 + 1] += y_t[1] * x_t[0] - y_t[0] * x_t[1];
    }
    __syncthreads();

    for (int stride = 1; stride < thread; stride *= 2) {

        if (x % (2 * stride) == 0 && x + stride < thread) {

            r_t[x * 2 + 0] += r_t[(x + stride) * 2 + 0];
            r_t[x * 2 + 1] += r_t[(x + stride) * 2 + 1];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        result[0] = r_t[0];
        result[1] = r_t[1];
    }
}

static __global__ void cDotProduct_v1_g2(double *result, double *x_i, double *y_i,
                                         const int size, const int N, const int block) {

//    int x = blockDim.x * blockIdx.x + threadIdx.x;

    int x = threadIdx.x;

    int x_t = blockDim.x * blockIdx.x + threadIdx.x;

    int thread = blockDim.x;

    __shared__ double r_t[8000];


//    const int sub = size / thread;

//    for (int j = 0; j < N ; j++){

//	    r_t[x * 2 + 0] = 0;
//	    r_t[x * 2 + 1] = 0;

//     for (int i = 0; i < sub; i++){



    double y_t_r = y_i[x_t * 2 + 0];
    double y_t_i = y_i[x_t * 2 + 1];

    for (int j = 0; j < N; j++) {

        double x_t_r = x_i[(j * size + x_t) * 2 + 0];
        double x_t_i = x_i[(j * size + x_t) * 2 + 1];

        r_t[x * 2 + 0] = y_t_r * x_t_r + y_t_i * x_t_i;
        r_t[x * 2 + 1] = y_t_i * x_t_r - y_t_r * x_t_i;

//     }

        __syncthreads();


        for (int stride = 1; stride < thread; stride *= 2) {

            if (x % (2 * stride) == 0 && x + stride < thread) {

                r_t[x * 2 + 0] += r_t[(x + stride) * 2 + 0];
                r_t[x * 2 + 1] += r_t[(x + stride) * 2 + 1];

            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {

            result[block * j * 2 + blockIdx.x * 2 + 0] = r_t[0];
            result[block * j * 2 + blockIdx.x * 2 + 1] = r_t[1];
        }
    }
}

static __global__ void cDotProduct_v2_g2(double *result, double *x_i,
                                         const int size, const int N, const int block, const int thread) {

    int x_t = blockDim.x * blockIdx.x + threadIdx.x;

    int x = threadIdx.x;

//	int thread = blockDim.x;

    __shared__ double r_t[8000];

//	for (int j = 0; j < N ; j++){

//		r_t[x * 2 + 0] = x_i[j * size * 2 + x_t * 2 + 0];
//		r_t[x * 2 + 1] = x_i[j * size * 2 + x_t * 2 + 1];

    int j = (x_t / size);
    int i = (x_t / thread) % block;

    r_t[x * 2 + 0] = x_i[x_t * 2 + 0];
    r_t[x * 2 + 1] = x_i[x_t * 2 + 1];

    __syncthreads();

    int x_j = x % thread;

    for (int stride = 1; stride < thread; stride *= 2) {

        if (x_j % (2 * stride) == 0 && x_j + stride < thread) {

            r_t[x * 2 + 0] += r_t[(x + stride) * 2 + 0];
            r_t[x * 2 + 1] += r_t[(x + stride) * 2 + 1];

        }
        __syncthreads();
    }

    if (x_j == 0) {
        result[block * j * 2 + i * 2 + 0] = r_t[x * 2 + 0];
        result[block * j * 2 + i * 2 + 1] = r_t[x * 2 + 1];
    }
//	}
}

static void cDotProduct(double *result, double *x_i, double *y_i,
                 const int s_x, const int s_y, const int s_z, const int s_t) {

    std::complex<double> *r_t = (std::complex<double> *) result;

    *r_t = 0;

    for (int t = 0; t < s_t; t++) {
        for (int z = 0; z < s_z; z++) {
            for (int y = 0; y < s_y; y++) {
                for (int x = 0; x < s_x; x++) {
                    int pos = s_x * s_y * s_z * t + s_x * s_y * z + s_x * y + x;
                    std::complex<double> *x_t = (std::complex<double> *) (x_i + pos * 24);
                    std::complex<double> *y_t = (std::complex<double> *) (y_i + pos * 24);
                    for (int j = 0; j < 12; j++) {
                        *r_t = *r_t + y_t[j] * std::conj(x_t[j]);
                    }
                }
            }
        }
    }
}

#endif //LATTICE_DOT_PRODUCT_KERNEL_H
