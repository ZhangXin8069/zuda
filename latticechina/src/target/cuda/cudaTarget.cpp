//
// Created by louis on 2021/8/29.
//


#include <iostream>
#include <mpi.h>
#include "target/cuda/cudaTarget.h"

// cuda header file
#include "cuda_runtime.h"
#include "cuda/cuda_complex.h"
#include "operator.h"


__global__ void transfer_x_f(double *src, double *dest_b, double *U,
                             const int v_x, const int v_y, const int v_z, const int v_t,
                             const int s_x, const int s_y, const int s_z, const int s_t,
                             const int rank, const int cb, const int flag) {

    int y = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.y * blockIdx.y + threadIdx.y;
    int tp = blockDim.z * blockIdx.z + threadIdx.z;

    if (y >= s_y || z >= s_z) { return; }

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int N_sub[4] = {v_x / s_x, v_y / s_y, v_z / s_z, v_t / s_t};

    const double half = 0.5;
    const cudaDoubleComplex I(0, 1);

    const int x_p = ((rank / N_sub[0]) % N_sub[1]) * s_y +
                    ((rank / (N_sub[1] * N_sub[0])) % N_sub[2]) * s_z +
                    (rank / (N_sub[2] * N_sub[1] * N_sub[0])) * s_t;

    const int s_x_cb = s_x >> 1;

//        int cont = 0;

//        for (int y = 0; y < s_y; y++) {
//            for (int z = 0; z < s_z; z++) {
//                for (int t = 0; t < s_t; t++) {

    int t = (y + z + 2 * tp + x_p) % 2 == cb ? 2 * tp + 1 : 2 * tp;
    if (t >= s_t) { return; }

    int cont = s_y * s_z * tp + s_y * z + y;

    int x = 0;
    cudaDoubleComplex tmp;
    cudaDoubleComplex *srcO = (cudaDoubleComplex * )(src + (s_x_cb * s_y * s_z * t +
                                                          s_x_cb * s_y * z +
                                                          s_x_cb * y + x +
                                                          (1 - cb) * subgrid_vol_cb) * 12 * 2);

//                    int b = cont * 6;
//                    cont += 1;

    for (int c2 = 0; c2 < 3; c2++) {
        tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half;
        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 0] = tmp.x;
        dest_b[cont * 6 * 2 + (0 * 3 + c2) * 2 + 1] = tmp.y;
        tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half;
        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 0] = tmp.x;
        dest_b[cont * 6 * 2 + (1 * 3 + c2) * 2 + 1] = tmp.y;
    }
}

__global__ void transfer_x_b(double *src, double *dest_f, double *U,
                             const int v_x, const int v_y, const int v_z, const int v_t,
                             const int s_x, const int s_y, const int s_z, const int s_t,
                             const int rank, const int cb, const int flag) {

    int y = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.y * blockIdx.y + threadIdx.y;
    int tp = blockDim.z * blockIdx.z + threadIdx.z;

    if (y >= s_y || z >= s_z) { return; }

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int N_sub[4] = {v_x / s_x, v_y / s_y, v_z / s_z, v_t / s_t};

    const double half = 0.5;
    const cudaDoubleComplex I(0, 1);

    const int x_p = ((rank / N_sub[0]) % N_sub[1]) * s_y +
                    ((rank / (N_sub[1] * N_sub[0])) % N_sub[2]) * s_z +
                    (rank / (N_sub[2] * N_sub[1] * N_sub[0])) * s_t;

    const int s_x_cb = s_x >> 1;

    int t = (y + z + 2 * tp + x_p) % 2 != cb ? 2 * tp + 1 : 2 * tp;
    if (t >= s_t) { return; }

    int cont = s_y * s_z * tp + s_y * z + y;

    int x = s_x_cb - 1;
    cudaDoubleComplex tmp;
    cudaDoubleComplex *AO = (cudaDoubleComplex * )(U + (s_x_cb * s_y * s_z * t +
                                                      s_x_cb * s_y * z +
                                                      s_x_cb * y +
                                                      x + (1 - cb) * subgrid_vol_cb) * 9 * 2);

    cudaDoubleComplex *srcO = (cudaDoubleComplex * )(src + (s_x_cb * s_y * s_z * t +
                                                          s_x_cb * s_y * z +
                                                          s_x_cb * y + x +
                                                          (1 - cb) * subgrid_vol_cb) * 12 * 2);


//		    int b = cont * 6;

    for (int c1 = 0; c1 < 3; c1++) {
        for (int c2 = 0; c2 < 3; c2++) {
            tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half * cudaConj(AO[c2 * 3 + c1]);

            dest_f[cont * 6 * 2 + (0 * 3 + c1) * 2 + 0] += tmp.x;
            dest_f[cont * 6 * 2 + (0 * 3 + c1) * 2 + 1] += tmp.y;

            tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half * cudaConj(AO[c2 * 3 + c1]);

            dest_f[cont * 6 * 2 + (1 * 3 + c1) * 2 + 0] += tmp.x;
            dest_f[cont * 6 * 2 + (1 * 3 + c1) * 2 + 1] += tmp.y;

        }
    }

//	}}}

}

__global__ void main_x_f(double *src, double *dest, double *U,
                         const int v_x, const int v_y, const int v_z, const int v_t,
                         const int s_x, const int s_y, const int s_z, const int s_t,
                         const int rank, const int cb, const int flag) {

    int y = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.y * blockIdx.y + threadIdx.y;
    int t = blockDim.z * blockIdx.z + threadIdx.z;

    if (y >= s_y || z >= s_z || t >= s_t) { return; }

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int N_sub[4] = {v_x / s_x, v_y / s_y, v_z / s_z, v_t / s_t};

    const double half = 0.5;
    const cudaDoubleComplex I(0, 1);

    const int x_p = ((rank / N_sub[0]) % N_sub[1]) * s_y +
                    ((rank / (N_sub[1] * N_sub[0])) % N_sub[2]) * s_z +
                    (rank / (N_sub[2] * N_sub[1] * N_sub[0])) * s_t;


    const int s_x_cb = s_x >> 1;

    int x_u = ((y + z + t + x_p) % 2 == cb || N_sub[0] == 1) ? s_x_cb : s_x_cb - 1;

    for (int x = 0; x < x_u; x++) {

        cudaDoubleComplex tmp;
        int f_x = ((y + z + t + x_p) % 2 == cb) ? x : (x + 1) % s_x_cb;

        cudaDoubleComplex *srcO = (cudaDoubleComplex * )(src + (s_x_cb * s_y * s_z * t +
                                                              s_x_cb * s_y * z +
                                                              s_x_cb * y +
                                                              f_x + (1 - cb) * subgrid_vol_cb) * 12 * 2);

        double *destE = dest + (s_x_cb * s_y * s_z * t +
                                s_x_cb * s_y * z +
                                s_x_cb * y +
                                x +
                                cb * subgrid_vol_cb) * 12 * 2;

        cudaDoubleComplex *AE = (cudaDoubleComplex * )(U + (s_x_cb * s_y * s_z * t +
                                                          s_x_cb * s_y * z +
                                                          s_x_cb * y +
                                                          x +
                                                          cb * subgrid_vol_cb) * 9 * 2);


        for (int c1 = 0; c1 < 3; c1++) {
            for (int c2 = 0; c2 < 3; c2++) {

                tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half * AE[c1 * 3 + c2];
                destE[(0 * 3 + c1) * 2 + 0] += tmp.x;
                destE[(0 * 3 + c1) * 2 + 1] += tmp.y;
                destE[(3 * 3 + c1) * 2 + 0] += flag * (I * tmp).x;
                destE[(3 * 3 + c1) * 2 + 1] += flag * (I * tmp).y;
                tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half * AE[c1 * 3 + c2];
                destE[(1 * 3 + c1) * 2 + 0] += tmp.x;
                destE[(1 * 3 + c1) * 2 + 1] += tmp.y;
                destE[(2 * 3 + c1) * 2 + 0] += flag * (I * tmp).x;
                destE[(2 * 3 + c1) * 2 + 1] += flag * (I * tmp).y;

            }
        }
    }

    int x_d = (((y + z + t + x_p) % 2) != cb || N_sub[0] == 1) ? 0 : 1;

    for (int x = x_d; x < s_x_cb; x++) {
        cudaDoubleComplex tmp;
        int b_x = ((t + z + y + x_p) % 2 == cb) ? (x - 1 + s_x_cb) % s_x_cb : x;

        cudaDoubleComplex *srcO = (cudaDoubleComplex * )(src + (s_x_cb * s_y * s_z * t +
                                                              s_x_cb * s_y * z +
                                                              s_x_cb * y +
                                                              b_x +
                                                              (1 - cb) * subgrid_vol_cb) * 12 * 2);

        double *destE = dest + (s_x_cb * s_y * s_z * t +
                                s_x_cb * s_y * z +
                                s_x_cb * y +
                                x +
                                cb * subgrid_vol_cb) * 12 * 2;

        cudaDoubleComplex *AO = (cudaDoubleComplex * )(U + (s_x_cb * s_y * s_z * t +
                                                          s_x_cb * s_y * z +
                                                          s_x_cb * y +
                                                          b_x +
                                                          (1 - cb) * subgrid_vol_cb) * 9 * 2);

        for (int c1 = 0; c1 < 3; c1++) {
            for (int c2 = 0; c2 < 3; c2++) {
                tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half * cudaConj(AO[c2 * 3 + c1]);

                destE[(0 * 3 + c1) * 2 + 0] += tmp.x;
                destE[(0 * 3 + c1) * 2 + 1] += tmp.y;
                destE[(3 * 3 + c1) * 2 + 0] += flag * (-I * tmp).x;
                destE[(3 * 3 + c1) * 2 + 1] += flag * (-I * tmp).y;

                tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half * cudaConj(AO[c2 * 3 + c1]);

                destE[(1 * 3 + c1) * 2 + 0] += tmp.x;
                destE[(1 * 3 + c1) * 2 + 1] += tmp.y;
                destE[(2 * 3 + c1) * 2 + 0] += flag * (-I * tmp).x;
                destE[(2 * 3 + c1) * 2 + 1] += flag * (-I * tmp).y;
            }
        }
    }

//            }
//        }
//    }	
}

__global__ void ghost_x_f(double *src_f, double *dest, double *U,
                          const int v_x, const int v_y, const int v_z, const int v_t,
                          const int s_x, const int s_y, const int s_z, const int s_t,
                          const int rank, const int cb, const int flag) {

    int y = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.y * blockIdx.y + threadIdx.y;
    int tp = blockDim.z * blockIdx.z + threadIdx.z;

    if (y >= s_y || z >= s_z) { return; }

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int N_sub[4] = {v_x / s_x, v_y / s_y, v_z / s_z, v_t / s_t};

    const double half = 0.5;
    const cudaDoubleComplex I(0, 1);

    const int x_p = ((rank / N_sub[0]) % N_sub[1]) * s_y +
                    ((rank / (N_sub[1] * N_sub[0])) % N_sub[2]) * s_z +
                    (rank / (N_sub[2] * N_sub[1] * N_sub[0])) * s_t;

    int t = (y + z + 2 * tp + x_p) % 2 == cb ? 2 * tp + 1 : 2 * tp;
    if (t >= s_t) { return; }

    const int s_x_cb = s_x >> 1;

    int cont = s_y * s_z * tp + s_y * z + y;

    cudaDoubleComplex tmp;

    int x = s_x_cb - 1;

    cudaDoubleComplex *srcO = (cudaDoubleComplex * )(&src_f[cont * 6 * 2]);

    double *destE = dest + (s_x_cb * s_y * s_z * t +
                            s_x_cb * s_y * z +
                            s_x_cb * y +
                            x +
                            cb * subgrid_vol_cb) * 12 * 2;

    cudaDoubleComplex *AE = (cudaDoubleComplex * )(U + (s_x_cb * s_y * s_z * t +
                                                      s_x_cb * s_y * z +
                                                      s_x_cb * y +
                                                      x +
                                                      cb * subgrid_vol_cb) * 9 * 2);


    for (int c1 = 0; c1 < 3; c1++) {
        for (int c2 = 0; c2 < 3; c2++) {
            tmp = srcO[0 * 3 + c2] * AE[c1 * 3 + c2];
            destE[(0 * 3 + c1) * 2 + 0] += tmp.x;
            destE[(0 * 3 + c1) * 2 + 1] += tmp.y;
            destE[(3 * 3 + c1) * 2 + 0] += flag * (I * tmp).x;
            destE[(3 * 3 + c1) * 2 + 1] += flag * (I * tmp).y;
            tmp = srcO[1 * 3 + c2] * AE[c1 * 3 + c2];
            destE[(1 * 3 + c1) * 2 + 0] += tmp.x;
            destE[(1 * 3 + c1) * 2 + 1] += tmp.y;
            destE[(2 * 3 + c1) * 2 + 0] += flag * (I * tmp).x;
            destE[(2 * 3 + c1) * 2 + 1] += flag * (I * tmp).y;
        }
    }
}

__global__ void ghost_x_b(double *src_b, double *dest, double *U,
                          const int v_x, const int v_y, const int v_z, const int v_t,
                          const int s_x, const int s_y, const int s_z, const int s_t,
                          const int rank, const int cb, const int flag) {

    int y = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.y * blockIdx.y + threadIdx.y;
    int tp = blockDim.z * blockIdx.z + threadIdx.z;

    if (y >= s_y || z >= s_z) { return; }

    int subgrid_vol = (s_x * s_y * s_z * s_t);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    int N_sub[4] = {v_x / s_x, v_y / s_y, v_z / s_z, v_t / s_t};

    const double half = 0.5;
    const cudaDoubleComplex I(0, 1);

    const int x_p = ((rank / N_sub[0]) % N_sub[1]) * s_y +
                    ((rank / (N_sub[1] * N_sub[0])) % N_sub[2]) * s_z +
                    (rank / (N_sub[2] * N_sub[1] * N_sub[0])) * s_t;

    int t = (y + z + 2 * tp + x_p) % 2 != cb ? 2 * tp + 1 : 2 * tp;
    if (t >= s_t) { return; }

    const int s_x_cb = s_x >> 1;

    int cont = s_y * s_z * tp + s_y * z + y;

    int x = 0;

    cudaDoubleComplex *srcO = (cudaDoubleComplex * )(&src_b[cont * 6 * 2]);
    double *destE = dest + (s_x_cb * s_y * s_z * t +
                            s_x_cb * s_y * z +
                            s_x_cb * y +
                            x + cb * subgrid_vol_cb) * 12 * 2;

    for (int c1 = 0; c1 < 3; c1++) {
        destE[(0 * 3 + c1) * 2 + 0] += srcO[0 * 3 + c1].x;
        destE[(0 * 3 + c1) * 2 + 1] += srcO[0 * 3 + c1].y;
        destE[(3 * 3 + c1) * 2 + 0] += flag * (-I * srcO[0 * 3 + c1]).x;
        destE[(3 * 3 + c1) * 2 + 1] += flag * (-I * srcO[0 * 3 + c1]).y;
        destE[(1 * 3 + c1) * 2 + 0] += srcO[1 * 3 + c1].x;
        destE[(1 * 3 + c1) * 2 + 1] += srcO[1 * 3 + c1].y;
        destE[(2 * 3 + c1) * 2 + 0] += flag * (-I * srcO[1 * 3 + c1]).x;
        destE[(2 * 3 + c1) * 2 + 1] += flag * (-I * srcO[1 * 3 + c1]).y;
    }
}

int test_2(double *src_g, double *dest_g, double *U_g,
           const int v_x, const int v_y, const int v_z, const int v_t,
           const int s_x, const int s_y, const int s_z, const int s_t,
           const int cb, const int flag) {

//cudaDeviceProp_t devProp;
//cudaGetDeviceProperties(&devProp, 0);

//std::cout << "Device name " << devProp.name << std::endl;

    int N_sub[4] = {v_x / s_x, v_y / s_y, v_z / s_z, v_t / s_t};


    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Request req[8 * size];
    MPI_Request reqr[8 * size];
    MPI_Status sta[8 * size];


    int site_x_f[4] = {(rank % N_sub[0] + 1) % N_sub[0],
                       (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    int site_x_b[4] = {(rank % N_sub[0] - 1 + N_sub[0]) % N_sub[0],
                       (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    const int nodenum_x_b = get_nodenum(site_x_b, N_sub, 4);
    const int nodenum_x_f = get_nodenum(site_x_f, N_sub, 4);

/*
double * src_g;
double * dest_g;
double * U_g;

int size_f = s_x * s_y * s_z * s_t * 12 * 2;
int size_u = s_x * s_y * s_z * s_t * 9 * 2;
//    allocate the memory on the device side
cudaMalloc((void**)&src_g,   size_f * sizeof(double));
cudaMalloc((void**)&dest_g,  size_f * sizeof(double));
cudaMalloc((void**)&U_g,     size_u * sizeof(double));

cudaMemcpy(src_g,  src,  size_f * sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(U_g,    U,    size_u * sizeof(double), cudaMemcpyHostToDevice); 
cudaMemset(dest_g, 0, size_f * sizeof(double));
*/


    int len_x = (s_y * s_z * s_t + cb) >> 1;

    double *resv_x_f = new double[len_x * 6 * 2];
    double *send_x_b = new double[len_x * 6 * 2];
    double *resv_x_b = new double[len_x * 6 * 2];
    double *send_x_f = new double[len_x * 6 * 2];

    if (N_sub[0] != 1) {
        double *tran_f;
        double *tran_b;
        int size_T = len_x * 6 * 2;

        cudaMalloc((void **) &tran_f, size_T * sizeof(double));
        cudaMemset(tran_f, 0, size_T * sizeof(double));

        cudaMalloc((void **) &tran_b, size_T * sizeof(double));
        cudaMemset(tran_b, 0, size_T * sizeof(double));

        transfer_x_f<<<dim3(s_y / 2, s_z / 2, s_t / 2), dim3(2, 2, 1) >> >
                                                        (src_g, tran_b, U_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
        transfer_x_b<<<dim3(s_y / 2, s_z / 2, s_t / 2), dim3(2, 2, 1) >> >
                                                        (src_g, tran_f, U_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        cudaMemcpy(send_x_b, tran_b, size_T * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(send_x_f, tran_f, size_T * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(tran_f);
        cudaFree(tran_b);

        MPI_Isend(send_x_b, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_b, 8 * rank, MPI_COMM_WORLD, &req[8 * rank]);
        MPI_Irecv(resv_x_f, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_f, 8 * nodenum_x_f, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_x_f]);

        MPI_Isend(send_x_f, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_f, 8 * rank + 1, MPI_COMM_WORLD, &req[8 * rank + 1]);
        MPI_Irecv(resv_x_b, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_b, 8 * nodenum_x_b + 1, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_x_b + 1]);

    }

    main_x_f<<<dim3(s_y / 2, s_z / 2, s_t), dim3(2, 2, 1) >> >
                                            (src_g, dest_g, U_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

    if (N_sub[0] != 1) {

        MPI_Wait(&reqr[8 * nodenum_x_f], &sta[8 * nodenum_x_f]);
        MPI_Wait(&reqr[8 * nodenum_x_b + 1], &sta[8 * nodenum_x_b + 1]);


        double *tran_f;
        double *tran_b;
        int size_T = len_x * 6 * 2;
        cudaMalloc((void **) &tran_f, size_T * sizeof(double));
        cudaMalloc((void **) &tran_b, size_T * sizeof(double));

        cudaMemcpy(tran_f, resv_x_f, size_T * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(tran_b, resv_x_b, size_T * sizeof(double), cudaMemcpyHostToDevice);

        ghost_x_f<<<dim3(s_y / 2, s_z / 2, s_t / 2), dim3(2, 2, 1) >> >
                                                     (tran_f, dest_g, U_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
        ghost_x_b<<<dim3(s_y / 2, s_z / 2, s_t / 2), dim3(2, 2, 1) >> >
                                                     (tran_b, dest_g, U_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        cudaFree(tran_f);
        cudaFree(tran_b);
    }

//    cudaMemcpy(dest, dest_g, size_f * sizeof(double), cudaMemcpyDeviceToHost);	

//    cudaFree(src_g);
//    cudaFree(dest_g);
//    cudaFree(U_g);	

    MPI_Barrier(MPI_COMM_WORLD);

    delete[] resv_x_f;
    delete[] send_x_b;
    delete[] resv_x_b;
    delete[] send_x_f;

    printf("PASSED!\n");

    return 0;
}

int test(double *src, double *dest, double *U,
         const int v_x, const int v_y, const int v_z, const int v_t,
         const int s_x, const int s_y, const int s_z, const int s_t,
         const int cb, const int flag) {

    cudaDeviceProp_t devProp;
    cudaGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    double *src_g;
    double *dest_g;
    double *U_g;

    int size_f = s_x * s_y * s_z * s_t * 12 * 2;
    int size_u = s_x * s_y * s_z * s_t * 9 * 2;
//    allocate the memory on the device side
    cudaMalloc((void **) &src_g, size_f * sizeof(double));
    cudaMalloc((void **) &dest_g, size_f * sizeof(double));
    cudaMalloc((void **) &U_g, size_u * sizeof(double));

    cudaMemcpy(src_g, src, size_f * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(U_g, U, size_u * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(dest_g, 0, size_f * sizeof(double));

    test_2(src_g, dest_g, U_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, cb, flag);

    cudaMemcpy(dest, dest_g, size_f * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(src_g);
    cudaFree(dest_g);
    cudaFree(U_g);

    printf("test PASSED!\n");

    return 0;

}

