//
// Created by louis on 2021/8/29.
//

#include <iostream>
#include <mpi.h>
#include <vector> 

#include "hip/hip_runtime.h"
#include "hip/hip_complex.h"

#include "operator.h"
#include "target/hip/legacyFunction.h"
#include "target/hip/transfer_kernel.h"
#include "target/hip/dslash_main_kernel.h"
#include "target/hip/ghost_kernel.h"
#include "target/hip/dot_product_kernel.h"
#include "target/hip/axpb_kernel.h"
#include "target/hip/psi_kernel.h"
#include "target/hip/operator_kernel.h"

#include "target/hip/DiracWilson.h"
#include "target/hip/DiracOverlapWilson.h"

const int Dx=2;
const int Dy=8;
const int Dz=8;
const int Dt=4;

inline int test_2(double *src_g, double *dest_g, double *U_x_g, double *U_y_g, double *U_z_g, double *U_t_g,
                  const int v_x, const int v_y, const int v_z, const int v_t,
                  const int s_x, const int s_y, const int s_z, const int s_t,
                  const int cb, const int flag) {

//hipDeviceProp_t devProp;
//hipGetDeviceProperties(&devProp, 0);

//std::cout << "Device name " << devProp.name << std::endl;

//    int rank =0;
//    int s_x_cb = s_x >> 1;	


    int N_sub[4] = {v_x / s_x, v_y / s_y, v_z / s_z, v_t / s_t};

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Request req[8 * size];
    MPI_Request reqr[8 * size];
    MPI_Status sta[8 * size];


    int site_x_f[4] = {(rank + 1) % N_sub[0],
                       (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    int site_x_b[4] = {(rank - 1 + N_sub[0]) % N_sub[0],
                       (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    const int nodenum_x_b = get_nodenum(site_x_b, N_sub, 4);
    const int nodenum_x_f = get_nodenum(site_x_f, N_sub, 4);

    int site_y_f[4] = {(rank % N_sub[0]),
                       (rank / N_sub[0] + 1) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    int site_y_b[4] = {(rank % N_sub[0]),
                       (rank / N_sub[0] - 1 + N_sub[1]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    const int nodenum_y_b = get_nodenum(site_y_b, N_sub, 4);
    const int nodenum_y_f = get_nodenum(site_y_f, N_sub, 4);

    int site_z_f[4] = {(rank % N_sub[0]),
                       (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0]) + 1) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    int site_z_b[4] = {(rank % N_sub[0]),
                       (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0]) - 1 + N_sub[2]) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    const int nodenum_z_b = get_nodenum(site_z_b, N_sub, 4);
    const int nodenum_z_f = get_nodenum(site_z_f, N_sub, 4);

    int site_t_f[4] = {(rank % N_sub[0]),
                       (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       (rank / (N_sub[2] * N_sub[1] * N_sub[0]) + 1) % N_sub[3]};

    int site_t_b[4] = {(rank % N_sub[0]),
                       (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       (rank / (N_sub[2] * N_sub[1] * N_sub[0]) - 1 + N_sub[3]) %  N_sub[3]};

    const int nodenum_t_b = get_nodenum(site_t_b, N_sub, 4);
    const int nodenum_t_f = get_nodenum(site_t_f, N_sub, 4);	

    int s_x_cb = s_x >> 1;

    int len_x = (s_y * s_z * s_t + cb) >> 1;

    double *resv_x_f ;
    double *send_x_b ;
    double *resv_x_b ;
    double *send_x_f ;


//    resv_x_f = new double[len_x * 6 * 2];
//    send_x_b = new double[len_x * 6 * 2];
//    resv_x_b = new double[len_x * 6 * 2];
//    send_x_f = new double[len_x * 6 * 2];



    if (N_sub[0] != 1) {

        resv_x_f = new double[len_x * 6 * 2];
        send_x_b = new double[len_x * 6 * 2];
        resv_x_b = new double[len_x * 6 * 2];
        send_x_f = new double[len_x * 6 * 2];

        double *tran_f;
        double *tran_b;
        int size_T = len_x * 6 * 2;

        hipMalloc((void **) &tran_f, size_T * sizeof(double));
        hipMemset(tran_f, 0, size_T * sizeof(double));

        hipMalloc((void **) &tran_b, size_T * sizeof(double));
        hipMemset(tran_b, 0, size_T * sizeof(double));

        transfer_x_f<<<dim3(s_y / Dy, s_z / Dz, s_t / Dt / 2), dim3(Dy, Dz, Dt) >>>
                                                        (src_g, tran_b, U_x_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
        transfer_x_b<<<dim3(s_y / Dy, s_z / Dz, s_t / Dt / 2), dim3(Dy, Dz, Dt) >>>
                                                        (src_g, tran_f, U_x_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        hipMemcpy(send_x_b, tran_b, size_T * sizeof(double), hipMemcpyDeviceToHost);
        hipMemcpy(send_x_f, tran_f, size_T * sizeof(double), hipMemcpyDeviceToHost);

        hipFree(tran_f);
        hipFree(tran_b);

        MPI_Isend(send_x_b, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_b, 8 * rank, MPI_COMM_WORLD, &req[8 * rank]);
        MPI_Irecv(resv_x_f, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_f, 8 * nodenum_x_f, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_x_f]);

        MPI_Isend(send_x_f, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_f, 8 * rank + 1, MPI_COMM_WORLD, &req[8 * rank + 1]);
        MPI_Irecv(resv_x_b, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_b, 8 * nodenum_x_b + 1, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_x_b + 1]);

    }

    int len_y = s_x_cb * s_z * s_t;



    double *resv_y_f;
    double *send_y_b;
    double *resv_y_b;
    double *send_y_f;

//    resv_y_f = new double[len_y * 6 * 2];
//    send_y_b = new double[len_y * 6 * 2];
//    resv_y_b = new double[len_y * 6 * 2];
//    send_y_f = new double[len_y * 6 * 2];

    if (N_sub[1] != 1) {

        resv_y_f = new double[len_y * 6 * 2];
        send_y_b = new double[len_y * 6 * 2];
        resv_y_b = new double[len_y * 6 * 2];
        send_y_f = new double[len_y * 6 * 2];

        double *tran_f;
        double *tran_b;
        int size_T = len_y * 6 * 2;

        hipMalloc((void **) &tran_f, size_T * sizeof(double));
        hipMemset(tran_f, 0, size_T * sizeof(double));

        hipMalloc((void **) &tran_b, size_T * sizeof(double));
        hipMemset(tran_b, 0, size_T * sizeof(double));

        transfer_y_f<<<dim3(s_x_cb / Dx, s_z / Dz, s_t / Dt), dim3(Dx, Dz, Dt) >>>(src_g, tran_b, U_y_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
        transfer_y_b<<<dim3(s_x_cb / Dx, s_z / Dz, s_t / Dt), dim3(Dx, Dz, Dt) >>>(src_g, tran_f, U_y_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        hipMemcpy(send_y_b, tran_b, size_T * sizeof(double), hipMemcpyDeviceToHost);
        hipMemcpy(send_y_f, tran_f, size_T * sizeof(double), hipMemcpyDeviceToHost);

        hipFree(tran_f);
        hipFree(tran_b);

        MPI_Isend(send_y_b, len_y * 6 * 2, MPI_DOUBLE, nodenum_y_b, 8 * rank + 2, MPI_COMM_WORLD, &req[8 * rank + 2]);
        MPI_Irecv(resv_y_f, len_y * 6 * 2, MPI_DOUBLE, nodenum_y_f, 8 * nodenum_y_f + 2, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_y_f + 2]);

        MPI_Isend(send_y_f, len_y * 6 * 2, MPI_DOUBLE, nodenum_y_f, 8 * rank + 3, MPI_COMM_WORLD, &req[8 * rank + 3]);
        MPI_Irecv(resv_y_b, len_y * 6 * 2, MPI_DOUBLE, nodenum_y_b, 8 * nodenum_y_b + 3, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_y_b + 3]);

    }

    int len_z = s_x_cb * s_y * s_t;

    double *resv_z_f;
    double *send_z_b;
    double *resv_z_b;
    double *send_z_f;

//    resv_z_f = new double[len_z * 6 * 2];
//    send_z_b = new double[len_z * 6 * 2];
//    resv_z_b = new double[len_z * 6 * 2];
//    send_z_f = new double[len_z * 6 * 2];

    if (N_sub[2] != 1) {

        resv_z_f = new double[len_z * 6 * 2];
        send_z_b = new double[len_z * 6 * 2];
        resv_z_b = new double[len_z * 6 * 2];
        send_z_f = new double[len_z * 6 * 2];

        double *tran_f;
        double *tran_b;
        int size_T = len_z * 6 * 2;

        hipMalloc((void **) &tran_f, size_T * sizeof(double));
        hipMemset(tran_f, 0, size_T * sizeof(double));

        hipMalloc((void **) &tran_b, size_T * sizeof(double));
        hipMemset(tran_b, 0, size_T * sizeof(double));

        transfer_z_f<<<dim3(s_x_cb / Dx, s_y / Dy, s_t / Dt), dim3(Dx, Dy, Dt) >>>
                                                       (src_g, tran_b, U_z_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
        transfer_z_b<<<dim3(s_x_cb / Dx, s_y / Dy, s_t / Dt), dim3(Dx, Dy, Dt) >>>
                                                       (src_g, tran_f, U_z_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        hipMemcpy(send_z_b, tran_b, size_T * sizeof(double), hipMemcpyDeviceToHost);
        hipMemcpy(send_z_f, tran_f, size_T * sizeof(double), hipMemcpyDeviceToHost);

        hipFree(tran_f);
        hipFree(tran_b);

        MPI_Isend(send_z_b, len_z * 6 * 2, MPI_DOUBLE, nodenum_z_b, 8 * rank + 4, MPI_COMM_WORLD, &req[8 * rank + 4]);
        MPI_Irecv(resv_z_f, len_z * 6 * 2, MPI_DOUBLE, nodenum_z_f, 8 * nodenum_z_f + 4, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_z_f + 4]);

        MPI_Isend(send_z_f, len_z * 6 * 2, MPI_DOUBLE, nodenum_z_f, 8 * rank + 5, MPI_COMM_WORLD, &req[8 * rank + 5]);
        MPI_Irecv(resv_z_b, len_z * 6 * 2, MPI_DOUBLE, nodenum_z_b, 8 * nodenum_z_b + 5, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_z_b + 5]);

    }	

    int len_t = s_x_cb * s_y * s_z;

    double *resv_t_f;
    double *send_t_b;
    double *resv_t_b;
    double *send_t_f;	

//    resv_t_f = new double[len_t * 6 * 2];
//    send_t_b = new double[len_t * 6 * 2];
//    resv_t_b = new double[len_t * 6 * 2];
//    send_t_f = new double[len_t * 6 * 2];

    if (N_sub[3] != 1) {

        resv_t_f = new double[len_t * 6 * 2];
        send_t_b = new double[len_t * 6 * 2];
        resv_t_b = new double[len_t * 6 * 2];
        send_t_f = new double[len_t * 6 * 2];

        double *tran_f;
        double *tran_b;
        int size_T = len_t * 6 * 2;

        hipMalloc((void **) &tran_f, size_T * sizeof(double));
//        hipMemset(tran_f, 0, size_T * sizeof(double));

        hipMalloc((void **) &tran_b, size_T * sizeof(double));
//        hipMemset(tran_b, 0, size_T * sizeof(double));

//        transfer_t_f<<<dim3(s_x_cb / Dx, s_y / Dy, s_z / Dz), dim3(Dx, Dy, Dz) >>>
//                                                       (src_g, tran_b, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);


        transfer_t_f<<<s_x_cb * s_y * s_z / Dx / Dy / Dz, Dx * Dy * Dz >>>
                                                          (src_g, tran_b, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);


//        transfer_t_b<<<dim3(s_x_cb / Dx, s_y / Dy, s_z / Dz), dim3(Dx, Dy, Dz) >>>
//                                                       (src_g, tran_f, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);


        transfer_t_b<<<s_x_cb * s_y * s_z / Dx / Dy / Dz, Dx * Dy * Dz >>>
                                                          (src_g, tran_f, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        hipMemcpy(send_t_b, tran_b, size_T * sizeof(double), hipMemcpyDeviceToHost);
        hipMemcpy(send_t_f, tran_f, size_T * sizeof(double), hipMemcpyDeviceToHost);

        hipFree(tran_f);
        hipFree(tran_b);

        MPI_Isend(send_t_b, len_t * 6 * 2, MPI_DOUBLE, nodenum_t_b, 8 * rank + 6, MPI_COMM_WORLD, &req[8 * rank + 6]);
        MPI_Irecv(resv_t_f, len_t * 6 * 2, MPI_DOUBLE, nodenum_t_f, 8 * nodenum_t_f + 6, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_t_f + 6]);

        MPI_Isend(send_t_f, len_t * 6 * 2, MPI_DOUBLE, nodenum_t_f, 8 * rank + 7, MPI_COMM_WORLD, &req[8 * rank + 7]);
        MPI_Irecv(resv_t_b, len_t * 6 * 2, MPI_DOUBLE, nodenum_t_b, 8 * nodenum_t_b + 7, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_t_b + 7]);

    }

/* 	
    float eventMs = 1.0f;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);
*/


    main_xyzt<<< s_x_cb * s_y * s_z * s_t / Dx / Dy / Dz , Dx * Dy * Dz >>>
                                             (src_g, dest_g, U_x_g, U_y_g, U_z_g, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);	

/*
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("main_xyzt time taken  = %6.3fms\n", eventMs);    	
*/

    if (N_sub[0] != 1) {

        MPI_Wait(&reqr[8 * nodenum_x_f], &sta[8 * nodenum_x_f]);
        MPI_Wait(&reqr[8 * nodenum_x_b + 1], &sta[8 * nodenum_x_b + 1]);

        double *tran_f;
        double *tran_b;
        int size_T = len_x * 6 * 2;
        hipMalloc((void **) &tran_f, size_T * sizeof(double));
        hipMalloc((void **) &tran_b, size_T * sizeof(double));

        hipMemcpy(tran_f, resv_x_f, size_T * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(tran_b, resv_x_b, size_T * sizeof(double), hipMemcpyHostToDevice);

        ghost_x_f<<<dim3(s_y / Dy, s_z / Dz, s_t / Dt / 2), dim3(Dy, Dz, Dt) >>>
                                                     (tran_f, dest_g, U_x_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
        ghost_x_b<<<dim3(s_y / Dy, s_z / Dz, s_t / Dt / 2), dim3(Dy, Dz, Dt) >>>
                                                     (tran_b, dest_g, U_x_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        hipFree(tran_f);
        hipFree(tran_b);
    }


    if (N_sub[1] != 1) {

        MPI_Wait(&reqr[8 * nodenum_y_f + 2], &sta[8 * nodenum_y_f + 2]);
        MPI_Wait(&reqr[8 * nodenum_y_b + 3], &sta[8 * nodenum_y_b + 3]);

        double *tran_f;
        double *tran_b;
        int size_T = len_y * 6 * 2;
        hipMalloc((void **) &tran_f, size_T * sizeof(double));
        hipMalloc((void **) &tran_b, size_T * sizeof(double));

        hipMemcpy(tran_f, resv_y_f, size_T * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(tran_b, resv_y_b, size_T * sizeof(double), hipMemcpyHostToDevice);

        ghost_y_f<<<dim3(s_x_cb / Dx, s_z / Dz, s_t / Dt), dim3(Dx, Dz, Dt) >>>
                                                    (tran_f, dest_g, U_y_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
        ghost_y_b<<<dim3(s_x_cb / Dx, s_z / Dz, s_t / Dt), dim3(Dx, Dz, Dt) >>>
                                                    (tran_b, dest_g, U_y_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        hipFree(tran_f);
        hipFree(tran_b);
    }

    if (N_sub[2] != 1) {

        MPI_Wait(&reqr[8 * nodenum_z_f + 4], &sta[8 * nodenum_z_f + 4]);
        MPI_Wait(&reqr[8 * nodenum_z_b + 5], &sta[8 * nodenum_z_b + 5]);

        double *tran_f;
        double *tran_b;
        int size_T = len_z * 6 * 2;
        hipMalloc((void **) &tran_f, size_T * sizeof(double));
        hipMalloc((void **) &tran_b, size_T * sizeof(double));

        hipMemcpy(tran_f, resv_z_f, size_T * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(tran_b, resv_z_b, size_T * sizeof(double), hipMemcpyHostToDevice);

        ghost_z_f<<<dim3(s_x_cb / Dx, s_y / Dy, s_t / Dt), dim3(Dx, Dy, Dt) >>>
                                                    (tran_f, dest_g, U_z_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
        ghost_z_b<<<dim3(s_x_cb / Dx, s_y / Dy, s_t / Dt), dim3(Dx, Dy, Dt) >>>
                                                    (tran_b, dest_g, U_z_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        hipFree(tran_f);
        hipFree(tran_b);
    }

    if (N_sub[3] != 1) {

        MPI_Wait(&reqr[8 * nodenum_t_f + 6], &sta[8 * nodenum_t_f + 6]);
        MPI_Wait(&reqr[8 * nodenum_t_b + 7], &sta[8 * nodenum_t_b + 7]);

        double *tran_f;
        double *tran_b;
        int size_T = len_t * 6 * 2;
        hipMalloc((void **) &tran_f, size_T * sizeof(double));
        hipMalloc((void **) &tran_b, size_T * sizeof(double));

        hipMemcpy(tran_f, resv_t_f, size_T * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(tran_b, resv_t_b, size_T * sizeof(double), hipMemcpyHostToDevice);

//        ghost_t_f<<<dim3(s_x_cb / Dx, s_y / Dy, s_z / Dz), dim3(Dx, Dy, Dz) >>>
//                                                    (tran_f, dest_g, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);



        ghost_t_f<<<s_x_cb * s_y * s_z / Dx / Dy / Dz, Dx * Dy * Dz >>>
                                                       (tran_f, dest_g, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

//        ghost_t_b<<<dim3(s_x_cb / Dx, s_y / Dy, s_z / Dz), dim3(Dx, Dy, Dz) >>>
//                                                    (tran_b, dest_g, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);


        ghost_t_b<<<s_x_cb * s_y * s_z / Dx / Dy / Dz, Dx * Dy * Dz >>>
                                                       (tran_b, dest_g, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);


        hipFree(tran_f);
        hipFree(tran_b);

    }


    MPI_Barrier(MPI_COMM_WORLD);

    if (N_sub[0] != 1) {
        delete[] resv_x_f;
        delete[] send_x_b;
        delete[] resv_x_b;
        delete[] send_x_f;
    }

    if (N_sub[1] != 1) {
        delete[] resv_y_f;
        delete[] send_y_b;
        delete[] resv_y_b;
        delete[] send_y_f;
    }

    if (N_sub[2] != 1) {
        delete[] resv_z_f;
        delete[] send_z_b;
        delete[] resv_z_b;
        delete[] send_z_f;
    }

    if (N_sub[3] != 1) {
        delete[] resv_t_f;
        delete[] send_t_b;
        delete[] resv_t_b;
        delete[] send_t_f;
    }
//    printf("PASSED!\n");

    return 0;
}

int test(double *src, double *dest, double *U_x, double *U_y, double *U_z, double *U_t,
         const int v_x, const int v_y, const int v_z, const int v_t,
         const int s_x, const int s_y, const int s_z, const int s_t,
         const int cb, const int flag) {

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

//    std::cout << "Device name " << devProp.name << std::endl;

    double *src_g;
    double *dest_g;
    double *U_x_g;
    double *U_y_g;
    double *U_z_g;
    double *U_t_g;

    int size_f = s_x * s_y * s_z * s_t * 12 * 2;
    int size_u = s_x * s_y * s_z * s_t * 9 * 2;
//    allocate the memory on the device side
    hipMalloc((void **) &src_g, size_f * sizeof(double));
    hipMalloc((void **) &dest_g, size_f * sizeof(double));
    hipMalloc((void **) &U_x_g, size_u * sizeof(double));
    hipMalloc((void **) &U_y_g, size_u * sizeof(double));
    hipMalloc((void **) &U_z_g, size_u * sizeof(double));
    hipMalloc((void **) &U_t_g, size_u * sizeof(double));

    hipMemcpy(src_g, src, size_f * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(U_x_g, U_x, size_u * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(U_y_g, U_y, size_u * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(U_z_g, U_z, size_u * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(U_t_g, U_t, size_u * sizeof(double), hipMemcpyHostToDevice);
    hipMemset(dest_g, 0, size_f * sizeof(double));

    test_2(src_g, dest_g, U_x_g, U_y_g, U_z_g, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, cb, flag);

//    hipDeviceSynchronize();

    hipMemcpy(dest, dest_g, size_f * sizeof(double), hipMemcpyDeviceToHost);

    hipFree(src_g);
    hipFree(dest_g);
    hipFree(U_x_g);
    hipFree(U_y_g);
    hipFree(U_z_g);
    hipFree(U_t_g);

//    printf("test PASSED!\n");
    return 0;
}

void *newApplyOverlapQuda(std::vector<double *> &evecs, std::vector<double> &evals,
                          std::vector <std::vector<double>> &coefs, std::vector<int> &sizes,
                          double *U_x, double *U_y, double *U_z, double *U_t,
                          const int v_x, const int v_y, const int v_z, const int v_t,
                          const int s_x, const int s_y, const int s_z, const int s_t,
                          const double kappa) {

    DiracOverlapWilson *ov_instance = new DiracOverlapWilson(evecs, evals, coefs, sizes, U_x, U_y, U_z, U_t, v_x, v_y,
                                                             v_z, v_t, s_x, s_y, s_z, s_t, kappa);

    return static_cast<void *>(ov_instance);

}

void delApplyOverlapQuda(void *ov_instance) {

    delete static_cast<DiracOverlapWilson *>(ov_instance);
}

void ApplyOverlapQuda(double *dest, double *src,
                      double k0, double k1, double k2, double prec,
                      void *ov_instance, int size) {

    DiracOverlapWilson *dirac = static_cast<DiracOverlapWilson *>(ov_instance);

    double *src_g;
    double *dest_g;

    int size_f = size * 12 * 2;

    hipMalloc((void **) &src_g, size_f * sizeof(double));
    hipMalloc((void **) &dest_g, size_f * sizeof(double));

    hipMemcpy(src_g, src, size_f * sizeof(double), hipMemcpyHostToDevice);
    hipMemset(dest_g, 0, size_f * sizeof(double));

    dirac->general_dov(dest_g, src_g, k0, k1, k2, prec);

    hipMemcpy(dest, dest_g, size_f * sizeof(double), hipMemcpyDeviceToHost);

    hipFree(src_g);
    hipFree(dest_g);

}

void *newApplyWilsonQuda(double *U_x, double *U_y, double *U_z, double *U_t,
                         const int v_x, const int v_y, const int v_z, const int v_t,
                         const int s_x, const int s_y, const int s_z, const int s_t) {

    DiracWilson *ov_instance = new DiracWilson(U_x, U_y, U_z, U_t, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t);
    return static_cast<void *>(ov_instance);

}

void delApplyWilsonQuda(void *ov_instance) {
    delete static_cast<DiracWilson *>(ov_instance);
}

void ApplyWilsonQuda(double *dest, double *src, void *ov_instance) {

    DiracWilson *dirac = static_cast<DiracWilson *>(ov_instance);

    double *src_g;
    double *dest_g;

    int size_f = dirac->volume * 12 * 2;

    hipMalloc((void **) &src_g, size_f * sizeof(double));
    hipMalloc((void **) &dest_g, size_f * sizeof(double));

    hipMemcpy(src_g, src, size_f * sizeof(double), hipMemcpyHostToDevice);
    //hipMemset(dest_g, 0, size_f * sizeof(double));

    dirac->Kernal(dest_g, src_g);

    hipMemcpy(dest, dest_g, size_f * sizeof(double), hipMemcpyDeviceToHost);

    hipFree(src_g);
    hipFree(dest_g);

}

void ApplyOverlapQuda(double *dest, double *src, double k0, double k1, double k2, double prec,
                      std::vector<double *> &evecs, std::vector<double> &evals,
                      std::vector<std::vector<double> > &coefs, std::vector<int> &sizes,
                      double *U_x, double *U_y, double *U_z, double *U_t,
                      const int v_x, const int v_y, const int v_z, const int v_t,
                      const int s_x, const int s_y, const int s_z, const int s_t,
                      const double kappa) {

    double *src_g;
    double *dest_g;

    int size_f = s_x * s_y * s_z * s_t * 12 * 2;

    hipMalloc((void **) &src_g, size_f * sizeof(double));
    hipMalloc((void **) &dest_g, size_f * sizeof(double));

    hipMemcpy(src_g, src, size_f * sizeof(double), hipMemcpyHostToDevice);

    hipMemset(dest_g, 0, size_f * sizeof(double));

    DiracOverlapWilson OW(evecs, evals, coefs, sizes, U_x, U_y, U_z, U_t, v_x, v_y, v_z, v_t, s_x, s_y, s_z,
                          s_t, kappa);

    OW.general_dov(dest_g, src_g, k0, k1, k2, prec);

    hipMemcpy(dest, dest_g, size_f * sizeof(double), hipMemcpyDeviceToHost);

    hipFree(src_g);
    hipFree(dest_g);
}

void dslash_g5_kernal(double *src, double *dest, double *U_x, double *U_y, double *U_z, double *U_t,
                      const int v_x, const int v_y, const int v_z, const int v_t,
                      const int s_x, const int s_y, const int s_z, const int s_t,
                      const double a, const int flag) {

    test_2(src, dest, U_x, U_y, U_z, U_t, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, 0, flag);
    test_2(src, dest, U_x, U_y, U_z, U_t, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, 1, flag);

    axpbyz_g<<<dim3(s_x / 2, s_y / 2, s_z / 2), dim3(2, 2, 2) >>> (1.0, src, a, dest, dest, s_x, s_y, s_z, s_t);

    psi_g5<<<dim3(s_x / 2, s_y / 2, s_z / 2), dim3(2, 2, 2) >>> (dest, s_x, s_y, s_z, s_t);

}


int dslash_g5(double *src, double *dest, double *U_x, double *U_y, double *U_z, double *U_t,
              const int v_x, const int v_y, const int v_z, const int v_t,
              const int s_x, const int s_y, const int s_z, const int s_t,
              const double a, const int flag) {

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

//    std::cout << "Device name " << devProp.name << std::endl;

    double *src_g;
    double *dest_g;
    double *U_x_g;
    double *U_y_g;
    double *U_z_g;
    double *U_t_g;

    int size_f = s_x * s_y * s_z * s_t * 12 * 2;
    int size_u = s_x * s_y * s_z * s_t * 9 * 2;
//    allocate the memory on the device side
    hipMalloc((void **) &src_g, size_f * sizeof(double));
    hipMalloc((void **) &dest_g, size_f * sizeof(double));
    hipMalloc((void **) &U_x_g, size_u * sizeof(double));
    hipMalloc((void **) &U_y_g, size_u * sizeof(double));
    hipMalloc((void **) &U_z_g, size_u * sizeof(double));
    hipMalloc((void **) &U_t_g, size_u * sizeof(double));

    hipMemcpy(src_g, src, size_f * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(U_x_g, U_x, size_u * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(U_y_g, U_y, size_u * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(U_z_g, U_z, size_u * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(U_t_g, U_t, size_u * sizeof(double), hipMemcpyHostToDevice);
    hipMemset(dest_g, 0, size_f * sizeof(double));

    dslash_g5_kernal(src_g, dest_g, U_x_g, U_y_g, U_z_g, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, a, flag);

/*
    test_2(src_g, dest_g, U_x_g, U_y_g, U_z_g, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, 0, flag);
    test_2(src_g, dest_g, U_x_g, U_y_g, U_z_g, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, 1, flag);

    psi_g5<<<dim3(s_x/2, s_y/2, s_z/2), dim3(2, 2, 2) >>>(dest_g, s_x, s_y, s_z, s_t);	
*/

//    hipDeviceSynchronize();

    hipMemcpy(dest, dest_g, size_f * sizeof(double), hipMemcpyDeviceToHost);

    hipFree(src_g);
    hipFree(dest_g);
    hipFree(U_x_g);
    hipFree(U_y_g);
    hipFree(U_z_g);
    hipFree(U_t_g);

//    printf("test PASSED!\n");

    return 0;

}

