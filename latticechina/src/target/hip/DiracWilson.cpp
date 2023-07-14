//
// Created by louis on 2022/2/20.
//

#include <mpi.h>

#include "target/hip/DiracWilson.h"
#include "target/hip/transfer_kernel.h"
#include "target/hip/dslash_main_kernel.h"
#include "target/hip/ghost_kernel.h"
#include "target/hip/operator_kernel.h"

DiracWilson::DiracWilson(double *U_x_in, double *U_y_in, double *U_z_in, double *U_t_in,
                         const int v_x_in, const int v_y_in, const int v_z_in, const int v_t_in,
                         const int s_x_in, const int s_y_in, const int s_z_in, const int s_t_in) :
        DiracSetup(U_x_in, U_y_in, U_z_in, U_t_in, v_x_in, v_y_in, v_z_in, v_t_in, s_x_in, s_y_in, s_z_in, s_t_in) {

}

DiracWilson::~DiracWilson(){

}

void DiracWilson::dslash(double *src_g, double *dest_g, const int cb, const int flag) {

    hipEventRecord(control, 0);

    MPI_Request req[8 * size];
    MPI_Request reqr[8 * size];
    MPI_Status sta[8 * size];

    if (N_sub[0] != 1) {

        hipStreamWaitEvent(stre_x_f, control, 0);

        transfer_x_f<<<s_y * s_z * s_t / 64 / 2, 64, 0, stre_x_f>>>(src_g, tran_x_b, U_x, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        hipStreamWaitEvent(stre_x_b, control, 0);

        transfer_x_b<<<s_y * s_z * s_t / 64 / 2, 64, 0, stre_x_b>>>(src_g, tran_x_f, U_x, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
    }

    if (N_sub[1] != 1) {

        hipStreamWaitEvent(stre_y_f, control, 0);

        transfer_y_f<<<s_x_cb * s_z * s_t / 64, 64, 0, stre_y_f>>>(src_g, tran_y_b, U_y, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        hipStreamWaitEvent(stre_y_b, control, 0);

        transfer_y_b<<<s_x_cb * s_z * s_t / 64, 64, 0, stre_y_b>>>(src_g, tran_y_f, U_y, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
    }

    if (N_sub[2] != 1) {

        hipStreamWaitEvent(stre_z_f, control, 0);

        transfer_z_f<<<s_x_cb * s_y * s_t / 64, 64, 0, stre_z_f>>>(src_g, tran_z_b, U_z, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        hipStreamWaitEvent(stre_z_b, control, 0);

        transfer_z_b<<<s_x_cb * s_y * s_t / 64, 64, 0, stre_z_b>>>(src_g, tran_z_f, U_z, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
    }

    if (N_sub[3] != 1) {

        hipStreamWaitEvent(stre_t_f, control, 0);

        transfer_t_f<<<s_x_cb * s_y * s_z / 64, 64, 0, stre_t_f>>>(src_g, tran_t_b, U_t, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        hipStreamWaitEvent(stre_t_b, control, 0);

        transfer_t_b<<<s_x_cb * s_y * s_z / 64, 64, 0, stre_t_b>>>(src_g, tran_t_f, U_t, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
    }

    main_xyzt<<< s_x_cb * s_y * s_z * s_t / 64, 64>>>(src_g, dest_g, U_x, U_y, U_z, U_t, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

    if (N_sub[0] != 1) {

        hipMemcpyAsync(send_x_b, tran_x_b, len_x * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_x_f);

        hipStreamSynchronize(stre_x_f);

        MPI_Isend(send_x_b, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_b, 8 * rank + 0, MPI_COMM_WORLD, &req[8 * rank + 0]);
        MPI_Irecv(resv_x_f, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_f, 8 * nodenum_x_f + 0, MPI_COMM_WORLD, &reqr[8 * nodenum_x_f + 0]);

        hipMemcpyAsync(send_x_f, tran_x_f, len_x * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_x_b);

        hipStreamSynchronize(stre_x_b);

        MPI_Isend(send_x_f, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_f, 8 * rank + 1, MPI_COMM_WORLD, &req[8 * rank + 1]);
        MPI_Irecv(resv_x_b, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_b, 8 * nodenum_x_b + 1, MPI_COMM_WORLD, &reqr[8 * nodenum_x_b + 1]);

        MPI_Wait(&reqr[8 * nodenum_x_f + 0], &sta[8 * nodenum_x_f + 0]);

        hipMemcpyAsync(ghos_x_f, resv_x_f, len_x * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_x_f);

        hipEventRecord(control, stre_x_f);
        hipStreamWaitEvent(0, control, 0);

        ghost_x_f<<<s_y * s_z * s_t / 64 / 2, 64>>>(ghos_x_f, dest_g, U_x, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        MPI_Wait(&reqr[8 * nodenum_x_b + 1], &sta[8 * nodenum_x_b + 1]);

        hipMemcpyAsync(ghos_x_b, resv_x_b, len_x * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_x_b);

        hipEventRecord(control, stre_x_b);
        hipStreamWaitEvent(0, control, 0);

        ghost_x_b<<<s_y * s_z * s_t / 64 / 2, 64>>>(ghos_x_b, dest_g, U_x, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
    }

    if (N_sub[1] != 1) {

        hipMemcpyAsync(send_y_b, tran_y_b, len_y * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_y_f);

        hipStreamSynchronize(stre_y_f);

        MPI_Isend(send_y_b, len_y * 6 * 2, MPI_DOUBLE, nodenum_y_b, 8 * rank + 2, MPI_COMM_WORLD, &req[8 * rank + 2]);
        MPI_Irecv(resv_y_f, len_y * 6 * 2, MPI_DOUBLE, nodenum_y_f, 8 * nodenum_y_f + 2, MPI_COMM_WORLD, &reqr[8 * nodenum_y_f + 2]);

        hipMemcpyAsync(send_y_f, tran_y_f, len_y * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_y_b);

        hipStreamSynchronize(stre_y_b);

        MPI_Isend(send_y_f, len_y * 6 * 2, MPI_DOUBLE, nodenum_y_f, 8 * rank + 3, MPI_COMM_WORLD, &req[8 * rank + 3]);
        MPI_Irecv(resv_y_b, len_y * 6 * 2, MPI_DOUBLE, nodenum_y_b, 8 * nodenum_y_b + 3, MPI_COMM_WORLD, &reqr[8 * nodenum_y_b + 3]);

        MPI_Wait(&reqr[8 * nodenum_y_f + 2], &sta[8 * nodenum_y_f + 2]);

        hipMemcpyAsync(ghos_y_f, resv_y_f, len_y * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_y_f);

        hipEventRecord(control, stre_y_f);
        hipStreamWaitEvent(0, control, 0);

        ghost_y_f<<<s_x_cb * s_z * s_t / 64, 64>>>(ghos_y_f, dest_g, U_y, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        MPI_Wait(&reqr[8 * nodenum_y_b + 3], &sta[8 * nodenum_y_b + 3]);

        hipMemcpyAsync(ghos_y_b, resv_y_b, len_y * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_y_b);

        hipEventRecord(control, stre_y_b);
        hipStreamWaitEvent(0, control, 0);

        ghost_y_b<<<s_x_cb * s_z * s_t / 64, 64>>>(ghos_y_b, dest_g, U_y, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
    }

    if (N_sub[2] != 1) {

        hipMemcpyAsync(send_z_b, tran_z_b, len_z * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_z_f);

        hipStreamSynchronize(stre_z_f);

        MPI_Isend(send_z_b, len_z * 6 * 2, MPI_DOUBLE, nodenum_z_b, 8 * rank + 4, MPI_COMM_WORLD, &req[8 * rank + 4]);
        MPI_Irecv(resv_z_f, len_z * 6 * 2, MPI_DOUBLE, nodenum_z_f, 8 * nodenum_z_f + 4, MPI_COMM_WORLD, &reqr[8 * nodenum_z_f + 4]);

        hipMemcpyAsync(send_z_f, tran_z_f, len_z * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_z_b);

        hipStreamSynchronize(stre_z_b);

        MPI_Isend(send_z_f, len_z * 6 * 2, MPI_DOUBLE, nodenum_z_f, 8 * rank + 5, MPI_COMM_WORLD, &req[8 * rank + 5]);
        MPI_Irecv(resv_z_b, len_z * 6 * 2, MPI_DOUBLE, nodenum_z_b, 8 * nodenum_z_b + 5, MPI_COMM_WORLD, &reqr[8 * nodenum_z_b + 5]);

        MPI_Wait(&reqr[8 * nodenum_z_f + 4], &sta[8 * nodenum_z_f + 4]);

        hipMemcpyAsync(ghos_z_f, resv_z_f, len_z * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_z_f);

        hipEventRecord(control, stre_z_f);
        hipStreamWaitEvent(0, control, 0);

        ghost_z_f<<<s_x_cb * s_y * s_t / 64, 64>>>(ghos_z_f, dest_g, U_z, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        MPI_Wait(&reqr[8 * nodenum_z_b + 5], &sta[8 * nodenum_z_b + 5]);

        hipMemcpyAsync(ghos_z_b, resv_z_b, len_z * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_z_b);

        hipEventRecord(control, stre_z_b);
        hipStreamWaitEvent(0, control, 0);

        ghost_z_b<<<s_x_cb * s_y * s_t / 64, 64>>>(ghos_z_b, dest_g, U_z, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
    }

    if (N_sub[3] != 1) {

        hipMemcpyAsync(send_t_b, tran_t_b, len_t * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_t_f);

        hipStreamSynchronize(stre_t_f);

        MPI_Isend(send_t_b, len_t * 6 * 2, MPI_DOUBLE, nodenum_t_b, 8 * rank + 6, MPI_COMM_WORLD, &req[8 * rank + 6]);
        MPI_Irecv(resv_t_f, len_t * 6 * 2, MPI_DOUBLE, nodenum_t_f, 8 * nodenum_t_f + 6, MPI_COMM_WORLD, &reqr[8 * nodenum_t_f + 6]);

        hipMemcpyAsync(send_t_f, tran_t_f, len_t * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_t_b);

        hipStreamSynchronize(stre_t_b);

        MPI_Isend(send_t_f, len_t * 6 * 2, MPI_DOUBLE, nodenum_t_f, 8 * rank + 7, MPI_COMM_WORLD, &req[8 * rank + 7]);
        MPI_Irecv(resv_t_b, len_t * 6 * 2, MPI_DOUBLE, nodenum_t_b, 8 * nodenum_t_b + 7, MPI_COMM_WORLD, &reqr[8 * nodenum_t_b + 7]);

        MPI_Wait(&reqr[8 * nodenum_t_f + 6], &sta[8 * nodenum_t_f + 6]);

        hipMemcpyAsync(ghos_t_f, resv_t_f, len_t * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_t_f);

        hipEventRecord(control, stre_t_f);
        hipStreamWaitEvent(0, control, 0);

        ghost_t_f<<<s_x_cb * s_y * s_z / 64, 64>>>(ghos_t_f, dest_g, U_t, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        MPI_Wait(&reqr[8 * nodenum_t_b + 7], &sta[8 * nodenum_t_b + 7]);

        hipMemcpyAsync(ghos_t_b, resv_t_b, len_t * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_t_b);

        hipEventRecord(control, stre_t_b);
        hipStreamWaitEvent(0, control, 0);

        ghost_t_b<<<s_x_cb * s_y * s_z / 64, 64>>>(ghos_t_b, dest_g, U_t, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
    }

    MPI_Barrier(MPI_COMM_WORLD);

}

void DiracWilson::Kernal(double *out, double *in) {
/*
        float eventMs = 1.0f;
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, 0);
*/
    dslash(in, out, 0, 1);
    dslash(in, out, 1, 1);
/*
	hipEventRecord(stop, 0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&eventMs, start, stop);
	printf("dslash time taken  = %6.3fms\n", eventMs);
*/
}