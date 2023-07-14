//
// Created by louis on 2022/2/20.
//

#include <mpi.h>

#include "target/hip/DiracOverlapWilson.h"
#include "target/hip/transfer_kernel.h"
#include "target/hip/dslash_main_kernel.h"
#include "target/hip/dslash_main_overlap_kernel.h"
#include "target/hip/ghost_kernel.h"
#include "target/hip/ghost_overlap_kernel.h"
#include "target/hip/axpb_kernel.h"
#include "target/hip/dot_product_kernel.h"
#include "target/hip/axpb_complex_kernel.h"
#include "target/hip/operator_kernel.h"

__global__ void overlapLinop(double *out, double *in, const double k0, const double k1, const double k2) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int j = i % 24;

    if (j < 12) {
        out[i] = k0 * in[i] + (k2 + k1) * out[i];
    } else {
        out[i] = k0 * in[i] + (k2 - k1) * out[i];
    }

}

DiracOverlapWilson::DiracOverlapWilson(std::vector<double *> &evecs,
                                       std::vector<double> &evals,
                                       std::vector <std::vector<double>> &coefs,
                                       std::vector<int> &sizes,
                                       double *U_x_in, double *U_y_in, double *U_z_in, double *U_t_in,
                                       const int v_x_in, const int v_y_in, const int v_z_in, const int v_t_in,
                                       const int s_x_in, const int s_y_in, const int s_z_in, const int s_t_in,
                                       const double kappa_in) :
        prec0(1e-12), hw_evec(evecs), hw_eval(evals), coef(coefs), hw_size(sizes),
        DiracSetup(U_x_in, U_y_in, U_z_in, U_t_in,
                   v_x_in, v_y_in, v_z_in, v_t_in,
                   s_x_in, s_y_in, s_z_in, s_t_in) {

    build_hw = false;
    kappa = kappa_in;
    rho = 4 - 0.5 / kappa;

    int size_f = s_x * s_y * s_z * s_t * 12 * 2;
    hipMalloc((void **) &tmp1, size_f * sizeof(double));
    hipMalloc((void **) &tmp2, size_f * sizeof(double));

    hw_eval = evals;

    hipMalloc((void **) &hw_evec_g, hw_evec.size() * size_f * sizeof(double));

    for (int i = 0; i < hw_evec.size(); i++) {

        hipMemcpy(hw_evec_g + i * size_f, hw_evec[i], size_f * sizeof(double), hipMemcpyHostToDevice);

    }

    double *hw_eval_g;
    hipMalloc((void **) &hw_eval_g, hw_evec.size() * sizeof(double));
    hipMemcpy(hw_eval_g, &hw_eval[0], hw_evec.size() * sizeof(double), hipMemcpyHostToDevice);

    hipMalloc((void **) &sig_hw_eval_g, hw_evec.size() * sizeof(int));
    sign<<<1, hw_evec.size() >>> (sig_hw_eval_g, hw_eval_g);
    hipFree(hw_eval_g);
}

DiracOverlapWilson::~DiracOverlapWilson() {
    hipFree(tmp1);
    hipFree(tmp2);
    hipFree(hw_evec_g);
    hipFree(sig_hw_eval_g);
}

void DiracOverlapWilson::dslash(double *src_g, double *dest_g, const int cb, const int flag, const double a, const double b) {

    hipEventRecord(control, 0);

    MPI_Request req[8 * size];
    MPI_Request reqr[8 * size];
    MPI_Status sta[8 * size];

    if (N_sub[0] != 1) {

        hipStreamWaitEvent(stre_x_f,control,0);

        transfer_x_f<<<s_y * s_z * s_t / 64 / 2, 64, 0, stre_x_f>>>(src_g, tran_x_b, U_x, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        hipStreamWaitEvent(stre_x_b,control,0);

        transfer_x_b<<<s_y * s_z * s_t / 64 / 2, 64, 0, stre_x_b>>>(src_g, tran_x_f, U_x, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
    }

    if (N_sub[1] != 1) {

        hipStreamWaitEvent(stre_y_f,control,0);

        transfer_y_f<<<s_x_cb * s_z * s_t / 64, 64, 0, stre_y_f>>>(src_g, tran_y_b, U_y, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        hipStreamWaitEvent(stre_y_b,control,0);

        transfer_y_b<<<s_x_cb * s_z * s_t / 64, 64, 0, stre_y_b>>>(src_g, tran_y_f, U_y, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
    }

    if (N_sub[2] != 1) {

        hipStreamWaitEvent(stre_z_f,control,0);

        transfer_z_f<<<s_x_cb * s_y * s_t / 64, 64, 0, stre_z_f>>>(src_g, tran_z_b, U_z, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        hipStreamWaitEvent(stre_z_b,control,0);

        transfer_z_b<<<s_x_cb * s_y * s_t / 64, 64, 0, stre_z_b>>>(src_g, tran_z_f, U_z, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
    }

    if (N_sub[3] != 1) {

        hipStreamWaitEvent(stre_t_f,control,0);

        transfer_t_f<<<s_x_cb * s_y * s_z / 64, 64, 0, stre_t_f>>>(src_g, tran_t_b, U_t, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);

        hipStreamWaitEvent(stre_t_b,control,0);

        transfer_t_b<<<s_x_cb * s_y * s_z / 64, 64, 0, stre_t_b>>>(src_g, tran_t_f, U_t, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag);
    }

    main_xyzt_abp5<<< s_x_cb * s_y * s_z * s_t / 64, 64>>>(src_g, dest_g, U_x, U_y, U_z, U_t, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag, a, b);

    if (N_sub[0] != 1) {

        hipMemcpyAsync(send_x_b, tran_x_b, len_x * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_x_f);

        hipStreamSynchronize(stre_x_f);

        MPI_Isend(send_x_b, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_b, 8 * rank + 0, MPI_COMM_WORLD,
                  &req[8 * rank + 0]);
        MPI_Irecv(resv_x_f, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_f, 8 * nodenum_x_f + 0, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_x_f + 0]);

        hipMemcpyAsync(send_x_f, tran_x_f, len_x * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_x_b);

        hipStreamSynchronize(stre_x_b);

        MPI_Isend(send_x_f, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_f, 8 * rank + 1, MPI_COMM_WORLD,
                  &req[8 * rank + 1]);
        MPI_Irecv(resv_x_b, len_x * 6 * 2, MPI_DOUBLE, nodenum_x_b, 8 * nodenum_x_b + 1, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_x_b + 1]);

        MPI_Wait(&reqr[8 * nodenum_x_f + 0], &sta[8 * nodenum_x_f + 0]);

        hipMemcpyAsync(ghos_x_f, resv_x_f, len_x * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_x_f);

        hipEventRecord(control, stre_x_f);
        hipStreamWaitEvent(0, control, 0);

        ghost_x_f_abp5<<<s_y * s_z * s_t / 64 / 2, 64>>>(ghos_x_f, dest_g, U_x, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag, b);

        MPI_Wait(&reqr[8 * nodenum_x_b + 1], &sta[8 * nodenum_x_b + 1]);

        hipMemcpyAsync(ghos_x_b, resv_x_b, len_x * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_x_b);

        hipEventRecord(control, stre_x_b);
        hipStreamWaitEvent(0, control, 0);

        ghost_x_b_abp5<<<s_y * s_z * s_t / 64 / 2, 64>>>(ghos_x_b, dest_g, U_x, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag, b);

    }

    if (N_sub[1] != 1) {

        hipMemcpyAsync(send_y_b, tran_y_b, len_y * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_y_f);

        hipStreamSynchronize(stre_y_f);

        MPI_Isend(send_y_b, len_y * 6 * 2, MPI_DOUBLE, nodenum_y_b, 8 * rank + 2, MPI_COMM_WORLD,
                  &req[8 * rank + 2]);
        MPI_Irecv(resv_y_f, len_y * 6 * 2, MPI_DOUBLE, nodenum_y_f, 8 * nodenum_y_f + 2, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_y_f + 2]);

        hipMemcpyAsync(send_y_f, tran_y_f, len_y * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_y_b);

        hipStreamSynchronize(stre_y_b);

        MPI_Isend(send_y_f, len_y * 6 * 2, MPI_DOUBLE, nodenum_y_f, 8 * rank + 3, MPI_COMM_WORLD,
                  &req[8 * rank + 3]);
        MPI_Irecv(resv_y_b, len_y * 6 * 2, MPI_DOUBLE, nodenum_y_b, 8 * nodenum_y_b + 3, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_y_b + 3]);

        MPI_Wait(&reqr[8 * nodenum_y_f + 2], &sta[8 * nodenum_y_f + 2]);

        hipMemcpyAsync(ghos_y_f, resv_y_f, len_y * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_y_f);

        hipEventRecord(control, stre_y_f);
        hipStreamWaitEvent(0, control, 0);

        ghost_y_f_abp5<<<s_x_cb * s_z * s_t / 64, 64>>>(ghos_y_f, dest_g, U_y, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag, b);

        MPI_Wait(&reqr[8 * nodenum_y_b + 3], &sta[8 * nodenum_y_b + 3]);

        hipMemcpyAsync(ghos_y_b, resv_y_b, len_y * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_y_b);

        hipEventRecord(control, stre_y_b);
        hipStreamWaitEvent(0, control, 0);

        ghost_y_b_abp5<<<s_x_cb * s_z * s_t / 64, 64>>>(ghos_y_b, dest_g, U_y, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag, b);

    }

    if (N_sub[2] != 1) {

        hipMemcpyAsync(send_z_b, tran_z_b, len_z * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_z_f);

        hipStreamSynchronize(stre_z_f);

        MPI_Isend(send_z_b, len_z * 6 * 2, MPI_DOUBLE, nodenum_z_b, 8 * rank + 4, MPI_COMM_WORLD,
                  &req[8 * rank + 4]);
        MPI_Irecv(resv_z_f, len_z * 6 * 2, MPI_DOUBLE, nodenum_z_f, 8 * nodenum_z_f + 4, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_z_f + 4]);

        hipMemcpyAsync(send_z_f, tran_z_f, len_z * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_z_b);

        hipStreamSynchronize(stre_z_b);

        MPI_Isend(send_z_f, len_z * 6 * 2, MPI_DOUBLE, nodenum_z_f, 8 * rank + 5, MPI_COMM_WORLD,
                  &req[8 * rank + 5]);
        MPI_Irecv(resv_z_b, len_z * 6 * 2, MPI_DOUBLE, nodenum_z_b, 8 * nodenum_z_b + 5, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_z_b + 5]);

        MPI_Wait(&reqr[8 * nodenum_z_f + 4], &sta[8 * nodenum_z_f + 4]);

        hipMemcpyAsync(ghos_z_f, resv_z_f, len_z * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_z_f);

        hipEventRecord(control, stre_z_f);
        hipStreamWaitEvent(0, control, 0);

        ghost_z_f_abp5<<<s_x_cb * s_y * s_t / 64, 64>>>(ghos_z_f, dest_g, U_z, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag, b);

        MPI_Wait(&reqr[8 * nodenum_z_b + 5], &sta[8 * nodenum_z_b + 5]);

        hipMemcpyAsync(ghos_z_b, resv_z_b, len_z * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_z_b);

        hipEventRecord(control, stre_z_b);
        hipStreamWaitEvent(0, control, 0);

        ghost_z_b_abp5<<<s_x_cb * s_y * s_t / 64, 64>>>(ghos_z_b, dest_g, U_z, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag, b);

    }

    if (N_sub[3] != 1) {

        hipMemcpyAsync(send_t_b, tran_t_b, len_t * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_t_f);

        hipStreamSynchronize(stre_t_f);

        MPI_Isend(send_t_b, len_t * 6 * 2, MPI_DOUBLE, nodenum_t_b, 8 * rank + 6, MPI_COMM_WORLD,
                  &req[8 * rank + 6]);
        MPI_Irecv(resv_t_f, len_t * 6 * 2, MPI_DOUBLE, nodenum_t_f, 8 * nodenum_t_f + 6, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_t_f + 6]);

        hipMemcpyAsync(send_t_f, tran_t_f, len_t * 6 * 2 * sizeof(double), hipMemcpyDeviceToHost, stre_t_b);

        hipStreamSynchronize(stre_t_b);

        MPI_Isend(send_t_f, len_t * 6 * 2, MPI_DOUBLE, nodenum_t_f, 8 * rank + 7, MPI_COMM_WORLD,
                  &req[8 * rank + 7]);
        MPI_Irecv(resv_t_b, len_t * 6 * 2, MPI_DOUBLE, nodenum_t_b, 8 * nodenum_t_b + 7, MPI_COMM_WORLD,
                  &reqr[8 * nodenum_t_b + 7]);

        MPI_Wait(&reqr[8 * nodenum_t_f + 6], &sta[8 * nodenum_t_f + 6]);

        hipMemcpyAsync(ghos_t_f, resv_t_f, len_t * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_t_f);

        hipEventRecord(control, stre_t_f);
        hipStreamWaitEvent(0, control, 0);

        ghost_t_f_abp5<<<s_x_cb * s_y * s_z / 64, 64>>>(ghos_t_f, dest_g, U_t, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag, b);

        MPI_Wait(&reqr[8 * nodenum_t_b + 7], &sta[8 * nodenum_t_b + 7]);

        hipMemcpyAsync(ghos_t_b, resv_t_b, len_t * 6 * 2 * sizeof(double), hipMemcpyHostToDevice, stre_t_b);

        hipEventRecord(control, stre_t_b);
        hipStreamWaitEvent(0, control, 0);

        ghost_t_b_abp5<<<s_x_cb * s_y * s_z / 64, 64>>>(ghos_t_b, dest_g, U_t, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, rank, cb, flag, b);

    }

    MPI_Barrier(MPI_COMM_WORLD);

}

void DiracOverlapWilson::Kernel(double *out, double *in) {
/*
        float eventMs = 1.0f;
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, 0);
*/
    const double a = -0.5 / (4 - rho);

    dslash(in, out, 0, 1, 1.0, -2 * a);
    dslash(in, out, 1, 1, 1.0, -2 * a);
/*
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&eventMs, start, stop);
        printf("dslash time taken  = %6.3fms\n", eventMs);
*/
}

void DiracOverlapWilson::KernelSq_scaled(double *out, double *in, double cut) {

    const double sc1 = 2 / ((1 + 8 * kappa) * (1 + 8 * kappa) * (1 - cut));
    const double sc2 = (1 + cut) / (1 - cut);

    Kernel(tmp1, in);

    Kernel(tmp2, tmp1);

    axpbyz_g2<<<s_x * s_y * s_z * s_t * 24 / 1024, 1024 >>> (sc1, tmp2, -sc2, in, out);

}

void DiracOverlapWilson::eps_l_g(double *out, double *in, int size) {

    const int size_f = s_x * s_y * s_z * s_t * 24;
    double *inner_g;
    hipMalloc((void **) &inner_g,   size * 2 * sizeof(double));

    double * &result_g = inner_g;

    if(size_f == 24 * 24 * 24 * 16 * 24){

        double * result_g1;

        hipMalloc((void **) &result_g1,  size * 2592 * 2 * sizeof(double));

        cDotProduct_v1_g2<<< size_f / 2 / 1024, 1024 >>> (result_g1, hw_evec_g, in, size_f / 2 , size, 2592);

        double * result_g2;

        hipMalloc((void **) &result_g2,  4* size * 2 * sizeof(double));

        cDotProduct_v2_g2<<< size * size_f / 2 / 1024 / 648, 648 >>> (result_g2, result_g1 , 2592  , size, 4, 648);

        hipFree(result_g1);

        cDotProduct_v2_g2<<< size * size_f / 2 / 1024 / 648 / 4 / 100, 400 >>> (result_g, result_g2 , 4  , size, 1, 4);

        hipFree(result_g2);

    }else if(size_f == 8 * 8 * 8 * 8 * 24){

        double * result_g1;

        hipMalloc((void **) &result_g1,  size * 48 * 2 * sizeof(double));

        cDotProduct_v1_g2<<< size_f / 2 / 1024, 1024 >>> (result_g1, hw_evec_g, in, size_f / 2 , size, 48);

        cDotProduct_v2_g2<<< size * size_f / 2 / 1024 / 48 / 10, 480 >>> (result_g, result_g1 , 48  , size, 1, 48);

        hipFree(result_g1);

    }

    double result[2 * size];
    hipMemcpy(result, result_g, size * 2 * sizeof(double), hipMemcpyDeviceToHost);

    double inner[2 * size];

    MPI_Reduce(&result[0], &inner[0], 2 * size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Bcast(&inner[0], 2 * size , MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    hipMemcpy(inner_g, inner, 2 * size * sizeof(double), hipMemcpyHostToDevice);

    cxmbyx_v1_g2<<<size_f / 2  / 768, 768 >>>(in, inner_g, hw_evec_g, size_f / 2, size);

    mult<<<1, 2*size>>>(inner_g, sig_hw_eval_g);

    cxpbyx_v1_g2<<<size_f / 2  / 768, 768 >>>(out, inner_g, hw_evec_g, size_f / 2, size);

    hipFree(inner_g);
}

void DiracOverlapWilson::general_dov(double *out, double *in, double k0, double k1, double k2, double prec) {

    float eventMs = 1.0f;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);


    int is = -(int) log(prec / 0.2);
    if (is < 0)is = 0;
    if (is > coef.size() - 1)is = coef.size() - 1;

    int size_f = s_x * s_y * s_z * s_t * 24;

    double *src, *high;
    hipMalloc((void **) &src, size_f * sizeof(double));
    hipMemcpy(src, in, size_f * sizeof(double), hipMemcpyDeviceToDevice);

    hipMalloc((void **) &high, size_f * sizeof(double));
    hipMemset(high, 0, size_f * sizeof(double));

    hipMemset(out, 0, size_f * sizeof(double));

    eps_l_g(out, src, hw_size[is]);

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    if(rank==0) printf("Kernel 1 time taken  = %6.3fms\n", eventMs);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);


    double cut = pow(hw_eval[hw_size[is] - 1] / (1 + 8 * kappa), 2);

    double *pbn2, *pbn1, *pbn0, *ptmp;
    hipMalloc((void **) &pbn0, size_f * sizeof(double));
    hipMemset(pbn0, 0, size_f * sizeof(double));
    hipMalloc((void **) &pbn1, size_f * sizeof(double));
    hipMemset(pbn1, 0, size_f * sizeof(double));
    hipMalloc((void **) &pbn2, size_f * sizeof(double));
    hipMemset(pbn2, 0, size_f * sizeof(double));

    for (size_t i = coef[is].size() - 1; i >= 1; i--) {

        if (i < coef[is].size() - 1)KernelSq_scaled(high, pbn1, cut);

        axpbyczw_g2<<<size_f / 1024 , 1024 >>>(2.0, high, -1.0, pbn0, coef[is][i], src, pbn2);

        ptmp = pbn0;
        pbn0 = pbn1;
        pbn1 = pbn2;
        pbn2 = ptmp;
    }

    KernelSq_scaled(high, pbn1, cut);

    axpbyczw_g2<<<size_f / 1024 , 1024 >>>(1.0, high, -1.0, pbn0, coef[is][0], src, pbn2);

    Kernel(high, pbn2);

    axpbyz_g2<<< size_f / 1024, 1024  >>>(1.0 / (1 + 8 * kappa), high, 1.0, out, out);

    hipFree(pbn0);
    hipFree(pbn1);
    hipFree(pbn2);

    overlapLinop<<< size_f / 1024, 1024 >>> (out, in, k0, k1, k2);

    hipFree(src);
    hipFree(high);

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    if(rank==0) {
        printf("Kernel 2 time taken  = %6.3fms\n", eventMs);
    }
}

