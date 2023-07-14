//
// Created by louis on 2022/2/20.
//

#ifndef LATTICE_DIRACSETUP_H
#define LATTICE_DIRACSETUP_H

#include "hip/hip_runtime.h"

class DiracSetup {

public:

    double *U_x;
    double *U_y;
    double *U_z;
    double *U_t;

    int v_x, v_y, v_z, v_t;
    int s_x, s_y, s_z, s_t;

    int N_sub[4];
    int rank;
    int size;

    int nodenum_x_b;
    int nodenum_x_f;
    int nodenum_y_b;
    int nodenum_y_f;
    int nodenum_z_b;
    int nodenum_z_f;
    int nodenum_t_b;
    int nodenum_t_f;

    int s_x_cb;

    int len_x;
    int len_y;
    int len_z;
    int len_t;

    double *resv_x_f;
    double *send_x_b;
    double *resv_x_b;
    double *send_x_f;

    double *resv_y_f;
    double *send_y_b;
    double *resv_y_b;
    double *send_y_f;

    double *resv_z_f;
    double *send_z_b;
    double *resv_z_b;
    double *send_z_f;

    double *resv_t_f;
    double *send_t_b;
    double *resv_t_b;
    double *send_t_f;

    double *tran_x_f;
    double *tran_x_b;
    double *tran_y_f;
    double *tran_y_b;
    double *tran_z_f;
    double *tran_z_b;
    double *tran_t_f;
    double *tran_t_b;

    double *ghos_x_f;
    double *ghos_x_b;
    double *ghos_y_f;
    double *ghos_y_b;
    double *ghos_z_f;
    double *ghos_z_b;
    double *ghos_t_f;
    double *ghos_t_b;

    hipStream_t stre_x_f;
    hipStream_t stre_x_b;
    hipStream_t stre_y_f;
    hipStream_t stre_y_b;
    hipStream_t stre_z_f;
    hipStream_t stre_z_b;
    hipStream_t stre_t_f;
    hipStream_t stre_t_b;

    hipEvent_t control;

    int volume;

    DiracSetup(double *U_x_in, double *U_y_in, double *U_z_in, double *U_t_in,
               const int v_x_in, const int v_y_in, const int v_z_in, const int v_t_in,
               const int s_x_in, const int s_y_in, const int s_z_in, const int s_t_in);
    ~DiracSetup();
};

#endif //LATTICE_DIRACSETUP_H
