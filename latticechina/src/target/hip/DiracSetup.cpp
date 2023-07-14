//
// Created by louis on 2022/2/20.
//

#include <mpi.h>

#include "operator.h"
#include "target/hip/DiracSetup.h"

DiracSetup::DiracSetup(double *U_x_in, double *U_y_in, double *U_z_in, double *U_t_in,
                       const int v_x_in, const int v_y_in, const int v_z_in, const int v_t_in,
                       const int s_x_in, const int s_y_in, const int s_z_in, const int s_t_in) {

    volume = s_x_in * s_y_in * s_z_in * s_t_in;

    int size_u = s_x_in * s_y_in * s_z_in * s_t_in * 9 * 2;

    hipMalloc((void **) &U_x, size_u * sizeof(double));
    hipMalloc((void **) &U_y, size_u * sizeof(double));
    hipMalloc((void **) &U_z, size_u * sizeof(double));
    hipMalloc((void **) &U_t, size_u * sizeof(double));

    hipMemcpy(U_x, U_x_in, size_u * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(U_y, U_y_in, size_u * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(U_z, U_z_in, size_u * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(U_t, U_t_in, size_u * sizeof(double), hipMemcpyHostToDevice);

    v_x = v_x_in;
    v_y = v_y_in;
    v_z = v_z_in;
    v_t = v_t_in;
    s_x = s_x_in;
    s_y = s_y_in;
    s_z = s_z_in;
    s_t = s_t_in;

    N_sub[0] = v_x / s_x;
    N_sub[1] = v_y / s_y;
    N_sub[2] = v_z / s_z;
    N_sub[3] = v_t / s_t;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int site_x_f[4] = {(rank + 1) % N_sub[0],
                       (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    int site_x_b[4] = {(rank - 1 + N_sub[0]) % N_sub[0],
                       (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    nodenum_x_b = get_nodenum(site_x_b, N_sub, 4);
    nodenum_x_f = get_nodenum(site_x_f, N_sub, 4);

    int site_y_f[4] = {(rank % N_sub[0]),
                       (rank / N_sub[0] + 1) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    int site_y_b[4] = {(rank % N_sub[0]),
                       (rank / N_sub[0] - 1 + N_sub[1]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    nodenum_y_b = get_nodenum(site_y_b, N_sub, 4);
    nodenum_y_f = get_nodenum(site_y_f, N_sub, 4);

    int site_z_f[4] = {(rank % N_sub[0]),
                       (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0]) + 1) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    int site_z_b[4] = {(rank % N_sub[0]),
                       (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0]) - 1 + N_sub[2]) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    nodenum_z_b = get_nodenum(site_z_b, N_sub, 4);
    nodenum_z_f = get_nodenum(site_z_f, N_sub, 4);

    int site_t_f[4] = {(rank % N_sub[0]),
                       (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       (rank / (N_sub[2] * N_sub[1] * N_sub[0]) + 1) % N_sub[3]};

    int site_t_b[4] = {(rank % N_sub[0]),
                       (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       (rank / (N_sub[2] * N_sub[1] * N_sub[0]) - 1 + N_sub[3]) % N_sub[3]};

    nodenum_t_b = get_nodenum(site_t_b, N_sub, 4);
    nodenum_t_f = get_nodenum(site_t_f, N_sub, 4);

    s_x_cb = s_x >> 1;

    len_x = s_y * s_z * s_t >> 1;
    len_y = s_x_cb * s_z * s_t;
    len_z = s_x_cb * s_y * s_t;
    len_t = s_x_cb * s_y * s_z;

    if (N_sub[0] != 1) {
/*
            resv_x_f = new double[len_x * 6 * 2];
            send_x_b = new double[len_x * 6 * 2];
            resv_x_b = new double[len_x * 6 * 2];
            send_x_f = new double[len_x * 6 * 2];
*/
        int size_T = len_x * 6 * 2;

        hipMallocManaged((void **) &resv_x_f, size_T * sizeof(double));
        hipMallocManaged((void **) &send_x_b, size_T * sizeof(double));
        hipMallocManaged((void **) &resv_x_b, size_T * sizeof(double));
        hipMallocManaged((void **) &send_x_f, size_T * sizeof(double));


        hipMalloc((void **) &tran_x_f, size_T * sizeof(double));
        hipMalloc((void **) &tran_x_b, size_T * sizeof(double));

        hipMalloc((void **) &ghos_x_f, size_T * sizeof(double));
        hipMalloc((void **) &ghos_x_b, size_T * sizeof(double));

    }

    if (N_sub[1] != 1) {
/*
            resv_y_f = new double[len_y * 6 * 2];
            send_y_b = new double[len_y * 6 * 2];
            resv_y_b = new double[len_y * 6 * 2];
            send_y_f = new double[len_y * 6 * 2];
*/
        int size_T = len_y * 6 * 2;

        hipMallocManaged((void **) &resv_y_f, size_T * sizeof(double));
        hipMallocManaged((void **) &send_y_b, size_T * sizeof(double));
        hipMallocManaged((void **) &resv_y_b, size_T * sizeof(double));
        hipMallocManaged((void **) &send_y_f, size_T * sizeof(double));

        hipMalloc((void **) &tran_y_f, size_T * sizeof(double));
        hipMalloc((void **) &tran_y_b, size_T * sizeof(double));

        hipMalloc((void **) &ghos_y_f, size_T * sizeof(double));
        hipMalloc((void **) &ghos_y_b, size_T * sizeof(double));

    }

    if (N_sub[2] != 1) {
/*
            resv_z_f = new double[len_z * 6 * 2];
            send_z_b = new double[len_z * 6 * 2];
            resv_z_b = new double[len_z * 6 * 2];
            send_z_f = new double[len_z * 6 * 2];
*/
        int size_T = len_z * 6 * 2;

        hipMallocManaged((void **) &resv_z_f, size_T * sizeof(double));
        hipMallocManaged((void **) &send_z_b, size_T * sizeof(double));
        hipMallocManaged((void **) &resv_z_b, size_T * sizeof(double));
        hipMallocManaged((void **) &send_z_f, size_T * sizeof(double));

        hipMalloc((void **) &tran_z_f, size_T * sizeof(double));
        hipMalloc((void **) &tran_z_b, size_T * sizeof(double));

        hipMalloc((void **) &ghos_z_f, size_T * sizeof(double));
        hipMalloc((void **) &ghos_z_b, size_T * sizeof(double));

    }

    if (N_sub[3] != 1) {
/*
            resv_t_f = new double[len_t * 6 * 2];
            send_t_b = new double[len_t * 6 * 2];
            resv_t_b = new double[len_t * 6 * 2];
            send_t_f = new double[len_t * 6 * 2];
*/
        int size_T = len_t * 6 * 2;

        hipMallocManaged((void **) &resv_t_f, size_T * sizeof(double));
        hipMallocManaged((void **) &send_t_b, size_T * sizeof(double));
        hipMallocManaged((void **) &resv_t_b, size_T * sizeof(double));
        hipMallocManaged((void **) &send_t_f, size_T * sizeof(double));

/*
            hipHostMalloc((void **) &resv_t_f, size_T * sizeof(double));
            hipHostMalloc((void **) &send_t_b, size_T * sizeof(double));
            hipHostMalloc((void **) &resv_t_f, size_T * sizeof(double));
            hipHostMalloc((void **) &send_t_b, size_T * sizeof(double));
*/

        hipMalloc((void **) &tran_t_f, size_T * sizeof(double));
        hipMalloc((void **) &tran_t_b, size_T * sizeof(double));

        hipMalloc((void **) &ghos_t_f, size_T * sizeof(double));
        hipMalloc((void **) &ghos_t_b, size_T * sizeof(double));

    }

    hipStreamCreate(&stre_x_f);
    hipStreamCreate(&stre_x_b);
    hipStreamCreate(&stre_y_f);
    hipStreamCreate(&stre_y_b);
    hipStreamCreate(&stre_t_f);
    hipStreamCreate(&stre_t_b);
    hipStreamCreate(&stre_z_f);
    hipStreamCreate(&stre_z_b);
    hipEventCreate(&control);

}

DiracSetup::~DiracSetup() {

    hipFree(U_x);
    hipFree(U_y);
    hipFree(U_z);
    hipFree(U_t);

    if (N_sub[0] != 1) {
/*
        		delete[] resv_x_f;
                	delete[] send_x_b;
                	delete[] resv_x_b;
                	delete[] send_x_f;
*/
        hipHostFree(resv_x_f);
        hipHostFree(send_x_b);
        hipHostFree(resv_x_b);
        hipHostFree(send_x_f);

        hipFree(tran_x_f);
        hipFree(tran_x_b);

        hipFree(ghos_x_f);
        hipFree(ghos_x_b);
    }

    if (N_sub[1] != 1) {
/*
        		delete[] resv_y_f;
                	delete[] send_y_b;
                	delete[] resv_y_b;
                	delete[] send_y_f;
*/
        hipHostFree(resv_y_f);
        hipHostFree(send_y_b);
        hipHostFree(resv_y_b);
        hipHostFree(send_y_f);

        hipFree(tran_y_f);
        hipFree(tran_y_b);

        hipFree(ghos_y_f);
        hipFree(ghos_y_b);
    }

    if (N_sub[2] != 1) {
/*
			delete[] resv_z_f;
			delete[] send_z_b;
			delete[] resv_z_b;
			delete[] send_z_f;
*/
        hipHostFree(resv_z_f);
        hipHostFree(send_z_b);
        hipHostFree(resv_z_b);
        hipHostFree(send_z_f);

        hipFree(tran_z_f);
        hipFree(tran_z_b);

        hipFree(ghos_z_f);
        hipFree(ghos_z_b);
    }

    if (N_sub[3] != 1) {
/*
			delete[] resv_t_f;
			delete[] send_t_b;
			delete[] resv_t_b;
			delete[] send_t_f;
*/

        hipHostFree(resv_t_f);
        hipHostFree(send_t_b);
        hipHostFree(resv_t_b);
        hipHostFree(send_t_f);

        hipFree(tran_t_f);
        hipFree(tran_t_b);

        hipFree(ghos_t_f);
        hipFree(ghos_t_b);
    }

    hipStreamDestroy(stre_x_f);
    hipStreamDestroy(stre_x_b);
    hipStreamDestroy(stre_y_f);
    hipStreamDestroy(stre_y_b);
    hipStreamDestroy(stre_t_f);
    hipStreamDestroy(stre_t_b);
    hipStreamDestroy(stre_z_f);
    hipStreamDestroy(stre_z_b);
    hipEventDestroy(control);
}