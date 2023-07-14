/**#
**# @file:   dslashMain.h
**# @brief:
**# @author: louis shaw
**# @data:   2021/08/17
#**/

#include <mpi.h>
#include "dslash.h"
#include "operator.h"
#include "target/hip/legacyFunction.h"


void DslashEE_gpu(lattice_fermion &src, lattice_fermion &dest, const double mass) {

    dest.clean();
    const double a = 4.0;
    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    for (int i = 0; i < subgrid_vol_cb * 3 * 4; i++) {
        dest.A[i] = (a + mass) * src.A[i];
//        if (dest.A[i].real() != 0) {
//            printf("dest=%f\n", dest.A[i].real());
//        }
    }

}

void DslashOO_gpu(lattice_fermion &src, lattice_fermion &dest, const double mass) {

    dest.clean();
    const double a = 4.0;
    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    for (int i = subgrid_vol_cb * 3 * 4; i < subgrid_vol * 3 * 4; i++) {
        dest.A[i] = (a + mass) * src.A[i];
    }
}

//cb = 0  EO  ;  cb = 1 OE
void Dslashoffd_gpu_apply(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const bool dag, const int cb) {

    double flag = (dag == true) ? -1 : 1;

//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int v_x = src.site_vec[0];
    int v_y = src.site_vec[1];
    int v_z = src.site_vec[2];
    int v_t = src.site_vec[3];

    int s_x = src.subgs[0];
    int s_y = src.subgs[1];
    int s_z = src.subgs[2];
    int s_t = src.subgs[3];

    double *src_g = (double *) src.A;
    double *U_x_g = (double *) U.A[0];
    double *U_y_g = (double *) U.A[1];
    double *U_z_g = (double *) U.A[2];
    double *U_t_g = (double *) U.A[3];
    double *dest_g = (double *) dest.A;

    test(src_g, dest_g, U_x_g, U_y_g, U_z_g, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, cb, flag);
}

void Dslashoffd_gpu_apply(lattice_fermion &src, lattice_fermion &dest,
                          double *U_x_g, double *U_y_g, double *U_z_g, double *U_t_g,
                          const bool dag, const int cb) {

    double flag = (dag == true) ? -1 : 1;

    int v_x = src.site_vec[0];
    int v_y = src.site_vec[1];
    int v_z = src.site_vec[2];
    int v_t = src.site_vec[3];

    int s_x = src.subgs[0];
    int s_y = src.subgs[1];
    int s_z = src.subgs[2];
    int s_t = src.subgs[3];

    double *src_g = (double *) src.A;
    double *dest_g = (double *) dest.A;

    test(src_g, dest_g, U_x_g, U_y_g, U_z_g, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, cb, flag);
}

void Dslashoffd_gpu_apply(double *src_g, double *dest_g, double *U_x_g, double *U_y_g, double *U_z_g, double *U_t_g,
                          const int v_x, const int v_y, const int v_z, const int v_t,
                          const int s_x, const int s_y, const int s_z, const int s_t,
                          const bool dag, const int cb) {

    double flag = (dag == true) ? -1 : 1;

    test(src_g, dest_g, U_x_g, U_y_g, U_z_g, U_t_g, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, cb, flag);

}

void dslash_gpu_g5(double *src, double *dest, double *U_x, double *U_y, double *U_z, double *U_t,
                   const int v_x, const int v_y, const int v_z, const int v_t,
                   const int s_x, const int s_y, const int s_z, const int s_t,
                   const double a, const bool dag) {

    double flag = (dag == true) ? -1 : 1;

    dslash_g5(src, dest, U_x, U_y, U_z, U_t, v_x, v_y, v_z, v_t, s_x, s_y, s_z, s_t, a, flag);
}

void ApplyOverlapQuda_gpu(double *dest, double *src, double k0, double k1, double k2, double prec,
                          std::vector<double *> evecs, std::vector<double> evals,
                          std::vector<std::vector<double> > coefs, std::vector<int> sizes,
                          double *U_x, double *U_y, double *U_z, double *U_t,
                          const int v_x, const int v_y, const int v_z, const int v_t,
                          const int s_x, const int s_y, const int s_z, const int s_t,
                          const double kappa) {

    ApplyOverlapQuda(dest, src, k0, k1, k2, prec, evecs, evals, coefs, sizes, U_x, U_y, U_z, U_t, v_x, v_y, v_z, v_t,
                     s_x, s_y, s_z, s_t, kappa);

}
