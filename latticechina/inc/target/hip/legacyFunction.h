//
// Created by louis on 2021/8/29.
//

#ifndef LATTICE_LEGACYFUNCTION_H
#define LATTICE_LEGACYFUNCTION_H

#include <vector>

int test(double *src, double *dest, double *U_x, double *U_y, double *U_z, double *U_t,
         const int v_x, const int v_y, const int v_z, const int v_t,
         const int s_x, const int s_y, const int s_z, const int s_t,
         const int cb, const int flag);

int dslash_g5(double *src, double *dest, double *U_x, double *U_y, double *U_z, double *U_t,
         const int v_x, const int v_y, const int v_z, const int v_t,
         const int s_x, const int s_y, const int s_z, const int s_t,
         const double a, const int flag);

void ApplyOverlapQuda(double *dest, double *src,double k0, double k1,double k2, double prec,
                std::vector<double*> &evecs, std::vector<double> &evals,
                std::vector<std::vector<double> > &coefs, std::vector<int> &sizes,
                double   *U_x,   double *U_y,   double *U_z,   double *U_t,
                const int v_x, const int v_y, const int v_z, const int v_t,
                const int s_x, const int s_y, const int s_z, const int s_t,
                const double kappa );

void ApplyOverlapQuda(double *dest, double *src, double k0, double k1, double k2, double prec, void *ov_instance, int size);

void * newApplyOverlapQuda( std::vector<double *> &evecs, std::vector<double> &evals,
                      std::vector<std::vector<double> > &coefs, std::vector<int> &sizes,
                      double *U_x, double *U_y, double *U_z, double *U_t,
                      const int v_x, const int v_y, const int v_z, const int v_t,
                      const int s_x, const int s_y, const int s_z, const int s_t,
                      const double kappa  );

void delApplyOverlapQuda(void *ov_instance);

void ApplyWilsonQuda(double *dest, double *src, void *ov_instance);

void * newApplyWilsonQuda(double *U_x, double *U_y, double *U_z, double *U_t,
                     	  const int v_x, const int v_y, const int v_z, const int v_t,
                     	  const int s_x, const int s_y, const int s_z, const int s_t);

#endif














