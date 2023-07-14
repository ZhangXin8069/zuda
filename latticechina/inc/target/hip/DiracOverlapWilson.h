//
// Created by louis on 2022/2/20.
//

#ifndef LATTICE_DIRACOVERLAPWILSON_H
#define LATTICE_DIRACOVERLAPWILSON_H

#include <vector>
#include "hip/hip_runtime.h"
#include "DiracSetup.h"

class DiracOverlapWilson:public DiracSetup {

protected:

    double prec0;
    std::vector<double *> hw_evec;
    std::vector<double> hw_eval;
    std::vector<std::vector<double>> coef;
    std::vector<int> hw_size;
    bool build_hw;

    void calc_coef(double cut);

public:

    double rho;
    double kappa;

    double *tmp1;
    double *tmp2;

    double *hw_evec_g;
    int * sig_hw_eval_g;

    DiracOverlapWilson(std::vector<double *> &evecs,
                       std::vector<double> &evals,
                       std::vector<std::vector<double> > &coefs,
                       std::vector<int> &sizes,
                       double *U_x_in, double *U_y_in, double *U_z_in, double *U_t_in,
                       const int v_x_in, const int v_y_in, const int v_z_in, const int v_t_in,
                       const int s_x_in, const int s_y_in, const int s_z_in, const int s_t_in,
                       const double kappa_in);

    ~DiracOverlapWilson();

    inline void dslash(double *src_g, double *dest_g, const int cb, const int flag, const double a, const double b);

    void Kernel(double *out, double *in);

    void KernelSq_scaled(double *out, double *in, double cut);

    void eps_l_g(double *out, double *in, int size);

    void general_dov(double *out, double *in, double k0, double k1, double k2, double prec);
};

#endif //LATTICE_DIRACOVERLAPWILSON_H
