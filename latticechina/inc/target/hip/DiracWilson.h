//
// Created by louis on 2022/2/20.
//

#ifndef LATTICE_DIRACWILSON_H
#define LATTICE_DIRACWILSON_H

#include "DiracSetup.h"

class DiracWilson:public DiracSetup{

public:

    DiracWilson(double *U_x_in, double *U_y_in, double *U_z_in, double *U_t_in,
                const int v_x_in, const int v_y_in, const int v_z_in, const int v_t_in,
                const int s_x_in, const int s_y_in, const int s_z_in, const int s_t_in);

    ~DiracWilson();

    void dslash(double *src_g, double *dest_g, const int cb, const int flag);

    void Kernal(double *out, double *in);

};

#endif //LATTICE_DIRACWILSON_H
