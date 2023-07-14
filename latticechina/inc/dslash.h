/**#
**# @file:   dslashMain.h
**# @brief:
**# @author: louis shaw
**# @data:   2021/08/17
#**/

#ifndef LATTICE_DSLASH_H
#define LATTICE_DSLASH_H

#include "lattice_fermion.h"
#include "lattice_gauge.h"
#include "lattice_propagator.h"
#include "operator.h"
#include "operator_mpi.h"
#include <vector>
#include "target/hip/legacyFunction.h"


void DslashEE(lattice_fermion &src, lattice_fermion &dest, const double mass);
void DslashOO(lattice_fermion &src, lattice_fermion &dest, const double mass);
//void DslashEO(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const bool dag);
//void DslashOE(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const bool dag);
void Dslashoffd(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const bool dag, int cb);

void DslashEE_gpu(lattice_fermion &src, lattice_fermion &dest, const double mass);
void DslashOO_gpu(lattice_fermion &src, lattice_fermion &dest, const double mass);
void Dslashoffd_gpu_apply(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const bool dag, int cb);
void Dslashoffd_gpu_apply(lattice_fermion &src, lattice_fermion &dest, double * U_x_g , double * U_y_g , double * U_z_g , double * U_t_g , const bool dag, int cb);


void Dslashoffd_gpu_apply(double * src_g, double * dest_g,  double * U_x_g , double * U_y_g , double * U_z_g , double * U_t_g, 
                          const int v_x, const int v_y, const int v_z, const int v_t, 
                          const int s_x, const int s_y, const int s_z, const int s_t,     
                          const bool dag, const int cb);


void  dslash_gpu_g5(double *src, double *dest, double *U_x, double *U_y, double *U_z, double *U_t,
         const int v_x, const int v_y, const int v_z, const int v_t,
         const int s_x, const int s_y, const int s_z, const int s_t,
         const double a, const bool dag);

void ApplyOverlapQuda_gpu(double *dest, double *src,double k0, double k1,double k2, double prec,
		          std::vector<double*> evecs, std::vector<double> evals,
               		  std::vector<std::vector<double> > coefs, std::vector<int> sizes,
       			  double   *U_x,   double *U_y,   double *U_z,   double *U_t,
          		  const int v_x, const int v_y, const int v_z, const int v_t,
           		  const int s_x, const int s_y, const int s_z, const int s_t,
           		  const double kappa );


#endif //LATTICE_DSLASH_H
