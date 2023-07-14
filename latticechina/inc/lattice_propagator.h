/**#
**# @file:   lattice_propagator.h
**# @brief:
**# @author: louis shaw
**# @data:   2021/08/17
#**/

#ifndef LATTICECHINA_LATTICE_PROPAGATOR_H
#define LATTICECHINA_LATTICE_PROPAGATOR_H

#include <complex>

using namespace std;

class lattice_propagator {
public:
    int *site_vec;
    complex<double> *A;
    int *subgs;

//    lattice_propagator(LatticePropagator &chroma_propagator, int *subgs1, int *site_vec1) {
//        A = (complex<double> *) &(chroma_propagator.elem(0).elem(0, 0).elem(0, 0));
//        subgs = subgs1;
//        site_vec = site_vec1;
//    }
    lattice_propagator(complex<double> *chroma_propagator, int *subgs1, int *site_vec1);

    complex<double> peeksite(const int *site, int ii = 0, int jj = 0, int ll = 0, int mm = 0);
};

#endif //LATTICECHINA_LATTICE_PROPAGATOR_H
