/**#
**# @file:   lattice_gauge.h
**# @brief:
**# @author: louis shaw
**# @data:   2021/08/17
#**/

#ifndef LATTICE_LATTICE_GAUGE_H
#define LATTICE_LATTICE_GAUGE_H

#include <complex>

using namespace std;

class lattice_gauge {
public:
    int *site_vec;
    complex<double> *A[4];
    int *subgs;

//    lattice_gauge(multi1d <LatticeColorMatrix> &chroma_gauge, int *subgs1, int *site_vec1);
    lattice_gauge(complex<double> *chroma_gauge[4], int *subgs1, int *site_vec1);

    complex<double> peeksite(const int *site, int ll = 0, int mm = 0, int dd = 0);
};

#endif //LATTICE_LATTICE_GAUGE_H
