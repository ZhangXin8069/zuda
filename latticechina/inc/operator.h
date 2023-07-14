/**#
**# @file:   operator.h
**# @brief:
**# @author: louis shaw
**# @data:   2021/08/17
#**/

#ifndef LATTICECHINA_OPERATOR_H
#define LATTICECHINA_OPERATOR_H

#include <complex>

static int local_site2(const int *coord, const int *latt_size) {
    int order = 0;
    for (int mmu = 3; mmu >= 1; --mmu) {
        order = latt_size[mmu - 1] * (coord[mmu] + order);
    }
    order += coord[0];
    return order;
}

static int get_nodenum(const int *x, int *l, int nd) {
    int i, n;
    n = 0;
    for (i = nd - 1; i >= 0; i--) {
        int k = i;
        n = (n * l[k]) + x[k];
    }
    return n;
}

// TODO: fix
template<typename T>
double norm_2_E(const T s) {\

    std::complex<double> s1(0.0, 0.0);
    for (int i = 0; i < (s.size / 2); i++) {
        s1 += s.A[i] * conj(s.A[i]);
    }
    return s1.real();
}

// TODO: fix
template<typename T>
double norm_2_O(const T s) {

    std::complex<double> s1(0.0, 0.0);
    for (int i = s.size / 2; i < s.size; i++) {
        s1 += s.A[i] * conj(s.A[i]);
    }
    return s1.real();
};

#endif //LATTICECHINA_OPERATOR_H
