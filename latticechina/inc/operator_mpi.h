/**#
**# @file:   operator_mpi.h
**# @brief:
**# @author: louis shaw
**# @data:   2021/08/17
#**/

#ifndef LATTICE_OPERATOR_MPI_H
#define LATTICE_OPERATOR_MPI_H

#include <mpi.h>

template<typename T>
double norm_2(const T s) {
//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//if (rank==0){
    std::complex<double> s1(0.0, 0.0);
    for (int i = 0; i < s.size; i++) {
        s1 += s.A[i] * conj(s.A[i]);
    }
    double sum_n = s1.real();
    double sum;
    MPI_Reduce(&sum_n, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    return sum;

//    return s1.real();
//    }else
//    {return 0;}
}

#endif //LATTICE_OPERATOR_MPI_H
