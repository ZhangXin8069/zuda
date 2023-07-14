/**#
**# @file:   lattice_fermion.h
**# @brief:
**# @author: louis shaw
**# @data:   2021/08/17
#**/

#ifndef LATTICE_LATTICE_FERMION_H
#define LATTICE_LATTICE_FERMION_H

#include <complex>
#include <vector>

using namespace std;

class lattice_fermion {
public:
    int *site_vec;
    complex<double> *A;
    int *subgs;
    int size;

//    lattice_fermion(LatticeFermion &chroma_fermi);
    lattice_fermion(complex<double> *chroma_fermi, int *subgs1, int *site_vec1);
    lattice_fermion(int *subgs1, int *site_vec1);

    lattice_fermion &operator-(const lattice_fermion &a);
    lattice_fermion &operator+(const lattice_fermion &a);

    void clean();

//    complex<double> peeksite(vector<int> site, vector<int> site_vec, int ii = 0, int ll = 0);
};

void Minus(lattice_fermion &src1, lattice_fermion &src2, lattice_fermion &a);
void Plus(lattice_fermion &src1, lattice_fermion &src2, lattice_fermion &a);

#endif //LATTICE_LATTICE_FERMION_H
