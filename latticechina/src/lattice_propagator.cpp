/**#
**# @file:   lattice_propagator.cpp
**# @brief:
**# @author: louis shaw
**# @data:   2021/08/17
#**/

#include <mpi.h>
#include "lattice_propagator.h"
#include "operator.h"

/*
lattice_propagator::lattice_propagator(LatticePropagator &chroma_propagator, int *subgs1, int *site_vec1) {
    A = (complex<double> *) &(chroma_propagator.elem(0).elem(0, 0).elem(0, 0));
    subgs = subgs1;
    site_vec = site_vec1;
}
*/

lattice_propagator::lattice_propagator(complex<double> *chroma_propagator, int *subgs1, int *site_vec1) {
    A = chroma_propagator;
    subgs = subgs1;
    site_vec = site_vec1;
}

complex<double> lattice_propagator::peeksite(const int *site,
                                             int ii,           //ii=spin_row
                                             int jj,           //jj=spin_rank
                                             int ll,           //ll=color_row
                                             int mm){          //mm=color_rank

    int coords[4] = {site[0] / subgs[0], site[1] / subgs[1], site[2] / subgs[2], site[3] / subgs[3]};
    int N_sub[4] = {site_vec[0] / subgs[0], site_vec[1] / subgs[1], site_vec[2] / subgs[2], site_vec[3] / subgs[3]};
//    int rank = QMP_get_node_number(); //当前节点编号

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int nodenum = get_nodenum(coords, N_sub, 4); //当前坐标节点编号

    double dest[2];

    if (rank == nodenum) {
        int coord[4];
        for (int i = 0; i < 4; i++) { coord[i] = site[i]; }
        int subgrid_vol_cb = (subgs[0] * subgs[1] * subgs[2] * subgs[3]) >> 1;

        int subgrid_cb_nrow[4];
        for (int i = 0; i < 4; i++) { subgrid_cb_nrow[i] = subgs[i]; }

        subgrid_cb_nrow[0] >>= 1;

        int cb = 0;
        for (int m = 0; m < 4; ++m)
            cb += coord[m];
        cb &= 1;
        int subgrid_cb_coord[4];
        subgrid_cb_coord[0] = (coord[0] >> 1) % subgrid_cb_nrow[0];
        for (int i = 1; i < 4; ++i)
            subgrid_cb_coord[i] = coord[i] % subgrid_cb_nrow[i];


        int t = local_site2(subgrid_cb_coord, subgrid_cb_nrow) + cb * subgrid_vol_cb;
        dest[0] = A[t * 144 + ii * 36 + jj * 9 + ll * 3 + mm].real();
        dest[1] = A[t * 144 + ii * 36 + jj * 9 + ll * 3 + mm].imag();

    }
    if (nodenum == 0) {
        MPI_Bcast(dest, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        complex<double> dest2(dest[0], dest[1]);
        printf("finished a sendToWait node 0=%i \n", rank);
        return dest2;
    } else {
        if (rank == nodenum) {

            MPI_Send(dest, 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        MPI_Recv(dest, 2, MPI_DOUBLE, nodenum, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(dest, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    printf("finished a sendToWait node=rank  %i \n", rank);
    complex<double> dest2(dest[0], dest[1]);
    return dest2;
}

