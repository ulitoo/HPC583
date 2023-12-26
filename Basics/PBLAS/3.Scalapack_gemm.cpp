#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include "scalapack.h"
#include "pblas.h"
#include "PBblacs.h"
//#include "redist.h"

using namespace std;

void PrintColMatrix(double *matrix, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (matrix[i + (j * m)] < 0.00000000000000000001)
            {
                cout << 0.0 << " ";
            }
            else
            {
                cout << matrix[i + (j * m)] << " ";
            }
        }
        cout << "\n";
    }
    cout << "\n";
}

/*extern "C" {
#include <cblacs.h>
#include <mpi.h>
#include <scalapack.h>
}*/

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int iam, nprocs, ictxt, nprow, npcol, myrow, mycol, nrows, ncols;
    int N = 1000; // Global matrix dimensions
    int NB = 128; // Block size

    MPI_Comm_rank(MPI_COMM_WORLD, &iam);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Initialize BLACS
    Cblacs_pinfo(&iam, &nprocs);
    Cblacs_get(-1, 0, &ictxt);

    // Determine the number of processes in each dimension of the grid
    nprow = static_cast<int>(sqrt(nprocs));
    npcol = nprocs / nprow;

    // Initialize the process grid
    Cblacs_gridinit(&ictxt, "Row-major", nprow, npcol);
    Cblacs_pcoord(ictxt, iam, &myrow, &mycol);

    // Determine local matrix dimensions
    nrows = numroc_(&N, &NB, &myrow, &CBLACS_ZERO, &nprow);
    ncols = numroc_(&N, &NB, &mycol, &CBLACS_ZERO, &npcol);

    // Allocate local matrix
    double* A_local = new double[nrows * ncols];

    // Global matrix descriptor
    int descA[9];
    descinit_(descA, &N, &N, &NB, &NB, &CBLACS_ZERO, &CBLACS_ZERO, &ictxt, &N, &info);

    // Local matrix descriptor
    int descA_local[9];
    descinit_(descA_local, &N, &N, &NB, &NB, &CBLACS_ZERO, &CBLACS_ZERO, &ictxt, &nrows, &info);

    // Distribute the global matrix
    pdgemr2d_(&N, &N, A_global, &CBLACS_ONE, &CBLACS_ONE, descA, A_local, &nrows, &CBLACS_ONE, &CBLACS_ONE, descA_local, &ictxt);

    // Perform computations on the local matrix using Scalapack functions

    // Deallocate memory and finalize BLACS
    delete[] A_local;
    Cblacs_gridexit(ictxt);

    MPI_Finalize();

    return 0;
}
