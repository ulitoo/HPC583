#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include "scalapack.h"
#include "pblas.h"
#include "PBblacs.h"

#include "redist.h"

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

    int info, iam, nprocs, ictxt, nprow, npcol, myrow, mycol, localrows, localcols;
    int N = 8; // Global matrix dimensions
    int NB = 2; // Block size
    int zero = 0;
    int uno = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &iam);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Initialize BLACS
    Cblacs_pinfo(&iam, &nprocs);
    Cblacs_get(-1, 0, &ictxt);

    // Determine the number of processes in each dimension of the grid
    nprow = static_cast<int>(sqrt(nprocs));
    npcol = nprocs / nprow;

    // Initialize the process grid
    char tmp[10]="Col-major";
    Cblacs_gridinit(&ictxt, tmp, nprow, npcol);
    //Cblacs_pcoord(ictxt, iam, &myrow, &mycol);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // Determine local matrix dimensions
    localrows = numroc_(&N, &NB, &myrow, &zero, &nprow);
    localcols = numroc_(&N, &NB, &mycol, &zero, &npcol);

    // Allocate local matrix
    double* A_local = new double[localrows * localcols];
    
    // Initialize the global matrix on the root process
    if (myrow == 0 && mycol == 0)
    {
        double* A_global = new double[N * N];

        // Initialize your global matrix A here
        // Example: A = identity matrix
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                A_global[i + N * j] = (i == j) ? 1.0 : 0.0;
            }
        }
    
        PrintColMatrix(A_global,N,N);
    }
    // Global matrix descriptor
    //int descA[9];
    MDESC descA;
    //descinit_(descA, &N, &N, &NB, &NB, &nprow, &npcol, &ictxt, &N, &info);

    // Local matrix descriptor
    MDESC descA_local;
    //int descA_local[9];
    //descinit_(descA_local, &N, &N, &NB, &NB, &nprow, &npcol, &ictxt, &localrows, &info);

    // Distribute the global matrix

    //Cpdgemr2d(N, N, A_global, 1, 1, &descA, A_local, 1, 1, &descA_local, ictxt);

    // Perform computations on the local matrix using Scalapack functions

    // Deallocate memory and finalize BLACS
    delete[] A_local;
    Cblacs_gridexit(ictxt);

    MPI_Finalize();

    return 0;
}
