#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <cblas.h>
#include <cmath>
#include "scalapack.h"

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

void printMatrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 4; // Global matrix size
    int NB = 2; // Block size
    int nprow = 2; // Number of rows in the process grid
    int npcol = 2; // Number of columns in the process grid
    int myrow, mycol;
    int uno = 1;
    int dos = 3;
    int context, info;
    int descA[9], descA_local[9];
    int irsrc = 0, icsrc = 0; // Source process coordinates
    char ColMajor[10] = "Col-major";

    Cblacs_pinfo(&rank, &size);
    Cblacs_get(-1, 0, &context);
    Cblacs_gridinit(&context, ColMajor, nprow, npcol);
    Cblacs_gridinfo(context, &nprow, &npcol, &myrow, &mycol);

    int localrows = numroc_(&N, &NB, &myrow, &irsrc, &nprow);
    int localcols = numroc_(&N, &NB, &mycol, &icsrc, &npcol);

    double *A_global = nullptr;
    double *A_local = new double[localrows * localcols];

    if (rank == 0) {
        // Initialize global matrix on root process
        A_global = new double[N * N];
        for (int i = 0; i < N * N; ++i) {
            A_global[i] = i + 1;
        }
        std::cout << "Global Matrix A:\n";
        PrintColMatrix(A_global, N, N);
        descinit_(descA, &N, &N, &NB, &NB, &irsrc, &icsrc, &context, &N, &info);
    }
    
    descinit_(descA_local, &N, &N, &NB, &NB,&irsrc, &icsrc, &context, &localrows, &info); // Initialize descA_local
    //pdgemr2d_(&N, &N, A_global,&uno, &uno, descA, A_local, &uno, &uno, descA_local, &context, &irsrc, &icsrc);

    pdgemr2d_(&N, &N, A_global,&myrow, &myrow, descA, A_local, &myrow, &mycol, descA_local, &context, &irsrc, &icsrc);

    cout << "\nIn RANK: " << rank << " , nprow:" << nprow << " npcol:" << npcol << ":: Also :: localrows:" << localrows << " and localcols:" << localcols << " myrow:" << myrow << ", mycol:" << mycol << " \n";

    std::cout << "Process " << rank << " Local Matrix A:\n";
    PrintColMatrix  (A_local, localrows, localcols);

    delete[] A_local;
    if (rank == 0)
        delete[] A_global;

    Cblacs_gridexit(context);
    MPI_Finalize();

    return 0;
}
