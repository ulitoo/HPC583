#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <cblas.h>
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

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " N (Dimension of Matrix)  NB (Dimension of Block)" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);  // Matrix size (N x N)
    int NB = std::atoi(argv[2]); // Matrix block (NB x NB)

    // constants
    char transa = 'N';
    char transb = 'N';
    double alpha = 1.0;
    double beta = 0.0;
    int uno = 1;
    int zero = 0;
    int info, context, nprow, npcol, myrow, mycol, localrows, localcols;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set up the grid for Scalapack
    Cblacs_pinfo(&rank, &size);
    Cblacs_get(-1, 0, &context);

    // Determine the number of processes in each dimension of the grid
    nprow = static_cast<int>(sqrt(size));
    npcol = size / nprow;

    Cblacs_gridinit(&context, (char *)"Col-major", nprow, npcol);
    Cblacs_gridinfo(context, &nprow, &npcol, &myrow, &mycol);

    // Determine local matrix dimensions
    localrows = numroc_(&N, &NB, &myrow, &zero, &nprow);
    localcols = numroc_(&N, &NB, &mycol, &zero, &npcol);

    cout << "\nIn RANK: " << rank << " , nprow:" << nprow << " npcol:" << npcol << ":: Also :: localrows:" << localrows << " and localcols:" << localcols << " myrow:" << myrow << ", mycol:" << mycol << " \n";

    // Allocate memory for the local matrices
    double *A_local = new double[localrows * localcols];
    double *B_local = new double[localrows * localcols];
    double *C_local = new double[localrows * localcols];

    double *A_global = new double[N * N];
    double *B_global = new double[N * N];
    double *C_global = new double[N * N];
    double *C1_global = new double[N * N];

    // Local matrix descriptor
    int descA_local[9], descB_local[9], descC_local[9];
    descinit_(descA_local, &N, &N, &NB, &NB, &zero, &zero, &context, &localrows, &info);
    descinit_(descB_local, &N, &N, &NB, &NB, &zero, &zero, &context, &localrows, &info);
    descinit_(descC_local, &N, &N, &NB, &NB, &zero, &zero, &context, &localrows, &info);

    // Global matrix descriptor
    int descA_global[9], descB_global[9], descC_global[9];

    // Initialize the global matrix on the root process
    // if (rank == 0)
    //{
    // Initialize your global matrix A here
    // Example: A = identity matrix or   ( 1 2 3 ....)
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            // A_global[i + N * j] = (i == j) ? 1.0 : 0.0; // Identity
            // A_global[i + N * j] = i * N + j + 1; // 1 2 3 4 ...
            A_global[i + N * j] = (i + N * j) + 1;         // 1 1 1 ...
            B_global[i + N * j] = 1.0 / ((i + N * j) + 1); // 2 2 2 ...
        }
    }
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A_global, N, B_global, N, 0.0, C1_global, N);

    descinit_(descA_global, &N, &N, &NB, &NB, &zero, &zero, &context, &N, &info);
    descinit_(descB_global, &N, &N, &NB, &NB, &zero, &zero, &context, &N, &info);
    descinit_(descC_global, &N, &N, &NB, &NB, &zero, &zero, &context, &N, &info);

    // Distribute the global matrix
    // Cpdgemr2d(N, N, A_global, 1, 1, &descA_global, A_local, 1, 1, descA_local, context);

    // Print Global Matrix A and B
    cout << "GLOBAL A:\n";
    PrintColMatrix(A_global, N, N);
    cout << "GLOBAL B:\n";
    PrintColMatrix(B_global, N, N);
    cout << "GLOBAL AxB=C1:\n";
    PrintColMatrix(C1_global, N, N);
    //}

    // Distribute the global Matrices into the different local processors with 2D block Cyclic
    // pdgemr2d_(&n, &n, global_A.data(), &n, &n, desc_a, local_A.data(), &nloc, &nloc, desc_a, &ctxt);
    //pdgemr2d_(&N, &N, A_global, &N, &N, descA_global, A_local, &localrows, &localcols, descA_local, &context);
    
    
    // Gather info from local matrices based on local  -> Formulas in PDF

    for (int i = 0; i < localrows * localcols; ++i)
        A_local[i] = (i + (localrows * localcols * mycol)) + 1;
    
    
    for (int i = 0; i < localrows * localcols; ++i)
        B_local[i] = 1.0 / ((i + (localrows * localcols * mycol)) + 1);
    for (int i = 0; i < localrows * localcols; ++i)
        C_local[i] = 0.0;

    cout << "LOCAL A:\n";
    PrintColMatrix(A_local, localrows, localcols);
    cout << "LOCAL B:\n";
    PrintColMatrix(B_local, localrows, localcols);
    // Perform the matrix multiplication using pdgemm
    pdgemm_(&transa,&transb, &N, &N, &N, &alpha, A_local, &uno, &uno, descA_local, B_local, &uno, &uno, descB_local, &beta, C_local, &uno, &uno, descC_local);

    cout << "Rank: " << rank << " of Size:" << size << "\n";
    PrintColMatrix(C_local, localrows, localcols);

    // Deallocate memory and finalize BLACS
    // Clean up
    delete[] A_local;
    delete[] B_local;
    delete[] C_local;

    Cblacs_gridexit(context);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
