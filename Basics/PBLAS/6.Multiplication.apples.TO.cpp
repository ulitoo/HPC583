#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <cblas.h>
#include <thread>
#include <lapacke.h>
#include <fstream>
#include <random>
#include "scalapack.h"
#include "JBG_BLAS.h"

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        //std::cerr << "Usage: " << argv[0] << " N (Dimension of Matrix)  NB (Dimension of Block) dime (what process local info to show)" << std::endl;
        std::cerr << "Usage: " << argv[0] << " N (Dimension of Matrix)  NB (Dimension of Block) iter (iterations)" << std::endl;
        // std::cerr << "Usage: " << argv[0] << " N (Dimension of Matrix)  NB (Dimension of Block)" << std::endl;
        return 1;
    }
    // Create a random number generator =>  Get a Seed from random device

    int seed = 13;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    int N = std::atoi(argv[1]);  // Matrix size (N x N)
    int NB = std::atoi(argv[2]); // Matrix block (NB x NB)
    int iter = std::atoi(argv[3]);
    int M = N;
    int MB = NB;

    // constants
    char transa = 'N';
    char transb = 'N';
    double alpha = 1.0;
    double beta = 0.0;
    int uno = 1;
    int zero = 0;
    int info, context, nprow, npcol, myprow, mypcol, localrows, localcols;

    // Timers
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time_Global, elapsed_time_Paralel, elapsed_time_Collect, elapsed_time_Scatter;

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
    Cblacs_gridinfo(context, &nprow, &npcol, &myprow, &mypcol);

    // Determine local matrix dimensions
    localrows = numroc_(&N, &NB, &myprow, &zero, &nprow);
    localcols = numroc_(&N, &NB, &mypcol, &zero, &npcol);

    // Allocate memory for the local matrices
    double *A_local = new double[localrows * localcols];
    double *B_local = new double[localrows * localcols];
    double *C_local = new double[localrows * localcols];

    double *A_global = nullptr;  //= new double[N * N];
    double *B_global = nullptr;  // new double[N * N];
    double *C_global = nullptr;  // new double[N * N];
    double *C1_global = nullptr; // new double[N * N];
    double *ellaps_Vector = nullptr;

    // Local matrix descriptor
    int descA_local[9], descB_local[9], descC_local[9];
    descinit_(descA_local, &N, &N, &NB, &NB, &zero, &zero, &context, &localrows, &info);
    descinit_(descB_local, &N, &N, &NB, &NB, &zero, &zero, &context, &localrows, &info);
    descinit_(descC_local, &N, &N, &NB, &NB, &zero, &zero, &context, &localrows, &info);

    // Initialize the global matrices on the root process
    if (rank == 0)
    {
        A_global = new double[N * N];
        B_global = new double[N * N];
        C_global = new double[N * N];
        C1_global = new double[N * N];

        for (int k = 0; k < N * N; k++)
        {
            A_global[k] = dist(rng) - 0.5;
            B_global[k] = dist(rng) - 0.5;
        }
        for (int k = 0; k < iter; k++)
        {

            start = std::chrono::high_resolution_clock::now();
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A_global, N, B_global, N, 0.0, C1_global, N);
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            elapsed_time_Global += duration.count() * 1.e-9;
        }
        elapsed_time_Global /= iter;
        cout << "\nCblas dgemm time:" << elapsed_time_Global << " sec.\n";
    }

    // Scatter the global Matrices into the different local processors with 2D block Cyclic
    ScatterMatrix(context, A_global, M, N, MB, NB, A_local, localrows, localcols, myprow, mypcol, nprow, npcol);
    ScatterMatrix(context, B_global, M, N, MB, NB, B_local, localrows, localcols, myprow, mypcol, nprow, npcol);


    elapsed_time_Paralel = 0.0;
    for (int i = 0; i < iter; i++)
    {
        start = std::chrono::high_resolution_clock::now();
        pdgemm_(&transa, &transb, &N, &N, &N, &alpha, A_local, &uno, &uno, descA_local, B_local, &uno, &uno, descB_local, &beta, C_local, &uno, &uno, descC_local);
        //CollectMatrix(context, C_global, M, N, MB, NB, C_local, localrows, localcols, myprow, mypcol, nprow, npcol);

        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
        elapsed_time_Paralel += duration.count() * 1.e-9;
    }
    elapsed_time_Paralel /= iter;

    if (rank == 0)
    {
        cout << "\nPDGEMM + Collect time:" << elapsed_time_Paralel << " sec.\n";
        cout << "\nSpeed-UP:" << elapsed_time_Global / (elapsed_time_Paralel) << "x.\n";

        // double C_error = Fwd_Error_diff(C1_global, C_global, N, N);
        // double C_residual = InfinityNorm_Error_diff(C1_global, C_global, N, N);
        // std::cout << "\nCollected FWD Error: " << C_error << std::endl;
        // std::cout << "\nCollected residual Error: " << C_residual << std::endl;
        delete[] A_global;
        delete[] B_global;
        delete[] C_global;
        delete[] C1_global;
    }
    Cblacs_gridexit(context);

    // Finalize MPI
    MPI_Finalize();
    delete[] A_local;
    delete[] B_local;
    delete[] C_local;

    return 0;
}
