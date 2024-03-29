#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <cblas.h>
#include <thread>
#include <lapacke.h>
#include <fstream>
#include <random>
#include "scalapack.h"
#include "JBG_BLAS.s.h"

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        // std::cerr << "Usage: " << argv[0] << " N (Dimension of Matrix)  NB (Dimension of Block) dime (what process local info to show)" << std::endl;
        std::cerr << "Usage: " << argv[0] << " N (Dimension of Matrix)  NB (Dimension of Block)" << std::endl;
        return 1;
    }
    // Create a random number generator =>  Get a Seed from random device
    
    int seed = 13;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    int N = std::atoi(argv[1]);  // Matrix size (N x N)
    int NB = std::atoi(argv[2]); // Matrix block (NB x NB)
    // int dime = std::atoi(argv[3]);
    int M = N;
    int MB = NB;

    // constants
    char transa = 'N';
    char transb = 'N';
    float alpha = 1.0;
    float beta = 0.0;
    int uno = 1;
    int zero = 0;
    int info, context, nprow, npcol, myprow, mypcol, localrows, localcols;

    // Timers
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time_Global, elapsed_time_Paralel,elapsed_time_Collect,elapsed_time_Scatter ;

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

    cout << "\nIn RANK: " << rank << " , nprow:" << nprow << " npcol:" << npcol << ":: Also :: localrows:" << localrows << " and localcols:" << localcols << " myrow:" << myprow << ", mycol:" << mypcol << " \n";

    // Allocate memory for the local matrices
    float *A_local = new float[localrows * localcols];
    float *B_local = new float[localrows * localcols];
    float *C_local = new float[localrows * localcols];

    float *A_global = nullptr;  //= new float[N * N];
    float *B_global = nullptr;  // new float[N * N];
    float *C_global = nullptr;  // new float[N * N];
    float *C1_global = nullptr; // new float[N * N];

    // Local matrix descriptor
    int descA_local[9], descB_local[9], descC_local[9];
    descinit_(descA_local, &N, &N, &NB, &NB, &zero, &zero, &context, &localrows, &info);
    descinit_(descB_local, &N, &N, &NB, &NB, &zero, &zero, &context, &localrows, &info);
    descinit_(descC_local, &N, &N, &NB, &NB, &zero, &zero, &context, &localrows, &info);

    // Print descA_local
    // cout << "\nDESCALOCAL: " << descA_local[0]  << " , " << descA_local[1]  << " , " << descA_local[2]  << " , " << descA_local[3]  << " , " << descA_local[4]  << " , " << descA_local[5]  << " , " << descA_local[6]  << " , " << descA_local[7] << " , " << descA_local[8] << " ,info: " << info << " \n";

    // Initialize the global matrices on the root process
    if (rank == 0)
    {
        A_global = new float[N * N];
        B_global = new float[N * N];
        C_global = new float[N * N];
        C1_global = new float[N * N];
/*
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                // A_global[i + N * j] = (i == j) ? 1.0 : 0.0; // Identity
                // A_global[i + N * j] = i * N + j + 1; // 1 2 3 4 ...
                A_global[i + N * j] = (i + N * j) + 1;         // 1 2 3 ...
                B_global[i + N * j] = 1.0 / ((i + N * j) + 1); // 1/1 1/2 1/3 1/4 ...
            }
        }
*/
        for (int k = 0; k < N * N; k++)
        {
            A_global[k] = dist(rng) - 0.5;
            B_global[k] = dist(rng) - 0.5;
        }
        start = std::chrono::high_resolution_clock::now();
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A_global, N, B_global, N, 0.0, C1_global, N);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
        elapsed_time_Global = duration.count() * 1.e-9;
        cout << "\nCblas dgemm time:" << elapsed_time_Global << " sec.\n";

        // Print Global Matrix A and B
        /*cout << "GLOBAL A:\n";
        PrintColMatrix(A_global, N, N);
        cout << "GLOBAL B:\n";
        PrintColMatrix(B_global, N, N);
        cout << "GLOBAL AxB=C1:\n";
        PrintColMatrix(C1_global, N, N);
        */
    }

    // Scatter the global Matrices into the different local processors with 2D block Cyclic
    start = std::chrono::high_resolution_clock::now();
    ScatterMatrix(context, A_global, M, N, MB, NB, A_local, localrows, localcols, myprow, mypcol, nprow, npcol);
    ScatterMatrix(context, B_global, M, N, MB, NB, B_local, localrows, localcols, myprow, mypcol, nprow, npcol);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time_Scatter = duration.count() * 1.e-9;
    cout << "\nScatter time:" << elapsed_time_Scatter << " sec.\n";

    // int ipiv[N];
        
    // Perform the matrix multiplication using pdgemm
    start = std::chrono::high_resolution_clock::now();
    psgemm_(&transa, &transb, &N, &N, &N, &alpha, A_local, &uno, &uno, descA_local, B_local, &uno, &uno, descB_local, &beta, C_local, &uno, &uno, descC_local);
    // Solver
    //pdgesv_(&N, &N, A_local, &uno, &uno, descA_local, ipiv, B_local, &uno, &uno, descB_local, &info);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time_Paralel = duration.count() * 1.e-9;
    cout << "\nScalapack pdgemm of Rank:" << rank << ", time:" << elapsed_time_Paralel << " sec.\n";
    
    /*
    if (rank == dime)
    {
        cout << "\nIn RANK: " << rank << " , nprow:" << nprow << " npcol:" << npcol << ":: Also :: localrows:" << localrows << " and localcols:" << localcols << " myrow:" << myprow << ", mycol:" << mypcol << " \n";
        std::cout << "Scattered Local C Matrix:" << std::endl;
        PrintColMatrix(C_local, localrows, localcols);
    }
    */
    start = std::chrono::high_resolution_clock::now();
    CollectMatrix(context, C_global, M, N, MB, NB, C_local, localrows, localcols, myprow, mypcol, nprow, npcol);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time_Collect = duration.count() * 1.e-9;
    cout << "\nCollect time:" << elapsed_time_Collect << " sec.\n";

    if (rank == 0)
    {
        cout << "\nPDGEMM + Collect time:" << elapsed_time_Paralel << " sec.\n"; 
        cout << "\nSpeed-UP:" << elapsed_time_Global/(elapsed_time_Paralel) << "x.\n"; 
        
        
        float C_error = Fwd_Error_diff(C1_global, C_global, N, N);
        float C_residual = InfinityNorm_Error_diff(C1_global, C_global, N, N);
        // std::cout << "Collected Matrix:" << std::endl;
        // PrintColMatrix(C_global, M, N);
        std::cout << "\nCollected FWD Error: " << C_error << std::endl;
        std::cout << "\nCollected residual Error: " << C_residual << std::endl;
    }

    // Deallocate memory and finalize BLACS
    // Clean up
    delete[] A_local;
    delete[] B_local;
    delete[] C_local;
    if (rank == 0)
    {
        delete[] A_global;
        delete[] B_global;
        delete[] C_global;
        delete[] C1_global;
    }
    Cblacs_gridexit(context);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
