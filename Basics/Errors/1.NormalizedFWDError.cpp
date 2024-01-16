#include <cblas.h>
#include <iostream>
#include <thread>
#include <lapacke.h>
#include <random>
#include "JBG_BLAS.h"

using namespace std;

///     MAIN :: For the sake of simplicity we will Consider all square matrices n x n AND n is multiple of 2
///     This one will do calculations from 1 to 512 and print the results

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " n (Dimension of Matrix) s (seed)" << std::endl;
        return 1;
    }
    const int n = std::atoi(argv[1]);
    const int seed = std::atoi(argv[2]);

    // Create a random number generator =>  Get a Seed from random device
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Timers
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time_BLAS, elapsed_time_mine;

    // Alloc Space for MATRICES Needed in Column Major Order
    double *matrixA = (double *)malloc(n * n * sizeof(double));
    double *matrixB = (double *)malloc(n * n * sizeof(double));
    double *matrixBPivot = (double *)malloc(n * n * sizeof(double));
    double *matrixL = (double *)malloc(n * n * sizeof(double));
    double *matrixU = (double *)malloc(n * n * sizeof(double));
    double *matrixP = (double *)malloc(n * n * sizeof(double)); // Permutation Matrix
    double *matrixY = (double *)malloc(n * n * sizeof(double));
    double *matrixX = (double *)malloc(n * n * sizeof(double));
    double *matrixA_original = (double *)malloc(n * n * sizeof(double)); // in case they get overwritten
    double *matrixB_original = (double *)malloc(n * n * sizeof(double)); // in case they get overwritten

    // Other Variables
    int INFO, IPIV[n], IPIVmine[n];

    // Create the matrices A and B and fill it with random values
    for (int i = 0; i < n * n; i++)
    {
        matrixA[i] = dist(rng);
        matrixB[i] = dist(rng);
    }

    // Backup A and B Matrices
    Write_A_over_B(matrixA, matrixA_original, n, n);
    Write_A_over_B(matrixB, matrixB_original, n, n);

    // ----------------- Start PIVOTED Algorithm HERE!

    start = std::chrono::high_resolution_clock::now();
    LUdecompositionRecursive4Pivot(matrixA, matrixL, matrixU, IPIVmine, n, n);
    ipiv_to_P(IPIVmine, n, matrixP);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, matrixP, n, matrixB, n, 0.0, matrixBPivot, n);
    LowerTriangularSolverRecursiveReal_0(matrixL, matrixBPivot, matrixY, n, n);
    UpperTriangularSolverRecursiveReal_0(matrixU, matrixY, matrixX, n, n);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time_mine = duration.count() * 1.e-9;

    cout << "\nCheck Accuracy and time of my AX=B (My Pivoted Recursive Algorithm):";
    ErrorCalc_Display_v2(matrixA_original, matrixB_original, matrixX, elapsed_time_mine, n, n);

    // Restore A and B Matrices After Calculation
    Write_A_over_B(matrixA_original, matrixA, n, n);
    Write_A_over_B(matrixB_original, matrixB, n, n);

    //  ----------------- Solve BLAS and compare with my implementation HERE!

    start = std::chrono::high_resolution_clock::now();
    LAPACK_dgesv(&n, &n, matrixA, &n, IPIV, matrixB, &n, &INFO);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time_BLAS = duration.count() * 1.e-9;

    cout << "Check Accuracy and time of LAPACK (dgesv): ";
    ErrorCalc_Display_v2(matrixA_original, matrixB_original, matrixB, elapsed_time_BLAS, n, n);

    cout << "Solution Calculation Speedup from BLAS to my_Pivot: " << (elapsed_time_mine) / elapsed_time_BLAS << "x.\n";

    return 0;
}