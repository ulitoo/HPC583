#include <cblas.h>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>
#include <pthread.h>
#include <thread>
#include <lapacke.h>
#include <random>
#include "JBG_BLAS.h"

using namespace std;

///     MAIN :: For the sake of simplicity we will Consider all square matrices n x n

int main(int argc, char *argv[])
{

    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " n (Dimension of Matrix) C (expected condition)" << std::endl;
        return 1;
    }
    const int n = std::atoi(argv[1]);
    const int expectedcondition = std::atoi(argv[2]);

    // Create a random number generator
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    double *matrixA = (double *)malloc(n * n * sizeof(double));
    double *matrixB = (double *)malloc(n * n * sizeof(double));
    
    // Create the matrix and fill it with random values
    while (1)
    {
        for (int i = 0; i < n * n; i++)
        {
            matrixA[i] = dist(rng);
        }
        if (ConditionNumber(matrixA, n, n) < (double)expectedcondition)
        {
            break;
        }
    }
    while (1)
    {
        for (int i = 0; i < n * n; i++)
        {
            matrixB[i] = dist(rng);
        }
        if (ConditionNumber(matrixB, n, n) < (double)expectedcondition)
        {
            break;
        }
    }

    cout << "\nMatrix A Condition Number: " << ConditionNumber(matrixA,n,n) << "\n";
    cout << "Matrix B Condition Number: " << ConditionNumber(matrixB,n,n) << "\n\n";
    
    PrintColMatrix(matrixA,n,n);
    cout << "\n";
    PrintColMatrix(matrixB,n,n);
    cout << "\n";
    

    ///   *******************************  FILES WRITTEN in files

    // Timers
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time_BLAS, elapsed_time_Pivot, elapsed_time_nonPivot, elapsed_time_Solve;

    // Alloc Space for MATRICES Needed in Column Major Order
    double *matrixBPivot = (double *)malloc(n * n * sizeof(double));
    double *matrixL = (double *)malloc(n * n * sizeof(double));
    double *matrixU = (double *)malloc(n * n * sizeof(double));
    double *matrixP = (double *)malloc(n * n * sizeof(double)); // Permutation Matrix
    double *matrixY = (double *)malloc(n * n * sizeof(double));
    double *matrixX = (double *)malloc(n * n * sizeof(double));
    double *matrixA_original = (double *)malloc(n * n * sizeof(double)); // in case they get overwritten
    double *matrixB_original = (double *)malloc(n * n * sizeof(double)); // in case they get overwritten

    // Other Variables
    double AConditionNumber;
    int INFO, IPIV[n], IPIVmine[n];

    // Backup A and B Matrices
    Write_A_over_B(matrixA, matrixA_original, n, n);
    Write_A_over_B(matrixB, matrixB_original, n, n);

    // ----------------- Calculate Condition Number of Matrix A

    AConditionNumber = ConditionNumber(matrixA, n, n);
    cout << "Matrix A Condition Number: " << (AConditionNumber) << "\n";

    // ----------------- Start PIVOTED Algorithm HERE!

    start = std::chrono::high_resolution_clock::now();
    // Recursive Implementation of LU decomposition for PA -> PIVOTED
    LUdecompositionRecursive4Pivot(matrixA, matrixL, matrixU, IPIVmine, n, n);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time_Pivot = duration.count() * 1.e-9;

    // Create Permutation Matrix based on the swap vector (indices of swapped rows in algo)
    ipiv_to_P(IPIVmine, n, matrixP);

    // Now use BPivot instead of B for Solving LUX=PB -> PAX=PB -> PA=LU
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, matrixP, n, matrixB, n, 0.0, matrixBPivot, n);

    // Now Solve system of linear equations given AX=B given B is n x n
    // Solve PAX=PB -> LUX=BPivot -> (2) UX=Y -> (1) LY=BPivot
    // Solve (1) LY=BPivot
    start = std::chrono::high_resolution_clock::now();
    LowerTriangularSolverRecursiveReal_0(matrixL, matrixBPivot, matrixY, n, n);
    //LowerTriangularSolverNaiveReal(matrixL, matrixBPivot, matrixY, n);
    // Solve (2) UX=Y
    UpperTriangularSolverRecursiveReal_0(matrixU, matrixY, matrixX, n, n);
    //UpperTriangularSolverNaiveReal(matrixU, matrixY, matrixX, n);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time_Solve = duration.count() * 1.e-9;

    cout << "\nCheck Accuracy and time of my AX=B (Pivoted):";
    ErrorCalc_Display(matrixA_original, matrixB_original, matrixX, elapsed_time_Pivot + elapsed_time_Solve, n, n);

    // Restore A and B Matrices After Calculation
    Write_A_over_B(matrixA_original, matrixA, n, n);
    Write_A_over_B(matrixB_original, matrixB, n, n);

    // ----------------- Start Non-PIVOTED Algorithm HERE!
    // Reset Result Matrices X and Y
    MakeZeroes(matrixY, n, n);
    MakeZeroes(matrixX, n, n);

    start = std::chrono::high_resolution_clock::now();
    // Recursive Implementation of LU decomposition for A -> NON - PIVOTED
    LUdecompositionRecursive2(matrixA, matrixL, matrixU, n, n);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time_nonPivot = duration.count() * 1.e-9;

    // Now Solve system of linear equations given AX=B given B is n x n
    // Solve AX=B -> LUX=B -> (2) UX=Y -> (1) LY=B
    // Solve (1) LY=B
    start = std::chrono::high_resolution_clock::now();
    LowerTriangularSolverRecursiveReal_0(matrixL, matrixB, matrixY, n, n);
    // Solve (2) UX=Y
    UpperTriangularSolverRecursiveReal_0(matrixU, matrixY, matrixX, n, n);
    stop = std::chrono::high_resolution_clock::now();

    cout << "Check Accuracy and time of my AX=B (non-Pivoted):";
    ErrorCalc_Display(matrixA_original, matrixB_original, matrixX, elapsed_time_nonPivot + elapsed_time_Solve, n, n);

    // Restore A and B Matrices After Calculation
    Write_A_over_B(matrixA_original, matrixA, n, n);
    Write_A_over_B(matrixB_original, matrixB, n, n);

    // ---------------- Done! Now to Show the Results and Compare with BLAS

    //   Solve BLAS and compare with my implementation

    start = std::chrono::high_resolution_clock::now();
    LAPACK_dgesv(&n, &n, matrixA, &n, IPIV, matrixB, &n, &INFO);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time_BLAS = duration.count() * 1.e-9;

    cout << "Check Accuracy and time of BLAS (dgesv): ";
    ErrorCalc_Display(matrixA_original, matrixB_original, matrixB, elapsed_time_BLAS, n, n);

    cout << "Pivot LU decomposition: " << (elapsed_time_Pivot) << " s.\n";
    cout << "Non-Pivot LU decomposition: " << (elapsed_time_nonPivot) << " s.\n";
    cout << "Lower + Upper Solve: " << (elapsed_time_Solve) << " s.\n\n";

    cout << "Solution Calculation Speedup from BLAS to my_Pivot: " << (elapsed_time_Pivot + elapsed_time_Solve) / elapsed_time_BLAS << "x.\n\n";
    cout << "Solution Calculation Speedup from BLAS to my_nonPivot: " << (elapsed_time_nonPivot + elapsed_time_Solve) / elapsed_time_BLAS << "x.\n\n";

    return 0;
}