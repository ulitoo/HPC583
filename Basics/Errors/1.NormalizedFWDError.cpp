#include <cblas.h>
#include <iostream>
#include <thread>
#include <lapacke.h>
#include <random>
#include "JBG_BLAS.h"

using namespace std;

///     MAIN :-> For the sake of simplicity we will Consider all square matrices n x n AND n is multiple of 2
///     This one will do calculations from 1 to 512 and print the results

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " n (Dimension of Matrix) s (seed)" << std::endl;
        return 1;
    }
    const int seed = std::atoi(argv[1]);
    int INFO;
    int max_size = 10;
    double *results_me = (double *)malloc( max_size * 7 * sizeof(double));
    double *results_lapack = (double *)malloc( max_size * 7 * sizeof(double));
            
    int n = 1;
    for (int i = 0; i < max_size; i++)
    {
        n *= 2;
        cout << "Matrix Size:" << n;

        // Create a random number generator =>  Get a Seed from random device
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // Alloc Space for MATRICES Needed in Column Major Order
        double *matrixA = (double *)malloc(n * n * sizeof(double));
        double *matrixB = (double *)malloc(n * n * sizeof(double));
        double *matrixA_original = (double *)malloc(n * n * sizeof(double)); // in case they get overwritten
        double *matrixB_original = (double *)malloc(n * n * sizeof(double)); // in case they get overwritten

        // Other Variables
        int *IPIV = (int *)malloc(n * sizeof(int));
        int *IPIVmine = (int *)malloc(n * sizeof(int));

        // Create the matrices A and B and fill it with random values
        for (int k = 0; k < n * n; k++)
        {
            matrixA[k] = dist(rng);
            matrixB[k] = dist(rng);
        }

        // Backup A and B Matrices
        Write_A_over_B(matrixA, matrixA_original, n, n);
        Write_A_over_B(matrixB, matrixB_original, n, n);

        if (i < 11)
        {
            // Alloc Space for MATRICES Needed in Column Major Order
            double *matrixBPivot = (double *)malloc(n * n * sizeof(double));
            double *matrixL = (double *)malloc(n * n * sizeof(double));
            double *matrixU = (double *)malloc(n * n * sizeof(double));
            double *matrixP = (double *)malloc(n * n * sizeof(double)); // Permutation Matrix
            double *matrixY = (double *)malloc(n * n * sizeof(double));
            double *matrixX = (double *)malloc(n * n * sizeof(double));
            // Backup A and B Matrices
            Write_A_over_B(matrixA, matrixA_original, n, n);
            Write_A_over_B(matrixB, matrixB_original, n, n);

            // ----------------- Start PIVOTED Algorithm HERE!
            LUdecompositionRecursive4Pivot(matrixA, matrixL, matrixU, IPIVmine, n, n);
            ipiv_to_P(IPIVmine, n, matrixP);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, matrixP, n, matrixB, n, 0.0, matrixBPivot, n);
            LowerTriangularSolverRecursiveReal_0(matrixL, matrixBPivot, matrixY, n, n);
            UpperTriangularSolverRecursiveReal_0(matrixU, matrixY, matrixX, n, n);

            cout << "\nCheck Accuracy and time of my AX=B (My Pivoted Recursive Algorithm):";
            results_me[7*i+6]=i;
            ErrorCalc_Display_v2(i,matrixA_original, matrixB_original, matrixX,results_me, n, n);

            // Restore A and B Matrices After Calculation
            Write_A_over_B(matrixA_original, matrixA, n, n);
            Write_A_over_B(matrixB_original, matrixB, n, n);

            free(matrixX);
            free(matrixY);
            free(matrixL);
            free(matrixU);
            free(matrixP);
            free(matrixBPivot);
        }
        //  ----------------- Solve BLAS and compare with my implementation HERE!
        LAPACK_dgesv(&n, &n, matrixA, &n, IPIV, matrixB, &n, &INFO);

        cout << "\nCheck Accuracy and time of LAPACK (dgesv): ";
        results_lapack[7*i+6]=i;
        ErrorCalc_Display_v2(i,matrixA_original, matrixB_original, matrixB, results_lapack, n, n);
        cout << "\n";

        // free memory
        free(matrixA);
        free(matrixB);
        free(matrixA_original);
        free(matrixB_original);
        free(IPIV);
        free(IPIVmine);
    }

    cout << "FINAL RESULTS:\n";
    cout << "Mine:\n";
    for (int k = 0; k < max_size; k++)
    {
        for (int kl = 0; kl < 7; kl++)
        {
            cout << results_me[k*7+kl] << ",";
        }
        cout << "\n";
    }
    cout << "\nLAPACK:\t";
    for (int k = 0; k < max_size; k++)
    {
        for (int kl = 0; kl < 7; kl++)
        {
            cout << results_lapack[k*7+kl] << ",";
        }
        cout << "\n";
    }  
    cout << "\n\n";
    return 0;
}