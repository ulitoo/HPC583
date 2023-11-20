#include <cblas.h>
#include <lapacke.h>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

// Problem , get a recursive algorithm for a Triangular solver that does the last step after LU factorization
// Ax=b -> LUx=b -> Ux=y // Ly=b //-> solve y from Ly=b and then solve x from Ux=y
// Case 1) left lower // Case 2) left upper
// then calculate distance (error from Ax to b)
// Compare Timing with dgesv (double real)/ zgesv (complex)
// SCOPE for REAL (double) First and leave COMPLEX case for next iteration

int recursion_count = 0;

/////////////////////////////     FUNCTIONS

void PrintRowMatrix(double *matrix, int m, int n)
{
    for (int i = 0; i < m * n; i++)
    {
        cout << matrix[i] << " ";
        if (i % n == n - 1)
        {
            cout << "\n";
        }
    }
}

void PrintColMatrix(double *matrix, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << matrix[i + (j * m)] << " ";
        }
        cout << "\n";
    }
}

void NaiveMatrixMultiplyCol(double *matrixa, double *matrixb, double *matrixc, int m, int n, int p)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                matrixc[i + (j * m)] += matrixa[i + (k * m)] * matrixb[k + (j * n)];
            }
        }
    }
}

double MatrixAbsDiff(double *matrixa, double *matrixb, int m, int p)
{
    double diff = 0.0;
    for (int i = 0; i < m * p; ++i)
    {
        diff += abs(matrixa[i] - matrixb[i]);
    }
    return diff;
}

void ColMajor_Transpose(double *matrix, int m, int n)
{
    double *tmpmatrix = (double *)malloc(m * n * sizeof(double));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            tmpmatrix[j + (i * n)] = matrix[i + (j * m)];
        }
    }
    for (int i = 0; i < m * n; i++)
    {
        matrix[i] = tmpmatrix[i];
    }
    free(tmpmatrix);
}

void MakeZeroes(double *matrix, int m, int n)
{
    for (int i = 0; i < m * n; i++)
    {
        matrix[i] = 0.0;
    }
}

// Collect will create the big matrix based on submatrices
void CollectSubmatrices(double *matrixc, double *C11, double *C12, double *C21, double *C22, int m, int p)
{
    int mm = m / 2;
    int mm2 = m - mm;
    int pp = p / 2;
    for (int i = 0; i < mm; ++i)
    {
        for (int j = 0; j < pp; ++j)
        {
            matrixc[i + (j * m)] += C11[i + (j * mm)];
        }
    }
    for (int i = 0; i < mm; ++i)
    {
        for (int j = pp; j < p; ++j)
        {
            matrixc[i + (j * m)] += C12[i + ((j - pp) * mm)];
        }
    }
    for (int i = mm; i < m; ++i)
    {
        for (int j = 0; j < pp; ++j)
        {
            matrixc[i + (j * m)] += C21[(i - mm) + (j * mm2)];
        }
    }
    for (int i = mm; i < m; ++i)
    {
        for (int j = pp; j < p; ++j)
        {
            matrixc[i + (j * m)] += C22[(i - mm) + ((j - pp) * mm2)];
        }
    }
}

// Initialize will Create the submatrices based on the big matrix
void InitializeSubmatrices(double *matrixc, double *C11, double *C12, double *C21, double *C22, int m, int p)
{
    int mm = m / 2;
    int mm2 = m - mm;
    int pp = p / 2;
    for (int i = 0; i < mm; ++i)
    {
        for (int j = 0; j < pp; ++j)
        {
            C11[i + (j * mm)] = matrixc[i + (j * m)];
        }
    }
    for (int i = 0; i < mm; ++i)
    {
        for (int j = pp; j < p; ++j)
        {
            C12[i + ((j - pp) * mm)] = matrixc[i + (j * m)];
        }
    }
    for (int i = mm; i < m; ++i)
    {
        for (int j = 0; j < pp; ++j)
        {
            C21[(i - mm) + (j * mm2)] = matrixc[i + (j * m)];
        }
    }
    for (int i = mm; i < m; ++i)
    {
        for (int j = pp; j < p; ++j)
        {
            C22[(i - mm) + ((j - pp) * mm2)] = matrixc[i + (j * m)];
        }
    }
}

void MMMultRecursive(double *matrixa, double *matrixb, double *matrixc, int m, int n, int p, int recursion_limit)
{
    recursion_count++;
    // if any of those is n=1 STOP??  Where to break recurse????
    // if n intermediate is small enough? if n==1? Break recurse??
    // not really!!! if m or n or p is 1 you HAVE to break recurse.
    // Then ->  do calculations by hand C = C + A*B
    // else COMPLETE RECURSE BLOCK

    if (m <= recursion_limit or n <= recursion_limit or p <= recursion_limit)
    {
        NaiveMatrixMultiplyCol(matrixa, matrixb, matrixc, m, n, p);
    }
    else
    {
        int mm = m / 2;
        int mm2 = m - mm;
        int nn = n / 2;
        int nn2 = n - nn;
        int pp = p / 2;
        int pp2 = p - pp;

        double *A11 = (double *)malloc(mm * nn * sizeof(double));
        MakeZeroes(A11, mm, nn);
        double *A12 = (double *)malloc(mm * nn2 * sizeof(double));
        MakeZeroes(A12, mm, nn2);
        double *A21 = (double *)malloc(mm2 * nn * sizeof(double));
        MakeZeroes(A21, mm2, nn);
        double *A22 = (double *)malloc(mm2 * nn2 * sizeof(double));
        MakeZeroes(A22, mm2, nn2);
        double *B11 = (double *)malloc(nn * pp * sizeof(double));
        MakeZeroes(B11, nn, pp);
        double *B12 = (double *)malloc(nn * pp2 * sizeof(double));
        MakeZeroes(B12, nn, pp2);
        double *B21 = (double *)malloc(nn2 * pp * sizeof(double));
        MakeZeroes(B21, nn2, pp);
        double *B22 = (double *)malloc(nn2 * pp2 * sizeof(double));
        MakeZeroes(B22, nn2, pp2);
        double *C11 = (double *)malloc(mm * pp * sizeof(double));
        MakeZeroes(C11, mm, pp);
        double *C12 = (double *)malloc(mm * pp2 * sizeof(double));
        MakeZeroes(C12, mm, pp2);
        double *C21 = (double *)malloc(mm2 * pp * sizeof(double));
        MakeZeroes(C21, mm2, pp);
        double *C22 = (double *)malloc(mm2 * pp2 * sizeof(double));
        MakeZeroes(C22, mm2, pp2);

        // Initializa Axx and Bxx matrices!
        InitializeSubmatrices(matrixa, A11, A12, A21, A22, m, n);
        InitializeSubmatrices(matrixb, B11, B12, B21, B22, n, p);

        // C11 is recurse1 and recurse2
        MMMultRecursive(A11, B11, C11, mm, nn, pp, recursion_limit);
        MMMultRecursive(A12, B21, C11, mm, nn2, pp, recursion_limit);

        // C12 is recurse3 and recurse4
        MMMultRecursive(A11, B12, C12, mm, nn, pp2, recursion_limit);
        MMMultRecursive(A12, B22, C12, mm, nn2, pp2, recursion_limit);
        // C21 is recurse5 and recurse6
        MMMultRecursive(A21, B11, C21, mm2, nn, pp, recursion_limit);
        MMMultRecursive(A22, B21, C21, mm2, nn2, pp, recursion_limit);
        // C22 is recurse7 and recurse8
        MMMultRecursive(A21, B12, C22, mm2, nn, pp2, recursion_limit);
        MMMultRecursive(A22, B22, C22, mm2, nn2, pp2, recursion_limit);

        // At the end Collect pieces of matrixc = matrixc + C11 + C12 + C21 + C22 and done!
        CollectSubmatrices(matrixc, C11, C12, C21, C22, m, p);

        free(A11);
        free(A12);
        free(A21);
        free(A22);
        free(B11);
        free(B12);
        free(B21);
        free(B22);
        free(C11);
        free(C12);
        free(C21);
        free(C22);
    }
}

void InitVector(double *vectorB, int n)
{
    for (int i = 0; i < n; i++)
    {
        vectorB[i] = (double)((n - i + 1) * (n - i + 1));
    }
}

void LowerTriangularSolverNaiveReal(double *matrixL, double *matrixB, double *matrixSol, int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            double tempval = 0;
            for (int k = 0; k < i; k++)
            {
                tempval += matrixL[i + k * n] * matrixSol[k + j * n];
            }
            matrixSol[j * n + i] = (matrixB[j * n + i] - tempval) / matrixL[i * n + i];
        }
    }
}

void UpperTriangularSolverNaiveReal(double *matrixU, double *matrixB, double *matrixSol, int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = n - 1; i >= 0; i--)
        {
            double tempval = 0;
            for (int k = n - 1; k > i; k--)
            {
                tempval += matrixU[i + k * n] * matrixSol[k + j * n];
            }
            matrixSol[j * n + i] = (matrixB[j * n + i] - tempval) / matrixU[i * n + i];
        }
    }
}

void LowerTriangularSolverRecursiveReal(double *matrixL, double *matrixB, double *matrixSol, int n)
{
    // PHASE 1: RECURSE on calculations with TRIANGULAR A11
    
    
    
    
    // PHASE 2: CALCULATE THE NEW B's for next Phase 
    // PHASE 3: RECURSE on REST of calculations with TRIANGULAR A22
}

void UpperTriangularSolverRecursiveReal(double *matrixU, double *matrixB, double *matrixSol, int n)
{
}

/////////////////////////////     MAIN
int main(int argc, char *argv[])
{

    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " <filenameL> <filenameU> <filenameB> n recursion_limit" << std::endl;
        return 1;
    }

    const int n = std::atoi(argv[4]);               // n
    const int recursion_limit = std::atoi(argv[5]); // depth of recursion

    // Input Matrices in Column Major Format
    double *matrixL = (double *)malloc(n * n * sizeof(double));
    double *matrixU = (double *)malloc(n * n * sizeof(double));
    double *matrixB = (double *)malloc(n * n * sizeof(double));
    // Solution Matrices
    double *LowerMatrixXsolNaive = (double *)malloc(n * n * sizeof(double));
    double *LowerMatrixXsol = (double *)malloc(n * n * sizeof(double));
    double *UpperMatrixXsolNaive = (double *)malloc(n * n * sizeof(double));
    double *UpperMatrixXsol = (double *)malloc(n * n * sizeof(double));
    // Matrices to Calculate Error
    double *LowerCalculatedBNaive = (double *)malloc(n * n * sizeof(double));
    double *UpperCalculatedBNaive = (double *)malloc(n * n * sizeof(double));
    double *LowerCalculatedB = (double *)malloc(n * n * sizeof(double));
    double *UpperCalculatedB = (double *)malloc(n * n * sizeof(double));

    // Open the binary file for reading INPUT Matrices and handle error
    std::ifstream inputA(argv[1], std::ios::binary);
    std::ifstream inputB(argv[2], std::ios::binary);
    std::ifstream inputC(argv[3], std::ios::binary);
    if (!inputA or !inputB or !inputC)
    {
        std::cerr << "Error: could not open file for reading" << std::endl;
        return 1;
    }
    // Read the binary data into the vector
    inputA.read(reinterpret_cast<char *>(matrixL), sizeof(double) * n * n);
    inputB.read(reinterpret_cast<char *>(matrixU), sizeof(double) * n * n);
    inputC.read(reinterpret_cast<char *>(matrixB), sizeof(double) * n * n);
    // Check if read was successful and handle error
    if (!inputA or !inputB or !inputC)
    {
        std::cerr << "Error: could not read file" << std::endl;
        return 1;
    }

    // Print the matrix elements
    cout << "\nMatrix L Print:\n";
    PrintColMatrix(matrixL, n, n);
    cout << "\nMatrix U Print:\n";
    PrintColMatrix(matrixU, n, n);
    cout << "\nMatrix B Print:\n";
    PrintColMatrix(matrixB, n, n);

    // Solve Naive
    LowerTriangularSolverNaiveReal(matrixL, matrixB, LowerMatrixXsolNaive, n);
    UpperTriangularSolverNaiveReal(matrixU, matrixB, UpperMatrixXsolNaive, n);

    // Solve Recursive
    LowerTriangularSolverRecursiveReal(matrixL, matrixB, LowerMatrixXsol, n);
    UpperTriangularSolverRecursiveReal(matrixL, matrixB, UpperMatrixXsol, n);

    // PRINT NAIVE Solutions
    // cout << "\nLower Matrix X Solution Naive:\n";
    // PrintColMatrix(LowerMatrixXsolNaive,n,n);
    // cout << "\nUpper Matrix X Solution Naive:\n";
    // PrintColMatrix(UpperMatrixXsolNaive,n,n);

    // ERROR CALCULATION and display
    // EXAMPLE with DGEMM : cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, matrixL, n, LowerMatrixXsolNaive, n, 0.0, LowerCalculatedBNaive, n);
    NaiveMatrixMultiplyCol(matrixL, LowerMatrixXsolNaive, LowerCalculatedBNaive, n, n, n);
    NaiveMatrixMultiplyCol(matrixU, UpperMatrixXsolNaive, UpperCalculatedBNaive, n, n, n);
    NaiveMatrixMultiplyCol(matrixL, LowerMatrixXsol, LowerCalculatedB, n, n, n);
    NaiveMatrixMultiplyCol(matrixU, UpperMatrixXsol, UpperCalculatedB, n, n, n);

    double Ldiffnaive = MatrixAbsDiff(matrixB, LowerCalculatedBNaive, n, n);
    double Udiffnaive = MatrixAbsDiff(matrixB, UpperCalculatedBNaive, n, n);
    double Ldiff = MatrixAbsDiff(matrixB, LowerCalculatedB, n, n);
    double Udiff = MatrixAbsDiff(matrixB, UpperCalculatedB, n, n);

    cout << "\nNaive Error (LX - B)-----------------> : " << Ldiffnaive << "\n";
    cout << "Naive Error (UX - B)-----------------> : " << Udiffnaive << "\n";
    cout << "\nRecursive Error (LX - B)-----------------> : " << Ldiff << "\n";
    cout << "Recursive Error (UX - B)-----------------> : " << Udiff << "\n";
    cout << "\nRecursion Count -----------------> : " << recursion_count << "\n";

    //    LAPACK_zgesv();
    //    LAPACK_dgesv();

    free(matrixL);
    free(matrixU);
    free(matrixB);
    free(LowerMatrixXsolNaive);
    free(LowerMatrixXsol);
    free(UpperMatrixXsolNaive);
    free(UpperMatrixXsol);
    free(LowerCalculatedBNaive);
    free(UpperCalculatedBNaive);
    free(LowerCalculatedB);
    free(UpperCalculatedB);
    return 0;
}
