#include <cblas.h>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>
#include <pthread.h>
#include <thread>
#include <lapacke.h>

using namespace std;

// 	Problem:
//  1 One Single Clean Code
//  2 Get Matrix
//  3 Find condition number for the Matrix
//  4 Run 3 routines with timing and Matrix Distance
//      -With Pivot
//      -With NO Pivot
//      -LAPACK BLAS
//  5. OPTIMIZED PIVOT CODE LUdecompositionRecursive4Pivot

/////////////////////////////   General Purpose  FUNCTIONS

int Read_Matrix_file(double *matrix, int size, char *filename)
{
    // Open the binary file for reading and handle error
    std::ifstream input(filename, std::ios::binary);
    if (!input)
    {
        std::cerr << "Error: could not open file for reading" << std::endl;
        return 1;
    }
    // Read the binary data into the vector
    input.read(reinterpret_cast<char *>(matrix), sizeof(double) * size);
    // Check if read was successful and handle error
    if (!input)
    {
        std::cerr << "Error: could not read file" << std::endl;
        return 1;
    }
    std::cout << "File " << filename << " read correctly!" << std::endl;
    return 0;
}
void Write_A_over_B(double *matrixA, double *matrixB, int m, int n)
{
    for (int i = 0; i < m * n; i++)
    {
        matrixB[i] = matrixA[i];
    }
}
void InverseMatrix(double *matrixA, int n)
{
    int ipiv[n];
    int info;
    // Compute the inverse using LAPACK's dgetrf and dgetri
    // Perform LU factorization
    info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, matrixA, n, ipiv);

    if (info == 0)
    {
        // LU factorization succeeded, now compute the inverse
        info = LAPACKE_dgetri(LAPACK_COL_MAJOR, n, matrixA, n, ipiv);

        if (info != 0)
        {
            std::cerr << "Error in LAPACKE_dgetri: " << info << std::endl;
        }
    }
    else
    {
        std::cerr << "Error in LAPACKE_dgetrf: " << info << std::endl;
    }
}
double InfinityNorm(double *matrixA, int n)
{
    // Find the biggest sum of abs (rows)
    double max = 0.0;
    double tmp = 0.0;
    for (int i = 0; i < n; i++)
    {
        tmp = 0.0;
        for (int j = 0; j < n; j++)
        {
            tmp += abs(matrixA[i + (j * n)]);
        }
        if (tmp > max)
        {
            max = tmp;
        }
    }
    return max;
}
double ConditionNumber(double *matrixA, int m, int n)
{
    //  Find condition number for the Matrix /Norm of matrix/ Infinity norm (max row or col)
    //  The infinity-norm of a square matrix is the maximum of the absolute row sum
    //  Condition number is the ||M|| times ||M^(-1)||, the closer to 1 the more stable
    double *matrixA_original = (double *)malloc(n * n * sizeof(double));
    Write_A_over_B(matrixA, matrixA_original, n, n);
    InverseMatrix(matrixA, n);

    double InfNormA, InfNormAinv;
    InfNormA = InfinityNorm(matrixA, n);
    InfNormAinv = InfinityNorm(matrixA_original, n);

    // restore original Matrix
    Write_A_over_B(matrixA_original, matrixA, n, n);
    free(matrixA_original);
    return InfNormA * InfNormAinv;
}
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
void MakeZeroes(double *matrix, int m, int n)
{
    for (int i = 0; i < m * n; i++)
    {
        matrix[i] = 0.0;
    }
}
void MakeIdentity(double *matrix, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                matrix[i + j * m] = 1.0;
            }
            else
            {
                matrix[i + j * m] = 0.0;
            }
        }
    }
}
int MaxRow(double *matrix, int n)
{
    int maxrow;
    double maxabsvalue = 0;
    double temp_abs_element;

    for (int i = 0; i < n; i++)
    {
        temp_abs_element = abs(matrix[i]);
        if (temp_abs_element > maxabsvalue)
        {
            maxrow = i;
            maxabsvalue = temp_abs_element;
        }
    }
    return maxrow;
}
int MaxRow2(double *matrix, int N, int n)
{
    int offset = N - n;
    int maxrow = offset;
    double maxabsvalue = 0.0;
    double temp_abs_element;

    for (int i = offset; i < N; i++)
    {
        temp_abs_element = abs(matrix[i + offset * N]);
        if (temp_abs_element > maxabsvalue)
        {
            maxrow = i;
            maxabsvalue = temp_abs_element;
        }
    }
    return maxrow;
}
double MatrixDistance(double *matrixa, double *matrixb, int m, int n)
{
    double diff = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
        diff += (matrixa[i] - matrixb[i]) * (matrixa[i] - matrixb[i]);
    }
    return sqrt(diff);
}
void SwapRow_ColMajMatrix(double *matrix, int from, int towards, int m, int n)
{
    double tmpval;
    for (int i = 0; i < n * m; i += m)
    {
        tmpval = matrix[towards + i];
        matrix[towards + i] = matrix[from + i];
        matrix[from + i] = tmpval;
    }
}

/////////////////////////////   Triangular Solver FUNCTIONS
void InitializeSubmatrices(double *matrixc, double *C11, double *C12, double *C21, double *C22, int m, int p)
{
    // Initialize will Create the submatrices based on the big matrix
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
void CollectSubmatrices(double *matrixc, double *C11, double *C12, double *C21, double *C22, int m, int p)
{
    // Collect Results of Xxx to the big matrix X
    MakeZeroes(matrixc, m, p);
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

void UpperTriangularSolverRecursiveReal_0(double *matrixU, double *matrixB, double *matrixX, int n, int p)
{
    // This is a Naive version with Malloc and free as crutch to avoid index calculation over the original matrix
    if (n == 1)
    {
            matrixX[0] = matrixB[0] / matrixU[0];
    }
    else
    {
        int nn = n / 2;
        int nn2 = n - nn;
        int pp = p / 2;
        int pp2 = p - pp;

        double *U11 = (double *)malloc(nn * nn * sizeof(double));
        double *U12 = (double *)malloc(nn * nn2 * sizeof(double));
        double *U21 = (double *)malloc(nn2 * nn * sizeof(double));
        double *U22 = (double *)malloc(nn2 * nn2 * sizeof(double));
        double *B11 = (double *)malloc(nn * pp * sizeof(double));
        double *B12 = (double *)malloc(nn * pp2 * sizeof(double));
        double *B21 = (double *)malloc(nn2 * pp * sizeof(double));
        double *B22 = (double *)malloc(nn2 * pp2 * sizeof(double));
        double *X11 = (double *)malloc(nn * pp * sizeof(double));
        double *X12 = (double *)malloc(nn * pp2 * sizeof(double));
        double *X21 = (double *)malloc(nn2 * pp * sizeof(double));
        double *X22 = (double *)malloc(nn2 * pp2 * sizeof(double));

        // Initializa Axx and Bxx matrices!
        InitializeSubmatrices(matrixU, U11, U12, U21, U22, n, n);
        InitializeSubmatrices(matrixB, B11, B12, B21, B22, n, p);

        // Recurse U22 X21 = B21
        UpperTriangularSolverRecursiveReal_0(U22, B21, X21, nn2, pp);
        // Recurse U22 X22 = B22
        UpperTriangularSolverRecursiveReal_0(U22, B22, X22, nn2, pp2);

        // PHASE 2: CALCULATE THE NEW B's for next Phase
        // B11' = B11 - U12 X21
        // B12' = B12 - U12 X22
        for (int i = 0; i < nn; ++i)
        {
            for (int j = 0; j < pp; ++j)
            {
                for (int k = 0; k < nn2; ++k)
                {
                    B11[i + ((j)*nn)] -= (U12[i + (k)*nn]) * (X21[k + (j)*nn2]);
                    B12[i + ((j)*nn)] -= (U12[i + (k)*nn]) * (X22[k + (j)*nn2]);
                }
            }
        }
          
        // PHASE 3: RECURSE on REST of calculations with TRIANGULAR A22
        // Recurse U11 X11 = B11'
        UpperTriangularSolverRecursiveReal_0(U11, B11, X11, nn, pp);
        // Recurse U11 X12 = B12'
        UpperTriangularSolverRecursiveReal_0(U11, B12, X12, nn, pp2);

        // At the end Collect pieces of matrixc = matrixc + C11 + C12 + C21 + C22 and done!
        CollectSubmatrices(matrixX, X11, X12, X21, X22, n, p);

        free(U11);
        free(U12);
        free(U21);
        free(U22);
        free(B11);
        free(B12);
        free(B21);
        free(B22);
        free(X11);
        free(X12);
        free(X21);
        free(X22);
    }
}
void LowerTriangularSolverRecursiveReal_0(double *matrixL, double *matrixB, double *matrixX, int n, int p)
{
    // PHASE 1: RECURSE on calculations based on TRIANGULAR L11
    if (n == 1)
    {
        matrixX[0] = matrixB[0] / matrixL[0];
    }
    else
    {
        int nn = n / 2;
        int nn2 = n - nn; // Size of right or lower side covers for odd cases
        int pp = (p / 2);
        int pp2 = p - pp;

        double *L11 = (double *)malloc(nn * nn * sizeof(double));
        double *L12 = (double *)malloc(nn * nn2 * sizeof(double));
        double *L21 = (double *)malloc(nn2 * nn * sizeof(double));
        double *L22 = (double *)malloc(nn2 * nn2 * sizeof(double));
        double *B11 = (double *)malloc(nn * pp * sizeof(double));
        double *B12 = (double *)malloc(nn * pp2 * sizeof(double));
        double *B21 = (double *)malloc(nn2 * pp * sizeof(double));
        double *B22 = (double *)malloc(nn2 * pp2 * sizeof(double));
        double *X11 = (double *)malloc(nn * pp * sizeof(double));
        double *X12 = (double *)malloc(nn * pp2 * sizeof(double));
        double *X21 = (double *)malloc(nn2 * pp * sizeof(double));
        double *X22 = (double *)malloc(nn2 * pp2 * sizeof(double));

        // Initialize Axx and Bxx matrices!
        InitializeSubmatrices(matrixL, L11, L12, L21, L22, n, n);
        InitializeSubmatrices(matrixB, B11, B12, B21, B22, n, p);

        // Recurse L11 X11 = B11
        LowerTriangularSolverRecursiveReal_0(L11, B11, X11, nn, pp);

        // Recurse L11 X12 = B12
        LowerTriangularSolverRecursiveReal_0(L11, B12, X12, nn, pp2);

        // PHASE 2: CALCULATE THE NEW B's for next Phase
        // B21' = B21 - L21 X11
        // B22' = B22 - L21 X12
        for (int i = 0; i < nn2; ++i)
        {
            for (int j = 0; j < pp; ++j)
            {
                for (int k = 0; k < nn; ++k)
                {
                    B21[i + (j * nn2)] -= (L21[i + (k * nn2)]) * (X11[k + (j)*nn]);
                    B22[i + (j * nn2)] -= (L21[i + (k * nn2)]) * (X12[k + (j)*nn]);
                    // Assumes even matrix (if not, you cannot mix these 2 and need j<pp2 )
                }
            }
        }

        // PHASE 3: RECURSE on REST of calculations with TRIANGULAR A22
        // Recurse L22 X21 = B21'
        LowerTriangularSolverRecursiveReal_0(L22, B21, X21, nn2, pp);
        // Recurse L22 X22 = B22'
        LowerTriangularSolverRecursiveReal_0(L22, B22, X22, nn2, pp2);

        // At the end Collect pieces of matrixc = matrixc + C11 + C12 + C21 + C22 and done!
        CollectSubmatrices(matrixX, X11, X12, X21, X22, n, p);

        free(L11);
        free(L12);
        free(L21);
        free(L22);
        free(B11);
        free(B12);
        free(B21);
        free(B22);
        free(X11);
        free(X12);
        free(X21);
        free(X22);
    }
}

void LowerTriangularSolverRecursiveReal_1even(double *BIGL, int lm, int ln, double *BIGB, int bm, int bn, double *BIGX,int xm, int xn, int n, int N)
{
    // PHASE 1: RECURSE on calculations based on TRIANGULAR L11
    if (n == 1)
    {
        BIGX[xm+N*xn] = BIGB[bm+N*bn] / BIGL[lm+N*ln];
    }
    else
    {
        int nn = n / 2;

        // Recurse L11 X11 = B11
        LowerTriangularSolverRecursiveReal_1even(BIGL,lm+0,ln+0,BIGB,bm+0,bn+0, BIGX,xm+0,xn+0, nn, N);

        // Recurse L11 X12 = B12
        LowerTriangularSolverRecursiveReal_1even(BIGL,lm+0,ln+0,BIGB,bm+0,bn+nn, BIGX,xm+0,xn+nn, nn,N);

        // PHASE 2: CALCULATE THE NEW B's for next Phase
        // B21' = B21 - L21 X11
        // B22' = B22 - L21 X12

        for (int i = 0; i < nn; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                for (int k = 0; k < nn; ++k)
                {
                    BIGB[i + bm + nn + ((j + bn) * N)] -= (BIGL[i + lm + nn + ( k + ln ) * N]) * (BIGX[k + xm + ((j + xn) * N)]);
                }
            }
        }

        // Recurse L22 X21 = B21'
        LowerTriangularSolverRecursiveReal_1even(BIGL,lm+nn,ln+nn,BIGB,bm+nn,bn+0, BIGX,xm+nn,xn+0, nn,N);

        // Recurse L22 X22 = B22'
        LowerTriangularSolverRecursiveReal_1even(BIGL,lm+nn,ln+nn,BIGB,bm+nn,bn+nn, BIGX,xm+nn,xn+nn, nn,N);
    }
}
void UpperTriangularSolverRecursiveReal_1even(double *BIGU, int lm, int ln, double *BIGB, int bm, int bn, double *BIGX,int xm, int xn, int n, int N)
{
    // PHASE 1: RECURSE on calculations based on TRIANGULAR L11
    if (n == 1)
    {
        BIGX[xm+N*xn] = BIGB[bm+N*bn] / BIGU[lm+N*ln];
    }
    else
    {
        int nn = n / 2;

        // Recurse U22 X21 = B21
        UpperTriangularSolverRecursiveReal_1even(BIGU, lm + nn, ln + nn, BIGB, bm + nn, bn + 0, BIGX, xm + nn, xn + 0, nn, N);

        // Recurse U22 X22 = B22
        UpperTriangularSolverRecursiveReal_1even(BIGU, lm + nn, ln + nn, BIGB, bm + nn, bn + nn, BIGX, xm + nn, xn + nn, nn, N);

        // PHASE 2: CALCULATE THE NEW B's for next Phase
        // B11' = B11 - U12 X21
        // B12' = B12 - U12 X22

        int i3=xm+nn+N*xn;
        for (int i = 0; i < nn; ++i)
        {
            int i2=i+lm+(nn+ln)*N;
            int i4=i+bm+bn*N;
            for (int j = 0; j < n; ++j)
            {
                
                for (int k = 0; k < nn; ++k)
                {
                    BIGB[i4 + (j * N)] -= (BIGU[i2 + k * N]) * (BIGX[k + i3 + (j * N)]);
                }
            }
        }

        // Recurse U11 X11 = B11'
        UpperTriangularSolverRecursiveReal_1even(BIGU, lm + 0, ln + 0, BIGB, bm + 0, bn + 0, BIGX, xm + 0, xn + 0, nn, N);

        // Recurse U11 X12 = B12'
        UpperTriangularSolverRecursiveReal_1even(BIGU, lm + 0, ln + 0, BIGB, bm + 0, bn + nn, BIGX, xm + 0, xn + nn, nn, N);
    }
}

/////////////////////////////   LU Decomposition FUNCTIONS
void ipiv_to_P(int *ipiv, int n, double *P)
{
    MakeIdentity(P, n, n);

    for (int i = 0; i < n; i++)
    {
        SwapRow_ColMajMatrix(P, i, ipiv[i], n, n);
    }
}
void SchurComplement(double *matrix, int N, int n)
{
    int offset = (N - n);
    int offset2 = offset * (N + 1);
    // ONLY 1 matrix, rewrite A22 as S22 ! ; N is the original A size ; n is the size of A in the recursion; n-1 is size of A22
    for (int i = 0; i < (n - 1); i++)
    {
        int i2 = (i + offset2 + 1);
        for (int j = 0; j < (n - 1); j++)
        {
            // This is in Column Major Order
            int k = (j + 1) * N;
            matrix[i2 + k] = matrix[(i2) + (k)] - ((matrix[offset2 + k]) * (matrix[i + 1 + (offset2)]) / matrix[offset2]);
        }
    }
}
void SchurComplement2(double *matrix, int n)
{
    // ONLY 1 matrix, rewrite A22 as S22 ! ; N is the original A size ; n is the size of A in the recursion; n-1 is size of A22
    // Without offset concept just 1 rank lower
    for (int i = 1; i < (n); i++)
    {
        for (int j = 1; j < (n); j++)
        {
            // This is in Column Major Order
            matrix[(i) + (j * n)] = matrix[(i) + (j * n)] - ((matrix[j * n]) * (matrix[i])) / matrix[0];
        }
    }
}
void LUdecompositionRecursive2(double *matrix, double *Lmatrix, double *Umatrix, int N, int n)
{
    // Assume square Matrix for simplicity

    int offset = (N - n);
    int offset2 = N * offset;
    int offset3 = (offset) + offset2;
    int j, i2;
    Umatrix[(offset) * (N + 1)] = matrix[(offset) * (N + 1)];
    Lmatrix[(offset) * (N + 1)] = 1.0;

    for (int i = 1; i < n; i++)
    {
        j = i * N + offset3;
        i2 = i + offset3;
        // offset*N unnecesary repeated calculations!!!!!!!!!!!!!!
        Lmatrix[i2] = matrix[i2] / matrix[offset3];
        // Lmatrix[j] = 0.0;  Redundant, Already Zero
        Umatrix[j] = matrix[j];
        // Umatrix[i2] = 0.0; Redundant, Already Zero
    }

    if (n == 2)
    {
        Umatrix[(offset + 1) + (offset + 1) * N] = matrix[(offset + 1) + (offset + 1) * N] - matrix[(offset + 1) + (offset)*N] * matrix[(offset) + (offset + 1) * N] / matrix[(offset) + (offset)*N];
        Lmatrix[(offset + 1) + (offset + 1) * N] = 1.0;
    }
    else
    {
        SchurComplement(matrix, N, n);
        LUdecompositionRecursive2(matrix, Lmatrix, Umatrix, N, n - 1);
    }
}
void LUdecompositionRecursive4Pivot(double *AmatrixBIG, double *LmatrixBIG, double *UmatrixBIG, int *IPIV, int N, int n)
{
    //   WITHOUT MALLOC
    int offset = (N - n);
    int offset2 = N * offset;
    int offset3 = (offset) + offset2;
    int offset4 = (offset) * (N + 1);
    int k, i2;

    // Assume square Matrices for simplicity
    IPIV[offset] = MaxRow2(AmatrixBIG, N, n); // Tracks ROW Exchange
    // Permutation
    SwapRow_ColMajMatrix(AmatrixBIG, offset, IPIV[offset], N, N);
    // Amatrix is now Amatrixbar(permutation done)

    UmatrixBIG[(offset4)] = AmatrixBIG[(offset4)];
    LmatrixBIG[(offset4)] = 1.0;

    if (n == 2)
    {
        IPIV[N - 1] = N - 1; // Last Row does NOT change!

        UmatrixBIG[(offset4 + N + 1)] = AmatrixBIG[(offset4 + N + 1)] - AmatrixBIG[(offset4 + 1)] * AmatrixBIG[(offset4 + N)] / AmatrixBIG[(offset4)];
        LmatrixBIG[(offset4 + N + 1)] = 1.0;

        LmatrixBIG[(offset4 + 1)] = AmatrixBIG[(offset4 + 1)] / AmatrixBIG[offset4];
        UmatrixBIG[(offset4 + N)] = AmatrixBIG[(offset4 + N)];
    }
    else
    {
        SchurComplement(AmatrixBIG, N, n);
        LUdecompositionRecursive4Pivot(AmatrixBIG, LmatrixBIG, UmatrixBIG, IPIV, N, n - 1);

        for (int i = 1; i < n; i++)
        {
            k = i * N + offset3;
            i2 = i + offset3;
            UmatrixBIG[k] = AmatrixBIG[k];
            LmatrixBIG[i2] += AmatrixBIG[i2] / AmatrixBIG[(offset3)];
        }
    }
}

void ErrorCalc_Display(double *matrixA, double *matrixB, double *matrixX, long double elapsed_time, int n, int p)
{
    double *CalculatedB = (double *)malloc(n * p * sizeof(double));
    MakeZeroes(CalculatedB, n, p);
    // NaiveMatrixMultiplyCol(matrixA, matrixX, CalculatedB, n, n, p);
    // Substitute by LAPACK dGEMM
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, matrixA, n, matrixX, n, 0.0, CalculatedB, n);
    double dist = MatrixDistance(matrixB, CalculatedB, n, p);
    cout << "\nVector Distance - Error (AX - B):----------------> : " << dist << "\n";
    cout << "Elapsed Time:------------------------------------> : " << elapsed_time << " s.\n\n";
    free(CalculatedB);
}

///     MAIN :: For the sake of simplicity we will Consider all square matrices n x n

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <filename A> <filename B> rank " << std::endl;
        return 1;
    }
    const int n = std::atoi(argv[3]); // rank

    // Timers
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time_BLAS, elapsed_time_Pivot, elapsed_time_nonPivot, elapsed_time_Solve;

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
    double AConditionNumber;
    int INFO, IPIV[n], IPIVmine[n];

    // READ Matrix A and B from arguments and FILES
    Read_Matrix_file(matrixA, n * n, argv[1]);
    Read_Matrix_file(matrixB, n * n, argv[2]);

    // Backup A and B Matrices
    Write_A_over_B(matrixA, matrixA_original, n, n);
    Write_A_over_B(matrixB, matrixB_original, n, n);

    // ----------------- Calculate Condition Number of Matrix A

    AConditionNumber = ConditionNumber(matrixA, n, n);
    cout << "\nMatrix A Condition Number: " << (AConditionNumber) << "\n";

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
    // Solve (2) UX=Y
    UpperTriangularSolverRecursiveReal_0(matrixU, matrixY, matrixX, n, n);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time_Solve = duration.count() * 1.e-9;

    //Not intermediate Malloc Version
    //LowerTriangularSolverRecursiveReal_1even(matrixL,0,0,matrixBPivot,0,0, matrixY,0,0, n, n);
    //UpperTriangularSolverRecursiveReal_1even(matrixU,0,0,matrixY,0,0, matrixX,0,0, n, n);

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
