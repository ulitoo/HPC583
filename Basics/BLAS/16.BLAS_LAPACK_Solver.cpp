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
//  Play Around with BLAS and LAPACK

/////////////////////////////     FUNCTIONS
int recursion_count = 0;

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
    std::cout << "File " << filename << " read correctly!\n"
              << std::endl;
    return 0;
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
int MaxRow(double *matrix, int N, int n)
{
    int maxrow;
    double maxabsvalue = 0;

    int offset = (N - n);
    for (int i = offset; i < N; i++)
    {
        if (abs(matrix[i + offset * N]) > maxabsvalue)
        {
            maxrow = i;
            maxabsvalue = abs(matrix[i + offset * N]);
        }
    }
    return maxrow;
}
double MatrixAbsDiff(double *matrixa, double *matrixb, int m, int n)
{
    double diff = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
        diff += abs(matrixa[i] - matrixb[i]);
    }
    return diff;
}
double MatrixDistance(double *matrixa, double *matrixb, int m, int n)
{
    double diff = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
        diff += (matrixa[i] - matrixb[i])*(matrixa[i] - matrixb[i]);
    }
    return sqrt(diff);
}
void Write_A_over_B(double *matrixA, double *matrixB, int m, int n)
{
    for (int i = 0; i < m * n; i++)
    {
        matrixB[i] = matrixA[i];
    }
}
void SwapCol_ColMajMatrix(double *matrix, int from, int towards, int m, int n)
{
    double tmpval;
    for (int i = 0; i < m; i++)
    {
        tmpval = matrix[i + towards * m];
        matrix[i + towards * m] = matrix[i + from * m];
        matrix[i + from * m] = tmpval;
    }
}
void SwapRow_ColMajMatrix(double *matrix, int from, int towards, int m, int n)
{
    double tmpval;
    for (int i = 0; i < n; i++)
    {
        tmpval = matrix[towards + i * m];
        matrix[towards + i * m] = matrix[from + i * m];
        matrix[from + i * m] = tmpval;
    }
}
void TransposeColMajor(double *matrix, int m, int n)
{
    double tmpvalue;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < i; j++)
        {
            tmpvalue = matrix[j + (i * m)];
            matrix[j + (i * m)] = matrix[i + (j * m)];
            matrix[i + (j * m)] = tmpvalue;
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
    double diff = MatrixAbsDiff(matrixB, CalculatedB, n, p);
    double dist = MatrixDistance(matrixB, CalculatedB, n, p);
    cout << "\nAbs cumulative Error (AX - B):----------------> : " << diff << "\n";
    cout << "Vector Distance (AX - B):---------------------> : " << dist << "\n";
    cout << "Elapsed Time:------------------> : " << elapsed_time << " s.\n\n";
    free(CalculatedB);
}
void From_A_to_PLU(double *A, int *ipiv, int n, double *L, double *U, double *P)
{
    int tmpPinv[n];

    for (int i = 0; i < n; i++)
    {
        tmpPinv[i] = i;
    }
    for (int i = 0; i < n; i++)
    {
        int tmp;
        // swap (tmpPinv [ipiv [i]], tmpPinv[i] ) and it is off by one
        tmp = tmpPinv[ipiv[i] - 1];
        tmpPinv[ipiv[i] - 1] = tmpPinv[i];
        tmpPinv[i] = tmp;
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i > j)
            {
                L[i + n * j] = A[i + n * j]; // Lower triangular part
            }
            else
            {
                U[i + n * j] = A[i + n * j]; // Upper triangular part
                if (i == j)
                {
                    L[i + n * j] = 1.0; // Diagonal elements of L are 1
                    P[(tmpPinv[j]) + n * j] = 1.0;
                }
            }
        }
    }
}

///     MAIN :: For the sake of simplicity we will Consider all square matrices n x n

int main(int argc, char *argv[])
{

    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <filename A> <filename B> rank" << std::endl;
        return 1;
    }
    const int n = std::atoi(argv[3]);          // rank

    // Timers
    auto start = std::chrono::high_resolution_clock::now();
    auto start1 = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto stop1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double Speedup;

    // Alloc Space for Matrices Needed in Column Major Order
    double *matrixA = (double *)malloc(n * n * sizeof(double));
    double *matrixB = (double *)malloc(n * n * sizeof(double));
    double *matrixBPivot = (double *)malloc(n * n * sizeof(double));
    double *matrixL = (double *)malloc(n * n * sizeof(double));
    double *matrixU = (double *)malloc(n * n * sizeof(double));
    double *matrixLU = (double *)malloc(n * n * sizeof(double));
    double *matrixP = (double *)malloc(n * n * sizeof(double)); // Permutation Matrix
    double *matrixY = (double *)malloc(n * n * sizeof(double));
    double *matrixX = (double *)malloc(n * n * sizeof(double));
    double *matrixB_Calc = (double *)malloc(n * n * sizeof(double));
    double *matrixA_original = (double *)malloc(n * n * sizeof(double)); // in case they get overwritten
    double *matrixB_original = (double *)malloc(n * n * sizeof(double)); // in case they get overwritten

    // Read Matrix A and B from arguments and files
    Read_Matrix_file(matrixA, n * n, argv[1]);
    Read_Matrix_file(matrixB, n * n, argv[2]);

    // Backup
    Write_A_over_B(matrixA, matrixA_original, n, n);
    Write_A_over_B(matrixB, matrixB_original, n, n);

    // Solve BLAS complete 100%
    int INFO;
    int IPIV[n];
    start = std::chrono::high_resolution_clock::now();
    LAPACK_dgesv(&n, &n, matrixA, &n, IPIV, matrixB, &n, &INFO);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    cout << "\nCheck Accuracy and time of Complete BLAS Solver (dgesv): ";
    ErrorCalc_Display(matrixA_original, matrixB_original, matrixB, duration.count() * 1.e-9, n, n);
    
    // Restore A&B
    Write_A_over_B(matrixA_original, matrixA, n, n);
    Write_A_over_B(matrixB_original, matrixB, n, n);

    // Now in Pieces
    // First LU Decomposition

    int ipiv[n];
    start = std::chrono::high_resolution_clock::now();
    LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, matrixA, n, ipiv);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    cout << "LU decomposition: " << (duration.count() * 1.e-9) << " s.";  
  
    From_A_to_PLU(matrixA,ipiv,n,matrixL,matrixU,matrixP);
    // LU = L *U
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, matrixL, n, matrixU, n, 0.0, matrixLU, n);
     
    cout << "\nCheck Accuracy and time of PLU=A : ";
    ErrorCalc_Display(matrixP, matrixA_original, matrixLU, duration.count() * 1.e-9, n, n);

    // Now Solve system of linear equations given AX=B given B is n x n
    // Solve AX=B -> PLUX=B -> LUX=P(inv)*B -> (2) UX=Y -> (1) LY=P(inv)*B
    // Solve (1) LY=P(inv)*B

    TransposeColMajor(matrixP,n,n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, matrixP, n, matrixB, n, 0.0, matrixB_Calc, n);
    
    start1 = std::chrono::high_resolution_clock::now();
    LAPACK_dgesv(&n, &n, matrixL, &n, ipiv, matrixB_Calc, &n, &INFO);
    // (2) UX=Y 
    LAPACK_dgesv(&n, &n, matrixU, &n, ipiv, matrixB_Calc, &n, &INFO);
    stop1 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop1 - start1);
    cout << "Lower+Upper Solve: " << (duration.count() * 1.e-9) << " s.\n";
  
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>((stop-start)+(stop1 - start1));
   
    cout << "\nCheck Accuracy and time of sequenced LU+Lsolve+Usolve -> AX=B:";
    ErrorCalc_Display(matrixA_original, matrixB_original, matrixB_Calc, duration.count() * 1.e-9, n, n);

    return 0;

}
