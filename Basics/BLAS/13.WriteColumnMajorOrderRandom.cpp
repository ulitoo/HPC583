#include <random>
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
void Write_A_over_B(double *matrixA, double *matrixB, int m,int n)
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
    double max=0.0;
    double tmp=0.0;
    for (int i = 0; i < n; i++)
    {
        tmp=0.0;
        for (int j = 0; j < n; j++)
        {
            tmp += abs(matrixA[i + (j * n)]);
        }
        if (tmp>max)
        {
            max=tmp;
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
    InverseMatrix(matrixA,n);

    double InfNormA, InfNormAinv;
    InfNormA = InfinityNorm(matrixA, n);
    InfNormAinv = InfinityNorm(matrixA_original, n);
    
    // restore original Matrix
    Write_A_over_B(matrixA_original, matrixA, n, n);
    free(matrixA_original);
    return InfNormA * InfNormAinv;

}


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

    // Write the matrices to file in binary format
    std::ofstream outfilea("matrix_A", std::ios::out | std::ios::binary);
    std::ofstream outfileb("matrix_B", std::ios::out | std::ios::binary);
    if (outfilea.is_open() and outfileb.is_open())
    {
        outfilea.write(reinterpret_cast<char *>(matrixA), sizeof(double) * n * n);
        outfilea.close();
        outfileb.write(reinterpret_cast<char *>(matrixB), sizeof(double) * n * n);
        outfileb.close();
    }
    else
    {
        std::cerr << "Failed to open file/s" << std::endl;
        return 1;
    }

    cout << "Random Matrices correctly created in files of size:" << n <<"x"<<n<<"\n";
    cout << "\nMatrix A Condition Number: " << ConditionNumber(matrixA,n,n) << "\n";
    cout << "Matrix B Condition Number: " << ConditionNumber(matrixB,n,n) << "\n";
    //PrintColMatrix(matrixA,n,n);cout << "\n";PrintColMatrix(matrixB,n,n);

    return 0;
}
