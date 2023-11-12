#include <cblas.h>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>
#include "gnuplot-iostream.h" // Gnuplot C++ interface


using namespace std;

// 	Problem , get a matrix from disk into Column Major Order
//  Implement GEMV
//  Implement Col Swap and Row Swap 
//  Compare with GEMV BLAS performance

/////////////////////////////     FUNCTIONS

void PrintColMatrix(std::vector<double> matrix)
{
    int n = round(sqrt(matrix.size()));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << matrix[j*n+i] << " ";
        }
        cout << "\n";
    }
}

void PrintVector(int n, double *vector)
{
    for (int i = 0; i < n; i++)
    {
        cout << vector[i] << " ";
    }
    cout << "\n";
}

void InitVector(int n, double *vector)
{
    for (int i = 0; i < n; i++)
    {
        vector[i] = (double)i;
    }
}

void gemv1(int n, double alpha, std::vector<double> matrix_A, double *vector_x, double beta, double *vector_y, double *vector_z)
{
    for (int i = 0; i < n; i++)
    {
        vector_z[i] = beta * vector_y[i];
        for (int j = 0; j < n; j++)
        {
            vector_z[i] += alpha * matrix_A[i+(n*j)] * vector_x[j];
        }  
    }
}

void gemv2(int n, double alpha, std::vector<double> matrix_A_trsps, double *vector_x, double beta, double *vector_y, double *vector_z)
{
    for (int i = 0; i < n; i++)
    {
        vector_z[i] = beta * vector_y[i];
        for (int j = 0; j < n; j++)
        {
            vector_z[i] += alpha * matrix_A_trsps[j+(n*i)] * vector_x[j];
        }  
    }
}

std::vector<double> SwapCol_ColMajMatrix(std::vector<double> matrix,int from, int towards)
{
    int n = round(sqrt(matrix.size()));
    double tmpval;
    for (int i = 0; i < n; i++)
    {
        tmpval = matrix[towards*n+i];
        matrix[towards*n+i]= matrix[from*n+i];
        matrix[from*n+i]=tmpval;
    }
    return matrix;
}

std::vector<double> SwapRow_ColMajMatrix(std::vector<double> matrix,int from, int towards)
{
    int n = round(sqrt(matrix.size()));
    double tmpval;
    for (int i = 0; i < n; i++)
    {
        tmpval = matrix[towards+i*n];
        matrix[towards+i*n]= matrix[from+i*n];
        matrix[from+i*n]=tmpval;
    }
    return matrix;
}

std::vector<double> RowtoColMajor_Transpose(std::vector<double> matrix)
{
    int n = round(sqrt(matrix.size()));
    std::vector<double> tmpmatrixCol(n * n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            tmpmatrixCol[j+(i*n)] = matrix[i+(j*n)];
        }
    }
    return tmpmatrixCol;
}


/////////////////////////////     MAIN

int main ( int argc, char* argv[] ) {
    // Timers
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time;

    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <filename> rank(A) beta alpha" << std::endl;
        return 1;
    }

    const int n = std::atoi(argv[2]); // rank
    double alpha = (double)std::atoi(argv[3]);
    double beta = (double)std::atoi(argv[4]);

    // Create a vector to store the Column Major matrix data
    std::vector<double> matrixCol(n * n);
    // Create and init the rest of the vectors for the GEMV 
    double vectorx[n] = {1.0};
    double vectory[n] = {1.0};
    double vectorz[n] = {0.0};
    InitVector(n, vectorx);
    InitVector(n, vectory);

    // Open the binary file for reading and handle error
    std::ifstream input(argv[1], std::ios::binary);
    if (!input){std::cerr << "Error: could not open file for reading" << std::endl; return 1;}

    // Read the binary data into the vector
    input.read(reinterpret_cast<char *>(matrixCol.data()), sizeof(double) * matrixCol.size());

    // Check if read was successful and handle error 
    if (!input) {std::cerr << "Error: could not read file" << std::endl; return 1;}

    // Print the matrix elements
    cout << "\nColMajor matrix Print:\n";
    PrintColMatrix(matrixCol);
    cout << "\nResult VectorX Print:\n";
    PrintVector(n, vectorx);
    cout << "\nResult VectorY Print:\n";
    PrintVector(n, vectory);
    cout << "\nResult VectorZ Print:\n";
    PrintVector(n, vectorz);
    
    
/*
    start = std::chrono::high_resolution_clock::now();
    gemv1(n,alpha,matrixCol,vectorx,beta,vectory,vectorz);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time = (duration.count() * 1.e-9);
    std::cout << "\nNaive Elapsed Time: " << elapsed_time << " s.\n";

  
    std::vector<double> matrixRow(n * n);
    matrixRow = RowtoColMajor_Transpose(matrixCol);
    start = std::chrono::high_resolution_clock::now();
    gemv2(n,alpha,matrixRow,vectorx,beta,vectory,vectorz);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time = (duration.count() * 1.e-9);
    std::cout << "\nTransposed Elapsed Time: " << elapsed_time << " s.\n";
*/

    //Compare with BLAS
    start = std::chrono::high_resolution_clock::now();
    cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, alpha, &matrixCol[0], n, vectorx, 1, beta, vectory, 1);
    cout << "\nResult VectorZ Print con CBLAS:\n";
    PrintVector(n, vectory);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time = (duration.count() * 1.e-9);
    std::cout << "\nCBLAS Elapsed Time: " << elapsed_time << " s.\n";


    cout << "\nColMajor matrix after swap Col alpha with beta Print:\n";
    PrintColMatrix(SwapCol_ColMajMatrix(matrixCol,round(alpha),round(beta)));
    
    cout << "\nColMajor matrix after swap Row alpha with beta Print:\n";
    PrintColMatrix(SwapRow_ColMajMatrix(matrixCol,round(alpha),round(beta)));    

    return 0;
}
