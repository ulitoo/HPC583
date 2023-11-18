#include <cblas.h>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>
#include "gnuplot-iostream.h" // Gnuplot C++ interface

using namespace std;

// 	Problem , get a recursive algorithm for Matrix Matrix Multiplication of any size 
//  Matrices given in row major order : A (m,n) x B (n,p) = C (m,p)
//  Algorithm to be done in Row Major 
//  USE POINTERS in this exercise double *C = (double *)malloc(m * n * sizeof(double));
//  free to avoid Memory leaks!!!!!!

/////////////////////////////     FUNCTIONS

void PrintRowMatrix(double *matrix, int m, int n)
{
    for (int i = 0; i < m * n; i++)
    {
        cout << matrix[i] << " ";
        if (i%n==n-1) {
            cout << "\n";
        }
    }  
}

void RowtoColMajor_Transpose(double *matrix, m, n)
{
    std::vector<double> tmpmatrixCol(m * n);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            tmpmatrixCol[j+(i*m)] = matrix[i+(j*m)];
        }
    }
    return tmpmatrixCol;
}

std::vector<double> MMMultRecursive(std::vector<double> matrixA,std::vector<double> matrixB,int m, int n, int p)
{
    
    std::vector<double> matrixC(n * p);
    double tmpval;
    return matrixC;
}

/////////////////////////////     MAIN

int main ( int argc, char* argv[] ) {
    
    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " <filenameA> <filenameB> m n p " << std::endl;
        return 1;
    }
    
    const int m = std::atoi(argv[3]); // m
    const int n = std::atoi(argv[4]); // n
    const int p = std::atoi(argv[5]); // p

    // Create a vector to store the Column Major matrix data and another for the Row Major
    double *matrixA = (double *)malloc(m * n * sizeof(double));
    double *matrixB = (double *)malloc(n * p * sizeof(double));
    double *matrixC = (double *)malloc(m * p * sizeof(double));
    double *matrixC_test = (double *)malloc(m * p * sizeof(double));

    // Open the binary file for reading and handle error
    std::ifstream inputA(argv[1], std::ios::binary);
    std::ifstream inputB(argv[2], std::ios::binary);
    //std::ifstream input("matrix_random.bin", std::ios::binary);
    if (!inputA or !inputB){std::cerr << "Error: could not open file for reading" << std::endl; return 1;}
    // Read the binary data into the vector
    inputA.read(reinterpret_cast<char *>(matrixA), sizeof(double) * m * n);
    inputB.read(reinterpret_cast<char *>(matrixB), sizeof(double) * n * p);
    // Check if read was successful and handle error 
    if (!inputA or !inputB) {std::cerr << "Error: could not read file" << std::endl; return 1;}
    
    double A[] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}; //n=3;m=3;p=3;
    double B[] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0};
    for (int i = 0; i < m * n; ++i) {
        matrixA[i]=A[i];
    }
    for (int i = 0; i < n*p; ++i) {
        matrixB[i]=B[i];
    }


    // Print the matrix elements
    cout << "\nMatrix A Print:\n";
    PrintRowMatrix(matrixA,m,n);
    cout << "\nMatrix B Print:\n";
    PrintRowMatrix(matrixB,n,p);
    
    // Transform Matrix A into Row Major to ensure adjacent data during multiplication of rows
    //matrixA = RowtoColMajor_Transpose (matrixA);
    
    // Multiply

    // Transform Matrix A back into Col Major
    //matrixA = RowtoColMajor_Transpose (matrixA);

    cout << "\nMatrix C Print:\n";
    PrintRowMatrix(matrixC,m,p);
   

    // NAIVE Multiply    
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < p; ++k){
                matrixC_test[i+j*n] += matrixA[i+k*n]*matrixB[k+j*n];
            }
        }
    }

    std::cout << "\nMatrix Ctest:" << std::endl;
    PrintRowMatrix (matrixC_test,m,p);

    double diff=0;
    //Error    
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            diff += abs(matrixC[i*n+j] - matrixC_test[i*n+j]);
        }
    }

    std::cout << "\nC - Ctest: " << diff << std::endl;

    free(matrixA);
    free(matrixB);
    free(matrixC);
    free(matrixC_test);
    return 0;
}
