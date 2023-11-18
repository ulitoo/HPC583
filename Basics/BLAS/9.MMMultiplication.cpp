#include <cblas.h>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>
#include "gnuplot-iostream.h" // Gnuplot C++ interface

using namespace std;

// 	Problem , get a recursive algorithm for Matrix Matrix Multiplication of any size 
//  Matrices given in COLUMN major order : A (m,n) x B (n,p) = C (m,p)
//  Algorithm to be done in COLUMN Major 
//  USE POINTERS in this exercise double *C = (double *)malloc(m * n * sizeof(double));
//  free to avoid Memory leaks!!!!!!

/*
    //THE OPTIMAL approach would be to transpose A so rows are adjecent in A*B multiplication
    //We will forget about this fact for now and focus in the recursive aspect
    //Later we can analyze this further optimization

    ColMajor_Transpose(matrixA,m,n);
    cout << "\nMatrix A traspose Print:\n";
    PrintColMatrix(matrixA,n,m);
    ColMajor_Transpose(matrixA,n,m);
    cout << "\nMatrix A trasposed again to original format:\n";
    PrintColMatrix(matrixA,m,n);
*/

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

void PrintColMatrix(double *matrix, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << matrix[i+(j*m)] << " ";
        }
        cout << "\n";
    }
}

void NaiveMatrixMultiplyCol(double *matrixa, double *matrixb, double *matrixc, int m, int n,int p)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                matrixc[i+(j*m)] += matrixa[i+(k*m)]*matrixb[k+(j*n)];
            }
        }
    }
}

double MatrixAbsDiff(double *matrixa, double *matrixb, int m, int p)
{
    double diff= 0.0;
    for (int i = 0; i < m*p; ++i)
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
            tmpmatrix[j+(i*n)] = matrix[i+(j*m)];
        }
    }
    for (int i = 0; i < m*n; i++)
    {
        matrix[i] = tmpmatrix[i];
    }
    free(tmpmatrix);
}

double *MMMultRecursive(double *matrixa, double *matrixb, int m, int n,int p)
{
    double *tmpmatrix = (double *)malloc(m * p * sizeof(double));
    int mm = m/2;
    int pp = p/2;
    int nn = n/2;

    double tmpval;
    return tmpmatrix;
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
    std::ifstream input("matrix_random.bin", std::ios::binary);
    if (!inputA or !inputB){std::cerr << "Error: could not open file for reading" << std::endl; return 1;}
    // Read the binary data into the vector
    inputA.read(reinterpret_cast<char *>(matrixA), sizeof(double) * m * n);
    inputB.read(reinterpret_cast<char *>(matrixB), sizeof(double) * n * p);
    // Check if read was successful and handle error 
    if (!inputA or !inputB) {std::cerr << "Error: could not read file" << std::endl; return 1;}
    
    // Print the matrix elements
    cout << "\nMatrix A Print:\n";
    PrintColMatrix(matrixA,m,n);
    cout << "\nMatrix B Print:\n";
    PrintColMatrix(matrixB,n,p);

    
    // Multiply


    cout << "\nMatrix C Print:\n";
    PrintColMatrix(matrixC,m,p);
   
    NaiveMatrixMultiplyCol(matrixA,matrixB,matrixC_test,m,n,p);

    std::cout << "\nMatrix Ctest:" << std::endl;
    PrintColMatrix (matrixC_test,m,p);

    double diff=MatrixAbsDiff(matrixC_test,matrixC,m,p);

    std::cout << "\nABS (C - Ctest): " << diff << std::endl;

    free(matrixA);
    free(matrixB);
    free(matrixC);
    free(matrixC_test);
    return 0;
}
