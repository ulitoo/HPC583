#include <cblas.h>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>
#include "gnuplot-iostream.h" // Gnuplot C++ interface

using namespace std;

// 	Problem , get a LU decomposition recursively
//  Algorithm to be done in Row Major 
// Now try to explore the PIVOT

/////////////////////////////     FUNCTIONS

void PrintRowMatrix(std::vector<double> matrix)
{
    int n = round(sqrt(matrix.size()));
    for (int i = 0; i < n * n; i++)
    {
        cout << matrix[i] << " ";
        if (i%n==n-1) {
            cout << "\n";
        }
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

std::vector<double> SchurComplement(std::vector<double> matrix)
{
    int n = round(sqrt(matrix.size()));
    std::vector<double> tmpmatrixCol((n-1) * (n-1));
    for (int i = 0; i < (n-1); i++)
    {
        for (int j = 0; j < (n-1); j++)
        {
            tmpmatrixCol[j+(i*(n-1))] = matrix[(j+1)+((i+1)*n)] - ((matrix[(i+1)*(n)]) * (matrix[j+1]) / matrix[0]) ;
        }   
    }
    return tmpmatrixCol;
}

std::vector<double> LUdecomposition1(std::vector<double> matrix)
{
    int n = round(sqrt(matrix.size()));
    std::vector<double> LUmatrix((n) * (n));
    std::vector<double> S22((n-1) * (n-1));
    std::vector<double> LU22((n-1) * (n-1));
    
    LUmatrix[0]=matrix[0];
    for (int i = 1; i < n; i++)
    {
        LUmatrix[i] = matrix[i] / matrix[0];
        LUmatrix[i*n] = matrix[i*n];
    }
    
    if (n==2) 
    {
        LUmatrix[3] = matrix [3] - matrix[1]*matrix[2]/matrix[0];  
        return LUmatrix;    
    }
    else
    {
        S22 = SchurComplement (matrix);
        LU22 = LUdecomposition1 (S22);
        for (int i = 1; i < n; i++)
        {
            for (int j = 1; j < n; j++)
            {
            LUmatrix[j+i*n] = LU22[(j-1)+(i-1)*(n-1)];
            }
        }
        return LUmatrix;
    }
}


/////////////////////////////     MAIN

int main ( int argc, char* argv[] ) {
    
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <filename> rank(A) " << std::endl;
        return 1;
    }
    
    const int n = std::atoi(argv[2]); // rank

    // Create a vector to store the Column Major matrix data and another for the Row Major
    std::vector<double> matrixCol(n * n);
    std::vector<double> matrixRow(n * n);
    std::vector<double> matrixLU(n * n);
    std::vector<double> matrixL(n * n);
    std::vector<double> matrixU(n * n);
    std::vector<double> A_fin(n * n);
    // Open the binary file for reading and handle error
    std::ifstream input(argv[1], std::ios::binary);
    //std::ifstream input("matrix_random.bin", std::ios::binary);
    if (!input){std::cerr << "Error: could not open file for reading" << std::endl; return 1;}
    // Read the binary data into the vector
    input.read(reinterpret_cast<char *>(matrixCol.data()), sizeof(double) * matrixCol.size());
    // Check if read was successful and handle error 
    if (!input) {std::cerr << "Error: could not read file" << std::endl; return 1;}
    // Transform Matrix into Row Major
    matrixRow = RowtoColMajor_Transpose (matrixCol);

    matrixRow = {7.0,1.0,5.0,4.0,3.0,5.0,6.0,1.0,2.0};

    // Print the matrix elements
    cout << "\nMatrix A Print:\n";
    PrintRowMatrix(matrixRow);

    matrixLU = LUdecomposition1(matrixRow);
    
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i==j){
                matrixL[j + i * n] = matrixLU[j + i * n];
                matrixU[j + i * n] = 1.0;
            }
            else if (i>j){
                matrixL[j + i * n] = matrixLU[j + i * n];
                matrixU[j + i * n] = 0.0;
            }
            else{
                matrixL[j + i * n] = 0.0;
                matrixU[j + i * n] = matrixLU[j + i * n];    
            }
        }
    }

    //cout << "\nMatrix LU Print:\n";
    //PrintRowMatrix(matrixLU);    
    cout << "\nMatrix L Print:\n";
    PrintRowMatrix(matrixL);
    cout << "\nMatrix U Print:\n";
    PrintRowMatrix(matrixU);

    //Multiply    
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < n; ++k){
                A_fin[i*n+j] += matrixL[i*n+k]*matrixU[k*n+j];
            }
        }
    }

    std::cout << "\nMatrix A_fin:L*U:" << std::endl;
    PrintRowMatrix (A_fin);

    double diff=0;
    //Error    
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            diff += abs(A_fin[i*n+j] - matrixRow[i*n+j]);
        }
    }

    std::cout << "\nCumulative error:" << diff << std::endl;
    return 0;
}
