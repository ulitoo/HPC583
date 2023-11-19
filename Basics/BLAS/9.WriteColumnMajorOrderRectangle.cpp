#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
// in case you want to include all the functions in a sigle file
//#include "6.WriteColumnMajorOrder_f.h"

using namespace std;

void PrintColMatrix(std::vector<double> matrix, int m, int n)
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

std::vector<double> RowtoColMajor_Transpose(std::vector<double> matrix,int m, int n)
{
    std::vector<double> tmpmatrixCol(m * n);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            tmpmatrixCol[j+(i*n)] = matrix[i+(j*m)];
        }
    }
    return tmpmatrixCol;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " m n (Dimension of  Matrix (m*n))" << std::endl;
        return 1;
    }

    const int m = std::atoi(argv[1]);
    const int n = std::atoi(argv[2]);
    
    // Create the matrix IN ROW MAJOR and fill it with Ordered N values
    std::vector<double> matrix(m * n);
    std::vector<double> matrixtrsp(m * n);
    for (int i = 0; i < m * n; i++) {matrix[i] = (double)(i+1);}
    matrixtrsp = RowtoColMajor_Transpose(matrix,m,n);

    // Write the Column Major matrix to file in binary format
    std::ofstream outfile("matrixCol.bin", std::ios::out | std::ios::binary);
    if (outfile.is_open())
    {
        outfile.write(reinterpret_cast<const char *>(matrix.data()), sizeof(double) * m * n);
        outfile.close();
    }
    else
    {
        std::cerr << "Failed to open file: matrix.bin" << std::endl;
        return 1;
    }
    
    // Print Matrix  Colum Major
    cout << "\nRowMajor matrix Print:\n";
    PrintColMatrix(matrix,m,n);
    // Print Matrix  Colum Major
    cout << "\nSame Trasposed matrix Print:\n";
    PrintColMatrix(matrixtrsp,n,m);

    cout << "\nMatrix correctly Created and written in file as ColMajor\n\n";

    return 0;
}
