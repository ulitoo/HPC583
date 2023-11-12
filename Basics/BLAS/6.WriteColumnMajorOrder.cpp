#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
// in case you want to include all the functions in a sigle file
//#include "6.WriteColumnMajorOrder_f.h"

using namespace std;

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

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " n (Dimension of square Matrix (n*n))" << std::endl;
        return 1;
    }

    const int n = std::atoi(argv[1]);
    
    // Create the matrix IN ROW MAJOR and fill it with Ordered N values
    std::vector<double> matrixRow(n * n);
    std::vector<double> matrixCol(n * n);
    for (int i = 0; i < n * n; i++) {matrixRow[i] = (double)i;}
    matrixCol = RowtoColMajor_Transpose(matrixRow);

    // Write the Column Major matrix to file in binary format
    std::ofstream outfile("matrixCol.bin", std::ios::out | std::ios::binary);
    if (outfile.is_open())
    {
        outfile.write(reinterpret_cast<const char *>(matrixCol.data()), sizeof(double) * n * n);
        outfile.close();
    }
    else
    {
        std::cerr << "Failed to open file: matrix.bin" << std::endl;
        return 1;
    }
    
    // Print Matrix in Row Major and Colum Major
    cout << "\nRowMajor matrix Print:\n";
    //PrintRowMatrix(matrixRow);
    cout << "\nColMajor matrix Print:\n";
    //PrintColMatrix(matrixCol);

    cout << "\nMatrix correctly Created and written in file as ColMajor\n\n";

    return 0;
}
