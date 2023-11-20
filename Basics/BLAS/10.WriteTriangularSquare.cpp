#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// CREATE LOWER and UPPER triangular Matrices of rank n

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
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " n (Dimension of  Matrix (nxn))" << std::endl;
        return 1;
    }

    const int n = std::atoi(argv[1]);
    
    // Create the matrices (L and U) IN ROW MAJOR and fill it with Ordered N values
    std::vector<double> matrixL(n * n);
    std::vector<double> matrixU(n * n);
    std::vector<double> matrixB(n * n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            matrixB[i + j * n] = (double)(n*n - i - j * n);
            if (i <= j)
            {
                matrixU[i + j * n] = (double)(i + j * n + 1);
            }
            else
            {
                matrixU[i + j * n] = 0;
            }
            if (i >= j)
            {
                matrixL[i + j * n] = (double)(i + j * n + 1);
            }
            else 
            {
                matrixL[i + j * n] = 0;
            }
        }
    }

    // Write the Column Major matrix to file in binary format
    std::ofstream outfile("Umatrix.bin", std::ios::out | std::ios::binary);
    if (outfile.is_open())
    {
        outfile.write(reinterpret_cast<const char *>(matrixU.data()), sizeof(double) * n * n);
        outfile.close();
    }
    else
    {
        std::cerr << "Failed to open file: matrixU.bin" << std::endl;
        return 1;
    }

    std::ofstream outfile2("Lmatrix.bin", std::ios::out | std::ios::binary);
    if (outfile2.is_open())
    {
        outfile2.write(reinterpret_cast<const char *>(matrixL.data()), sizeof(double) * n * n);
        outfile2.close();
    }
    else
    {
        std::cerr << "Failed to open file: matrixL.bin" << std::endl;
        return 1;
    }
    
        std::ofstream outfile3("Bmatrix.bin", std::ios::out | std::ios::binary);
    if (outfile3.is_open())
    {
        outfile3.write(reinterpret_cast<const char *>(matrixB.data()), sizeof(double) * n * n);
        outfile3.close();
    }
    else
    {
        std::cerr << "Failed to open file: matrixB.bin" << std::endl;
        return 1;
    }

    // Print Matrix L Colum Major
    //cout << "\nmatrixL Print:\n";
    //PrintColMatrix(matrixL,n,n);
    // Print Matrix U Colum Major
    //cout << "\nmatrixU Print:\n";
    //PrintColMatrix(matrixU,n,n);
    // Print Matrix B Colum Major
    //cout << "\nmatrixB Print:\n";
    //PrintColMatrix(matrixB,n,n);

    cout << "\nMatrices " << n <<" x " << n << " correctly Created and written in file as ColMajor\n\n";

    return 0;
}
