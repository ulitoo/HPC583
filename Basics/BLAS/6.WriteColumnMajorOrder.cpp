#include <iostream>
#include <fstream>
#include <vector>
#include "6.WriteColumnMajorOrder_f.h"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " n (Dimension of square Matrix (n*n))" << std::endl;
        return 1;
    }

    const int n = std::atoi(argv[1]);
    
    // Create the matrix and fill it with Ordered N values
    std::vector<double> matrix(n * n);
    cout << "This is the Matrix in Row Major Order \n";
    for (int i = 0; i < n * n; i++)
    {
        matrix[i] = (double)i;
        cout << matrix[i] << " ";
        if (i%n==n-1) {
            cout << "\n";
        }
    }
    cout << "\nThis is the Matrix in Column Major Order \n";
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << matrix[j*n+i] << " ";
        }
        cout << "\n";
    }

    // Write the matrix to file in binary format
    std::ofstream outfile("matrix.bin", std::ios::out | std::ios::binary);
    if (outfile.is_open())
    {
        outfile.write(reinterpret_cast<const char *>(matrix.data()), sizeof(double) * n * n);
        outfile.close();
    }
    else
    {
        std::cerr << "Failed to open file: matrix.bin" << std::endl;
        return 1;
    }

    cout << "\nMatrix correctly Created\n\n";

    return 0;
}
