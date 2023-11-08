#include <random>
#include <cblas.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>

using namespace std;

// Perform LU decomposition using Doolittle algorithm
void luDecomposition(std::vector<std::vector<double>> &matrix)
{
    int size = matrix.size(); // Get the size of the matrix

    for (int pivot = 0; pivot < size; ++pivot)
    {
        // Check for division by zero error
        if (matrix[pivot][pivot] == 0)
        {
            throw std::runtime_error("Zero pivot encountered");
        }

        // Compute the lower triangular entries
        for (int row = pivot + 1; row < size; ++row)
        {
            matrix[row][pivot] /= matrix[pivot][pivot];
        }

        // Compute the upper triangular entries
        for (int row = pivot + 1; row < size; ++row)
        {
            for (int col = pivot + 1; col < size; ++col)
            {
                matrix[row][col] -= matrix[row][pivot] * matrix[pivot][col];
            }
        }
    }
}

void PrintMatrix(std::vector<std::vector<double>> &A)
{
    int n = A.size();

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main ( int argc, char* argv[] )
{
    
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <filename> rank(A)" << std::endl;
        return 1;
    }

    const int n = std::atoi(argv[2]); // rank

    // Open the binary file for reading
    std::ifstream input(argv[1], std::ios::binary);
    if (!input)
    {
        std::cerr << "Error: could not open file for reading" << std::endl;
        return 1;
    }

    // Create a vector to store the matrix data
    std::vector<double> matrix(n * n);

    // Read the binary data into the vector
    input.read(reinterpret_cast<char *>(matrix.data()), sizeof(double) * matrix.size());

    // Check if read was successful
    if (!input)
    {
        std::cerr << "Error: could not read file" << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> A(n, std::vector<double> (n, 0.0));
    
    // Assign read matrix from File to A
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A[i][j]=matrix[i * n + j];
        }
    }
 
    std::vector<std::vector<double>> L(n, std::vector<double> (n, 0.0));
    std::vector<std::vector<double>> U(n, std::vector<double> (n, 0.0));
    std::vector<std::vector<double>> A_fin(n, std::vector<double> (n, 0.0));

    std::cout << "Matrix A:" << std::endl;
    PrintMatrix (A);

    luDecomposition(A);

    std::cout << "\nMatrix A Decomposed:" << std::endl;
    PrintMatrix (A);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i > j)
            {
                L[i][j] = A[i][j];
            }
            else if (i == j)
            {
                L[i][j] = 1.0;
            }
            else
            {
                L[i][j] = 0.0;
            }
        }
    }


    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i <= j)
            {
                U[i][j] = A[i][j];
            }
            else
            {
                U[i][j] = 0.0;
            }
        }
    }

    // Print the L and U matrices

    std::cout << "\nMatrix L:" << std::endl;
    PrintMatrix (L);
    std::cout << "\nMatrix U:" << std::endl;
    PrintMatrix (U);

    //Multiply    
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < n; ++k){
                A_fin[i][j] += L[i][k]*U[k][j];
            }
        }
    }

    std::cout << "\nMatrix A_fin:L*U:" << std::endl;
    PrintMatrix (A_fin);

    return 0;
}
