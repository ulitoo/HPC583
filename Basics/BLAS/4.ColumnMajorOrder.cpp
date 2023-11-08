#include <random>
#include <cblas.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>

using namespace std;

// 	Problem , get a matrix from disk into Column Major Order
// Compute the decompositions depicted in linear algebra

int main ( int argc, char* argv[] ) {

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

    // Print the matrix elements
    cout << "This is the Matrix in Row Major Order \n";
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << matrix[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    cout << "\nThis is the Matrix in Column Major Order \n";
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << matrix[j * n + i] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
