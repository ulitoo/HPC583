#include <iostream>
#include <fstream>
#include <random>
#include <vector>

using namespace std;

void PrintColMatrix(double *matrix, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << matrix[i + (j * m)] << " ";
        }
        cout << "\n";
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " n (Dimension of Matrix)" << std::endl;
        return 1;
    }

    const int n = std::atoi(argv[1]);

    // Create a random number generator
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    double *matrixA = (double *)malloc(n * n * sizeof(double));
    double *matrixB = (double *)malloc(n * n * sizeof(double));
    
    // Create the matrix and fill it with random values
    for (int i = 0; i < n * n; i++)
    {
        matrixA[i] = dist(rng);
        matrixB[i] = dist(rng);
    }
    
    // Write the matrices to file in binary format
    std::ofstream outfilea("matrix_A", std::ios::out | std::ios::binary);
    std::ofstream outfileb("matrix_B", std::ios::out | std::ios::binary);
    if (outfilea.is_open() and outfileb.is_open())
    {
        outfilea.write(reinterpret_cast<char *>(matrixA), sizeof(double) * n * n);
        outfilea.close();
        outfileb.write(reinterpret_cast<char *>(matrixB), sizeof(double) * n * n);
        outfileb.close();
    }
    else
    {
        std::cerr << "Failed to open file/s" << std::endl;
        return 1;
    }

    cout << "Random Matrices correctly created in files of size:" << n <<"x"<<n<<"\n";
    //PrintColMatrix(matrixA,n,n);cout << "\n";PrintColMatrix(matrixB,n,n);

    return 0;
}
