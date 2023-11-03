#include <iostream>
#include <fstream>
#include <random>
#include <vector>

using namespace std;

// Analize the CODE then analyze the STR code at the end:

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
    //cout << "rng: " << rng<< "\n";
    //cout << "DIST: " << dist << " and Dist(RNG): " << dist(rng) << " and Dist: " << dist(rng) << "\n";
    
    // Create the matrix and fill it with random values
    std::vector<double> matrix(n * n);
    std::ofstream outfiletxt("matrix.txt");
    outfiletxt << "This is the Matrix in Row Marjor Order \n";
    for (int i = 0; i < n * n; i++)
    {
        matrix[i] = dist(rng);
        outfiletxt << matrix[i] << " ";
        if (i%n==n-1) {
            outfiletxt << "\n";
        }
    }
    outfiletxt << "\nThis is the Matrix in Column Marjor Order \n";
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            outfiletxt << matrix[j*n+i] << " ";
        }
        outfiletxt << "\n";
    }

    outfiletxt.close();

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

    cout << "Data: " << *matrix.data() << "\n";

    return 0;
}
