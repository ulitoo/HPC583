#include <iostream>
#include <vector>
#include <cblas.h>

// Perform LU decomposition
void luDecomposition(std::vector<std::vector<double>>& A) {
    int n = A.size();

    for (int k = 0; k < n; ++k) {
        for (int i = k + 1; i < n; ++i) {
            A[i][k] /= A[k][k];
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
}

int main() {
    // Create a sample square matrix A
    std::vector<std::vector<double>> A = {
        {4.0, 3.0, 2.0},
        {2.0, 3.0, 4.0},
        {3.0, 5.0, 1.0}
    };

    int n = A.size();

    luDecomposition(A);

    // Print the L and U matrices
    std::cout << "Matrix L:" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i > j) {
                std::cout << A[i][j] << " ";
            } else if (i == j) {
                std::cout << "1.0 ";
            } else {
                std::cout << "0.0 ";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "Matrix U:" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i <= j) {
                std::cout << A[i][j] << " ";
            } else {
                std::cout << "0.0 ";
            }
        }
        std::cout << std::endl;
    }

    return 0;
}
