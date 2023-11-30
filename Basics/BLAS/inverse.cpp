#include <iostream>
#include <cblas.h>
#include <lapacke.h>

int main() {
    // Define your matrix and its dimensions
    int n = 3;  // Adjust the size accordingly
    double matrix[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };

    // Compute the inverse using LAPACK's dgetrf and dgetri
    int ipiv[n];
    int info;

    // Perform LU factorization
    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, matrix, n, ipiv);

    if (info == 0) {
        // LU factorization succeeded, now compute the inverse
        info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, matrix, n, ipiv);

        if (info == 0) {
            // Inversion succeeded, 'matrix' now contains the inverse
            std::cout << "Inverse matrix:" << std::endl;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    std::cout << matrix[i * n + j] << " ";
                }
                std::cout << std::endl;
            }
        } else {
            std::cerr << "Error in LAPACKE_dgetri: " << info << std::endl;
        }
    } else {
        std::cerr << "Error in LAPACKE_dgetrf: " << info << std::endl;
    }

    return 0;
}
