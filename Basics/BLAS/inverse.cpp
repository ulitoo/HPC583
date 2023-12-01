#include <iostream>
#include <cblas.h>
#include <lapacke.h>

void InverseMatrix(double *matrixA, int n)
    {
        int ipiv[n];
        int info;
        // Compute the inverse using LAPACK's dgetrf and dgetri
        // Perform LU factorization
        info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, matrixA, n, ipiv);

        if (info == 0)
        {
            // LU factorization succeeded, now compute the inverse
            info = LAPACKE_dgetri(LAPACK_COL_MAJOR, n, matrixA, n, ipiv);

            if (info != 0)
            {
                std::cerr << "Error in LAPACKE_dgetri: " << info << std::endl;
            }
        }
        else
        {
            std::cerr << "Error in LAPACKE_dgetrf: " << info << std::endl;
        }
    }


int main() {
    // Define your matrix and its dimensions
    int n = 3;  // Adjust the size accordingly
    double matrix[] = {
        112.0, 2.0, 3.0,
        4.0, 11.0, 6.0,
        12.0, 8.0, 92.0
    };

    InverseMatrix(matrix,n);

    
    return 0;
}
