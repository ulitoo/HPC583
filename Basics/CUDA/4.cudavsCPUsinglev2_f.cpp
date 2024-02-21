#include <cblas.h>
#include <iostream>
#include <fstream>
#include <lapacke.h>
using namespace std;

/////////////////////////////   General Purpose  FUNCTIONS

int Read_Matrix_file(float *matrix, int size, char *filename)
{
    // Open the binary file for reading and handle error
    std::ifstream input(filename, std::ios::binary);
    if (!input)
    {
        std::cerr << "Error: could not open file for reading" << std::endl;
        return 1;
    }
    // Read the binary data into the vector
    input.read(reinterpret_cast<char *>(matrix), sizeof(float) * size);
    // Check if read was successful and handle error
    if (!input)
    {
        std::cerr << "Error: could not read file" << std::endl;
        return 1;
    }
    std::cout << "File " << filename << " read correctly!" << std::endl;
    return 0;
}
void Write_A_over_B(float *matrixA, float *matrixB, int m, int n)
{
    for (int i = 0; i < m * n; i++)
    {
        matrixB[i] = matrixA[i];
    }
}
void InverseMatrix(float *matrixA, int n)
{
    int ipiv[n];
    int info;
    // Compute the inverse using LAPACK's dgetrf and dgetri
    // Perform LU factorization
    info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, matrixA, n, ipiv);

    if (info == 0)
    {
        // LU factorization succeeded, now compute the inverse
        info = LAPACKE_sgetri(LAPACK_COL_MAJOR, n, matrixA, n, ipiv);

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
void PrintColMatrix(float *matrix, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (abs(matrix[i + (j * m)]) < 0.00000000000000000001)
            {
                cout << 0.0 << " ";
            }
            else
            {
                cout << matrix[i + (j * m)] << " ";
            }
        }
        cout << "\n";
    }
    cout << "\n";
}
void MakeZeroes(float *matrix, int m, int n)
{
    for (int i = 0; i < m * n; i++)
    {
        matrix[i] = 0.0;
    }
}
void MakeIdentity(float *matrix, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                matrix[i + j * m] = 1.0;
            }
            else
            {
                matrix[i + j * m] = 0.0;
            }
        }
    }
}
int MaxRow(float *matrix, int n)
{
    int maxrow;
    float maxabsvalue = 0;
    float temp_abs_element;

    for (int i = 0; i < n; i++)
    {
        temp_abs_element = abs(matrix[i]);
        if (temp_abs_element > maxabsvalue)
        {
            maxrow = i;
            maxabsvalue = temp_abs_element;
        }
    }
    return maxrow;
}
int MaxRow2(float *matrix, int N, int n)
{
    int offset = N - n;
    int maxrow = offset;
    float maxabsvalue = 0.0;
    float temp_abs_element;

    for (int i = offset; i < N; i++)
    {
        temp_abs_element = abs(matrix[i + offset * N]);
        if (temp_abs_element > maxabsvalue)
        {
            maxrow = i;
            maxabsvalue = temp_abs_element;
        }
    }
    return maxrow;
}
float MatrixDistance_norm2(float *matrixa, float *matrixb, int m, int n)
{
    float diff = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
        diff += (matrixa[i] - matrixb[i]) * (matrixa[i] - matrixb[i]);
    }
    return sqrt(diff);
}
void SwapRow_ColMajMatrix(float *matrix, int from, int towards, int m, int n)
{
    float tmpval;
    for (int i = 0; i < n * m; i += m)
    {
        tmpval = matrix[towards + i];
        matrix[towards + i] = matrix[from + i];
        matrix[from + i] = tmpval;
    }
}
float float_machine_epsilon()
{
    float epsilon,tmp;
    float one=1.;
    float half=1./2.;
    int j;

    for (j=0;;j++){
        tmp = 1.+pow(half,static_cast<float>(j)); 
        
        if ((tmp-one) == 0.){
            break;
        } 

    }
    //std::cout << "Iter: " << j-1 << " Value:" << pow(half,static_cast<float>(j-1)) << endl;
    epsilon = pow(half,static_cast<float>(j-1));
    return epsilon;
}

/////////////////////////////   Triangular Solver FUNCTIONS
void InitializeSubmatrices(float *matrixc, float *C11, float *C12, float *C21, float *C22, int m, int p)
{
    // Initialize will Create the submatrices based on the big matrix
    int mm = m / 2;
    int mm2 = m - mm;
    int pp = p / 2;
    for (int i = 0; i < mm; ++i)
    {
        for (int j = 0; j < pp; ++j)
        {
            C11[i + (j * mm)] = matrixc[i + (j * m)];
        }
    }
    for (int i = 0; i < mm; ++i)
    {
        for (int j = pp; j < p; ++j)
        {
            C12[i + ((j - pp) * mm)] = matrixc[i + (j * m)];
        }
    }
    for (int i = mm; i < m; ++i)
    {
        for (int j = 0; j < pp; ++j)
        {
            C21[(i - mm) + (j * mm2)] = matrixc[i + (j * m)];
        }
    }
    for (int i = mm; i < m; ++i)
    {
        for (int j = pp; j < p; ++j)
        {
            C22[(i - mm) + ((j - pp) * mm2)] = matrixc[i + (j * m)];
        }
    }
}
void CollectSubmatrices(float *matrixc, float *C11, float *C12, float *C21, float *C22, int m, int p)
{
    // Collect Results of Xxx to the big matrix X
    MakeZeroes(matrixc, m, p);
    int mm = m / 2;
    int mm2 = m - mm;
    int pp = p / 2;
    for (int i = 0; i < mm; ++i)
    {
        for (int j = 0; j < pp; ++j)
        {
            matrixc[i + (j * m)] += C11[i + (j * mm)];
        }
    }
    for (int i = 0; i < mm; ++i)
    {
        for (int j = pp; j < p; ++j)
        {
            matrixc[i + (j * m)] += C12[i + ((j - pp) * mm)];
        }
    }
    for (int i = mm; i < m; ++i)
    {
        for (int j = 0; j < pp; ++j)
        {
            matrixc[i + (j * m)] += C21[(i - mm) + (j * mm2)];
        }
    }
    for (int i = mm; i < m; ++i)
    {
        for (int j = pp; j < p; ++j)
        {
            matrixc[i + (j * m)] += C22[(i - mm) + ((j - pp) * mm2)];
        }
    }
}
void UpperTriangularSolverRecursiveReal_0(float *matrixU, float *matrixB, float *matrixX, int n, int p)
{
    // This is a Naive version with Malloc and free as crutch to avoid index calculation over the original matrix
    if (n == 1)
    {
        for (int j = 0; j < p; j++)
        {
            matrixX[j] = matrixB[j] / matrixU[0];
        }
    }
    else
    {
        int nn = n / 2;
        int nn2 = n - nn;
        int pp = p / 2;
        int pp2 = p - pp;

        float *U11 = (float *)malloc(nn * nn * sizeof(float));
        float *U12 = (float *)malloc(nn * nn2 * sizeof(float));
        float *U21 = (float *)malloc(nn2 * nn * sizeof(float));
        float *U22 = (float *)malloc(nn2 * nn2 * sizeof(float));
        float *B11 = (float *)malloc(nn * pp * sizeof(float));
        float *B12 = (float *)malloc(nn * pp2 * sizeof(float));
        float *B21 = (float *)malloc(nn2 * pp * sizeof(float));
        float *B22 = (float *)malloc(nn2 * pp2 * sizeof(float));
        float *X11 = (float *)malloc(nn * pp * sizeof(float));
        float *X12 = (float *)malloc(nn * pp2 * sizeof(float));
        float *X21 = (float *)malloc(nn2 * pp * sizeof(float));
        float *X22 = (float *)malloc(nn2 * pp2 * sizeof(float));

        // Initializa Axx and Bxx matrices!
        InitializeSubmatrices(matrixU, U11, U12, U21, U22, n, n);
        InitializeSubmatrices(matrixB, B11, B12, B21, B22, n, p);

        // Recurse U22 X21 = B21
        UpperTriangularSolverRecursiveReal_0(U22, B21, X21, nn2, pp);
        // Recurse U22 X22 = B22
        UpperTriangularSolverRecursiveReal_0(U22, B22, X22, nn2, pp2);

        // PHASE 2: CALCULATE THE NEW B's for next Phase
        // B11' = B11 - U12 X21
        // B12' = B12 - U12 X22
        for (int i = 0; i < nn; ++i)
        {
            for (int j = 0; j < pp; ++j)
            {
                for (int k = 0; k < nn2; ++k)
                {
                    B11[i + ((j)*nn)] -= (U12[i + (k)*nn]) * (X21[k + (j)*nn2]);
                    B12[i + ((j)*nn)] -= (U12[i + (k)*nn]) * (X22[k + (j)*nn2]);
                }
            }
        }
        // PHASE 3: RECURSE on REST of calculations with TRIANGULAR A22
        // Recurse U11 X11 = B11'
        UpperTriangularSolverRecursiveReal_0(U11, B11, X11, nn, pp);

        // Recurse U11 X12 = B12'
        UpperTriangularSolverRecursiveReal_0(U11, B12, X12, nn, pp2);

        // At the end Collect pieces of matrixc = matrixc + C11 + C12 + C21 + C22 and done!
        CollectSubmatrices(matrixX, X11, X12, X21, X22, n, p);

        free(U11);
        free(U12);
        free(U21);
        free(U22);
        free(B11);
        free(B12);
        free(B21);
        free(B22);
        free(X11);
        free(X12);
        free(X21);
        free(X22);
    }
}
void LowerTriangularSolverRecursiveReal_0(float *matrixL, float *matrixB, float *matrixX, int n, int p)
{
    // PHASE 1: RECURSE on calculations based on TRIANGULAR L11
    if (n == 1)
    {
        for (int j = 0; j < p; j++)
        {
            matrixX[j] = matrixB[j] / matrixL[0];
        }
    }
    else
    {
        int nn = n / 2;
        int nn2 = n - nn; // Size of right or lower side covers for odd cases
        int pp = (p / 2);
        int pp2 = p - pp;

        float *L11 = (float *)malloc(nn * nn * sizeof(float));
        float *L12 = (float *)malloc(nn * nn2 * sizeof(float));
        float *L21 = (float *)malloc(nn2 * nn * sizeof(float));
        float *L22 = (float *)malloc(nn2 * nn2 * sizeof(float));
        float *B11 = (float *)malloc(nn * pp * sizeof(float));
        float *B12 = (float *)malloc(nn * pp2 * sizeof(float));
        float *B21 = (float *)malloc(nn2 * pp * sizeof(float));
        float *B22 = (float *)malloc(nn2 * pp2 * sizeof(float));
        float *X11 = (float *)malloc(nn * pp * sizeof(float));
        float *X12 = (float *)malloc(nn * pp2 * sizeof(float));
        float *X21 = (float *)malloc(nn2 * pp * sizeof(float));
        float *X22 = (float *)malloc(nn2 * pp2 * sizeof(float));

        // Initialize Axx and Bxx matrices!
        InitializeSubmatrices(matrixL, L11, L12, L21, L22, n, n);
        InitializeSubmatrices(matrixB, B11, B12, B21, B22, n, p);

        // Recurse L11 X11 = B11
        LowerTriangularSolverRecursiveReal_0(L11, B11, X11, nn, pp);

        // Recurse L11 X12 = B12
        LowerTriangularSolverRecursiveReal_0(L11, B12, X12, nn, pp2);

        // PHASE 2: CALCULATE THE NEW B's for next Phase
        // B21' = B21 - L21 X11
        // B22' = B22 - L21 X12
        for (int i = 0; i < nn2; ++i)
        {
            for (int j = 0; j < pp; ++j)
            {
                for (int k = 0; k < nn; ++k)
                {
                    B21[i + (j * nn2)] -= (L21[i + (k * nn2)]) * (X11[k + (j)*nn]);
                    B22[i + (j * nn2)] -= (L21[i + (k * nn2)]) * (X12[k + (j)*nn]);
                }
            }
        }
        // PHASE 3: RECURSE on REST of calculations with TRIANGULAR A22
        // Recurse L22 X21 = B21'
        LowerTriangularSolverRecursiveReal_0(L22, B21, X21, nn2, pp);

        // Recurse L22 X22 = B22'
        LowerTriangularSolverRecursiveReal_0(L22, B22, X22, nn2, pp2);

        // At the end Collect pieces of matrixc = matrixc + C11 + C12 + C21 + C22 and done!
        CollectSubmatrices(matrixX, X11, X12, X21, X22, n, p);

        free(L11);
        free(L12);
        free(L21);
        free(L22);
        free(B11);
        free(B12);
        free(B21);
        free(B22);
        free(X11);
        free(X12);
        free(X21);
        free(X22);
    }
}

void LowerTriangularSolverNaiveReal(float *matrixL, float *matrixB, float *matrixSol, int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            float tempval = 0;
            for (int k = 0; k < i; k++)
            {
                tempval += matrixL[i + k * n] * matrixSol[k + j * n];
            }
            matrixSol[j * n + i] = (matrixB[j * n + i] - tempval) / matrixL[i * n + i];
        }
    }
}
void UpperTriangularSolverNaiveReal(float *matrixU, float *matrixB, float *matrixSol, int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = n - 1; i >= 0; i--)
        {
            float tempval = 0;
            for (int k = n - 1; k > i; k--)
            {
                tempval += matrixU[i + k * n] * matrixSol[k + j * n];
            }
            matrixSol[j * n + i] = (matrixB[j * n + i] - tempval) / matrixU[i * n + i];
        }
    }
}

/////////////////////////////   LU Decomposition FUNCTIONS
void ipiv_to_P(int *ipiv, int n, float *P)
{
    MakeIdentity(P, n, n);

    for (int i = 0; i < n; i++)
    {
        SwapRow_ColMajMatrix(P, i, ipiv[i], n, n);
    }
}
void SchurComplement(float *matrix, int N, int n)
{
    int offset = (N - n);
    int offset2 = offset * (N + 1);
    // ONLY 1 matrix, rewrite A22 as S22 ! ; N is the original A size ; n is the size of A in the recursion; n-1 is size of A22
    for (int i = 0; i < (n - 1); i++)
    {
        int i2 = (i + offset2 + 1);
        for (int j = 0; j < (n - 1); j++)
        {
            // This is in Column Major Order
            int k = (j + 1) * N;
            matrix[i2 + k] = matrix[(i2) + (k)] - ((matrix[offset2 + k]) * (matrix[i + 1 + (offset2)]) / matrix[offset2]);
        }
    }
}
void SchurComplement2(float *matrix, int n)
{
    // ONLY 1 matrix, rewrite A22 as S22 ! ; N is the original A size ; n is the size of A in the recursion; n-1 is size of A22
    // Without offset concept just 1 rank lower
    for (int i = 1; i < (n); i++)
    {
        for (int j = 1; j < (n); j++)
        {
            // This is in Column Major Order
            matrix[(i) + (j * n)] = matrix[(i) + (j * n)] - ((matrix[j * n]) * (matrix[i])) / matrix[0];
        }
    }
}
void LUdecompositionRecursive2(float *matrix, float *Lmatrix, float *Umatrix, int N, int n)
{
    // Assume square Matrix for simplicity

    int offset = (N - n);
    int offset2 = N * offset;
    int offset3 = (offset) + offset2;
    int j, i2;
    Umatrix[(offset) * (N + 1)] = matrix[(offset) * (N + 1)];
    Lmatrix[(offset) * (N + 1)] = 1.0;

    for (int i = 1; i < n; i++)
    {
        j = i * N + offset3;
        i2 = i + offset3;
        // offset*N unnecesary repeated calculations!!!!!!!!!!!!!!
        Lmatrix[i2] = matrix[i2] / matrix[offset3];
        // Lmatrix[j] = 0.0;  Redundant, Already Zero
        Umatrix[j] = matrix[j];
        // Umatrix[i2] = 0.0; Redundant, Already Zero
    }

    if (n == 2)
    {
        Umatrix[(offset + 1) + (offset + 1) * N] = matrix[(offset + 1) + (offset + 1) * N] - matrix[(offset + 1) + (offset)*N] * matrix[(offset) + (offset + 1) * N] / matrix[(offset) + (offset)*N];
        Lmatrix[(offset + 1) + (offset + 1) * N] = 1.0;
    }
    else
    {
        SchurComplement(matrix, N, n);
        LUdecompositionRecursive2(matrix, Lmatrix, Umatrix, N, n - 1);
    }
}
void LUdecompositionRecursive4Pivot(float *AmatrixBIG, float *LmatrixBIG, float *UmatrixBIG, int *IPIV, int N, int n)
{
    //   WITHOUT MALLOC
    int offset = (N - n);
    int offset2 = N * offset;
    int offset3 = (offset) + offset2;
    int offset4 = (offset) * (N + 1);
    int k, i2;

    // Assume square Matrices for simplicity
    IPIV[offset] = MaxRow2(AmatrixBIG, N, n); // Tracks ROW Exchange
    // Permutation
    SwapRow_ColMajMatrix(AmatrixBIG, offset, IPIV[offset], N, N);
    // Amatrix is now Amatrixbar(permutation done)

    UmatrixBIG[(offset4)] = AmatrixBIG[(offset4)];
    LmatrixBIG[(offset4)] = 1.0;

    if (n == 2)
    {
        IPIV[N - 1] = N - 1; // Last Row does NOT change!

        UmatrixBIG[(offset4 + N + 1)] = AmatrixBIG[(offset4 + N + 1)] - AmatrixBIG[(offset4 + 1)] * AmatrixBIG[(offset4 + N)] / AmatrixBIG[(offset4)];
        LmatrixBIG[(offset4 + N + 1)] = 1.0;

        LmatrixBIG[(offset4 + 1)] = AmatrixBIG[(offset4 + 1)] / AmatrixBIG[offset4];
        UmatrixBIG[(offset4 + N)] = AmatrixBIG[(offset4 + N)];
    }
    else
    {
        SchurComplement(AmatrixBIG, N, n);
        LUdecompositionRecursive4Pivot(AmatrixBIG, LmatrixBIG, UmatrixBIG, IPIV, N, n - 1);

        for (int i = 1; i < n; i++)
        {
            k = i * N + offset3;
            i2 = i + offset3;
            UmatrixBIG[k] = AmatrixBIG[k];
            LmatrixBIG[i2] += AmatrixBIG[i2] / AmatrixBIG[(offset3)];
        }
    }
}

/////////////////////////////// Error Calculating Functions
float InfinityNorm(float *matrixA, int n)
{
    // Find the biggest sum of abs (rows)
    float max = 0.0;
    float tmp = 0.0;
    for (int i = 0; i < n; i++)
    {
        tmp = 0.0;
        for (int j = 0; j < n; j++)
        {
            tmp += abs(matrixA[i + (j * n)]);
        }
        if (tmp > max)
        {
            max = tmp;
        }
    }
    return max;
}
float InfinityNormVector(float *vectorA, int n)
{
    // Find the biggest abs
    float max = 0.0;
    float tmp = 0.0;
    for (int i = 0; i < n; i++)
    {
        tmp = abs(vectorA[i]); 
        if (tmp > max)
        {
            max = tmp;
        }
    }
    return max;
}
float ConditionNumber(float *matrixA, int m, int n)
{
    //  Find condition number for the Matrix /Norm of matrix/ Infinity norm (max row or col)
    //  The infinity-norm of a square matrix is the maximum of the absolute row sum
    //  Condition number is the ||M|| times ||M^(-1)||, the closer to 1 the more stable
    float *matrixA_original = (float *)malloc(n * n * sizeof(float));
    float InfNormA, InfNormAinv;

    Write_A_over_B(matrixA, matrixA_original, n, n);

    InfNormA = InfinityNorm(matrixA, n);
    InverseMatrix(matrixA, n);
    InfNormAinv = InfinityNorm(matrixA, n);
    
    // restore original Matrix
    Write_A_over_B(matrixA_original, matrixA, n, n);
    free(matrixA_original);
    return InfNormA * InfNormAinv;
}
float residual_matrix(float *matrixA, float *matrixX, float *matrixB, float *matrixResidual, int m, int n)
{
    float *CalculatedB = (float *)malloc(m * n * sizeof(float));
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, matrixA, n, matrixX, n, 0.0, CalculatedB, n);
    for (int i = 0; i < m*n; i++)
    {
        matrixResidual[i] = CalculatedB[i] - matrixB[i];
    }
    free(CalculatedB);
    return InfinityNorm(matrixResidual,m);
}
void ErrorCalc_Display(float *matrixA, float *matrixB, float *matrixX, long double elapsed_time, int n, int p)
{
    float *CalculatedB = (float *)malloc(n * p * sizeof(float));
    MakeZeroes(CalculatedB, n, p);
    // NaiveMatrixMultiplyCol(matrixA, matrixX, CalculatedB, n, n, p);
    // Substitute by LAPACK dGEMM
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, matrixA, n, matrixX, n, 0.0, CalculatedB, n);
    float dist = MatrixDistance_norm2(matrixB, CalculatedB, n, p);
    cout << "\nVector Distance - Error (AX - B):----------------> : " << dist << "\n";
    cout << "Elapsed Time:------------------------------------> : " << elapsed_time << " s.\n\n";
    free(CalculatedB);
}
void ErrorCalc_Display_v2(int i, float *matrixA, float *matrixB, float *matrixX, float *results, int m, int n)
{
    // !!! this function returns a vector with : [7] [Matrix size, Residual Norm,A Norm, X Norm, Machine Epsilon, Fwd Error,Elapsed Time]
    // Infinity norm of a vector is its max absolute value
    
    float *matrixResidual = (float *)malloc(m * n * sizeof(float));
    float residual_norm = residual_matrix(matrixA,matrixX,matrixB,matrixResidual,m,n);
    float epsilon = float_machine_epsilon();
    float A_norm = InfinityNorm(matrixA,m);
    float X_norm = InfinityNorm(matrixX,m);
    float fwd_error = residual_norm/(A_norm*X_norm*epsilon);
    cout << "\nResidual Norm:----------------> : " << residual_norm << "\n";
    cout << "A Norm:----------------> : " << A_norm << "\n";
    cout << "X Norm:----------------> : " << X_norm << "\n";
    cout << "Machine epsilon:----------------> : " << epsilon << "\n";
    cout << "|Residual| / (|A||X|epsilon) : FWD Error:--------> : " << fwd_error << "\n";
    cout << "Elapsed Time:------------------------------------> : " << results[7*i+6] << " s.\n\n";
    results[7*i+0]=m;
    results[7*i+1]=residual_norm;
    results[7*i+2]=A_norm;
    results[7*i+3]=X_norm;
    results[7*i+4]=epsilon;
    results[7*i+5]=fwd_error;
    free(matrixResidual);

}

