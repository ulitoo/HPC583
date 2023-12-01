#include <cblas.h>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>
#include <pthread.h>
#include <thread>
#include <lapacke.h>

using namespace std;

// 	Problem:
//  1 One Single Clean Code
//  2 Get Matrix
//  3 Find condition number for the Matrix 
//  4 Run 3 routines with timing and Matrix Distance 
//      -With Pivot
//      -With NO Pivot
//      -LAPACK BLAS
//  5. OPTIMIZE PIVOT CODE? LUdecompositionRecursive4Pivot


/////////////////////////////     FUNCTIONS

int Read_Matrix_file(double *matrix, int size, char *filename)
{
    // Open the binary file for reading and handle error
    std::ifstream input(filename, std::ios::binary);
    if (!input)
    {
        std::cerr << "Error: could not open file for reading" << std::endl;
        return 1;
    }
    // Read the binary data into the vector
    input.read(reinterpret_cast<char *>(matrix), sizeof(double) * size);
    // Check if read was successful and handle error
    if (!input)
    {
        std::cerr << "Error: could not read file" << std::endl;
        return 1;
    }
    std::cout << "File " << filename << " read correctly!" << std::endl;
    return 0;
}
void Write_A_over_B(double *matrixA, double *matrixB, int m,int n)
{
    for (int i = 0; i < m * n; i++)
    {
        matrixB[i] = matrixA[i];
    }
}
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
double InfinityNorm(double *matrixA, int n)
{
    // Find the biggest sum of abs (rows)
    double max=0.0;
    double tmp=0.0;
    for (int i = 0; i < n; i++)
    {
        tmp=0.0;
        for (int j = 0; j < n; j++)
        {
            tmp += abs(matrixA[i + (j * n)]);
        }
        if (tmp>max)
        {
            max=tmp;
        }
    }
    return max;
}
double ConditionNumber(double *matrixA, int m, int n)
{
    //  Find condition number for the Matrix /Norm of matrix/ Infinity norm (max row or col)
    //  The infinity-norm of a square matrix is the maximum of the absolute row sum
    //  Condition number is the ||M|| times ||M^(-1)||, the closer to 1 the more stable
    double *matrixA_original = (double *)malloc(n * n * sizeof(double));
    Write_A_over_B(matrixA, matrixA_original, n, n);
    InverseMatrix(matrixA,n);

    double InfNormA, InfNormAinv;
    InfNormA = InfinityNorm(matrixA, n);
    InfNormAinv = InfinityNorm(matrixA_original, n);
    
    // restore original Matrix
    Write_A_over_B(matrixA_original, matrixA, n, n);
    free(matrixA_original);
    return InfNormA * InfNormAinv;

}
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
void MakeZeroes(double *matrix, int m, int n)
{
    for (int i = 0; i < m * n; i++)
    {
        matrix[i] = 0.0;
    }
}
void MakeIdentity(double *matrix, int m, int n)
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
int MaxRow(double *matrix, int n)
{
    int maxrow;
    double maxabsvalue = 0;
    double temp_abs_element;

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
double MatrixDistance(double *matrixa, double *matrixb, int m, int n)
{
    double diff = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
        diff += (matrixa[i] - matrixb[i])*(matrixa[i] - matrixb[i]);
    }
    return sqrt(diff);
}
void ColMajor_Transpose(double *matrix, int m, int n)
{
    double *tmpmatrix = (double *)malloc(m * n * sizeof(double));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            tmpmatrix[j + (i * n)] = matrix[i + (j * m)];
        }
    }
    for (int i = 0; i < m * n; i++)
    {
        matrix[i] = tmpmatrix[i];
    }
    free(tmpmatrix);
}

void SwapCol_ColMajMatrix(double *matrix,int from, int towards, int m, int n)
{
    double tmpval;
    int towards2=towards*m;
    int from2=from*m;
    for (int i = 0; i < m; i++)
    {
        tmpval = matrix[i+towards2];
        matrix[i+towards2] = matrix[i+from2];
        matrix[i+from2] = tmpval;
    }
}
void SwapRow_ColMajMatrix(double *matrix,int from, int towards, int m, int n)
{
    double tmpval;
    for (int i = 0; i < n*m; i+=m)
    {
        tmpval = matrix[towards+i];
        matrix[towards+i] = matrix[from+i];
        matrix[from+i] = tmpval;
    }
}

void InitializeSubmatrices(double *matrixc, double *C11, double *C12, double *C21, double *C22, int m, int p)
{
    //Initialize will Create the submatrices based on the big matrix
    int mm = m / 2;
    int mm2 = m - mm;
    int pp = p / 2;
    for (int i = 0; i < mm; ++i)
    {
        for (int j = 0; j < pp; ++j)
        {
            C11[i + (j * mm)] = matrixc[i + (j * m)] ;
        }
    }
    for (int i = 0; i < mm; ++i)
    {
        for (int j = pp; j < p; ++j)
        {
            C12[i + ((j-pp) * mm)] = matrixc[i + (j * m)];
        }
    }
    for (int i = mm; i < m; ++i)
    {
        for (int j = 0; j < pp; ++j)
        {
            C21[(i-mm) + (j * mm2)] = matrixc[i + (j * m)];
        }
    }
    for (int i = mm; i < m; ++i)
    {
        for (int j = pp; j < p; ++j)
        {
            C22[(i-mm) + ((j-pp) * mm2)] = matrixc[i + (j * m)];
        }
    }
}
void CollectSubmatrices(double *matrixc, double *C11, double *C12, double *C21, double *C22, int m, int p)
{
    //Collect Results of Xxx to the big matrix X
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
            matrixc[i + (j * m)] += C12[i + ((j-pp) * mm)];
        }
    }
    for (int i = mm; i < m; ++i)
    {
        for (int j = 0; j < pp; ++j)
        {
            matrixc[i + (j * m)] += C21[(i-mm) + (j * mm2)];
        }
    }
    for (int i = mm; i < m; ++i)
    {
        for (int j = pp; j < p; ++j)
        {
            matrixc[i + (j * m)] += C22[(i-mm) + ((j-pp) * mm2)];
        }
    }
}

void UpperTriangularSolverRecursiveReal_0(double *matrixU, double *matrixB, double *matrixX, int n, int p)
{
    // This is a Naive version with Malloc and free as crutch to avoid index calculation over the original matrix
    if (n==1)
    {
        for (int j = 0; j < p; j++)
        {
            matrixX[j] = matrixB[j]/matrixU[0];
        }
    }
    else
    {
        int nn = n / 2;
        int nn2 = n - nn;
        int pp = p / 2;
        int pp2 = p - pp;
        
        double *U11 = (double *)malloc(nn * nn * sizeof(double));
        MakeZeroes(U11, nn, nn);
        double *U12 = (double *)malloc(nn * nn2 * sizeof(double));
        MakeZeroes(U12, nn, nn2);
        double *U21 = (double *)malloc(nn2 * nn * sizeof(double));
        MakeZeroes(U21, nn2, nn);
        double *U22 = (double *)malloc(nn2 * nn2 * sizeof(double));
        MakeZeroes(U22, nn2, nn2);
        double *B11 = (double *)malloc(nn * pp * sizeof(double));
        MakeZeroes(B11, nn, pp);
        double *B12 = (double *)malloc(nn * pp2 * sizeof(double));
        MakeZeroes(B12, nn, pp2);
        double *B21 = (double *)malloc(nn2 * pp * sizeof(double));
        MakeZeroes(B21, nn2, pp);
        double *B22 = (double *)malloc(nn2 * pp2 * sizeof(double));
        MakeZeroes(B22, nn2, pp2);
        double *X11 = (double *)malloc(nn * pp * sizeof(double));
        MakeZeroes(X11, nn, pp);
        double *X12 = (double *)malloc(nn * pp2 * sizeof(double));
        MakeZeroes(X12, nn, pp2);
        double *X21 = (double *)malloc(nn2 * pp * sizeof(double));
        MakeZeroes(X21, nn2, pp);
        double *X22 = (double *)malloc(nn2 * pp2 * sizeof(double));
        MakeZeroes(X22, nn2, pp2);

        // Initializa Axx and Bxx matrices!
        InitializeSubmatrices(matrixU, U11, U12, U21, U22, n,n);
        InitializeSubmatrices(matrixB, B11, B12, B21, B22, n,p);

        // Recurse U22 X21 = B21
        UpperTriangularSolverRecursiveReal_0(U22,B21,X21,nn2,pp);
        // Recurse U22 X22 = B22
        UpperTriangularSolverRecursiveReal_0(U22,B22,X22,nn2,pp2);
        
        // PHASE 2: CALCULATE THE NEW B's for next Phase     
        // B11' = B11 - U12 X21
        for (int i = 0; i < nn; ++i)
        {
            for (int j = 0; j < pp; ++j)
            {
                for (int k = 0; k < nn2; ++k)
                {
                    B11[i + ((j) * nn)] -= (U12[i + (k) * nn]) * (X21[ k + (j) * nn2]);
                    // A(i,k)*B(k,j)
                }
            }
        }
        // PHASE 3: RECURSE on REST of calculations with TRIANGULAR A22
        // Recurse U11 X11 = B11'
        UpperTriangularSolverRecursiveReal_0(U11,B11,X11,nn,pp);
        
        // B12' = B12 - U12 X22
        for (int i = 0; i < nn; ++i)
        {
            for (int j = 0; j < pp2; ++j)
            {
                for (int k = 0; k < nn2; ++k)
                {
                    B12[i + ((j) * nn)] -= (U12[i + (k) * nn]) * (X22[ k + (j) * nn2]);
                    // A(i,k)*B(k,j)
                }
            }
        }
        // Recurse U11 X12 = B12'
        UpperTriangularSolverRecursiveReal_0(U11,B12,X12,nn,pp2);
    
        // At the end Collect pieces of matrixc = matrixc + C11 + C12 + C21 + C22 and done!
        CollectSubmatrices(matrixX,X11,X12,X21,X22,n,p);
        
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
void LowerTriangularSolverRecursiveReal_0(double *matrixL, double *matrixB, double *matrixX, int n, int p)
{
    //cout << " This is the iteration " << n << " x " << p << "\n"; // This Line is for debugging
    // PHASE 1: RECURSE on calculations based on TRIANGULAR L11
    if (n == 1)
    {   
        for (int j = 0; j < p; j++)
        {
            matrixX[j] = matrixB[j]/matrixL[0];
            //matrixSol[B_n1 + (B_p1 + j)*major_n] = matrixB[B_n1 + (B_p1 + j)*major_n]/matrixL[L_n1 + L_n2*major_n];  
        }
    }
    else
    {
        int nn = n / 2;
        int nn2 = n - nn; // Size of right or lower side covers for odd cases 
        int pp = (p/2); 
        int pp2 = p - pp;

        double *L11 = (double *)malloc(nn * nn * sizeof(double));
        MakeZeroes(L11, nn, nn);
        double *L12 = (double *)malloc(nn * nn2 * sizeof(double));
        MakeZeroes(L12, nn, nn2);
        double *L21 = (double *)malloc(nn2 * nn * sizeof(double));
        MakeZeroes(L21, nn2, nn);
        double *L22 = (double *)malloc(nn2 * nn2 * sizeof(double));
        MakeZeroes(L22, nn2, nn2);
        double *B11 = (double *)malloc(nn * pp * sizeof(double));
        MakeZeroes(B11, nn, pp);
        double *B12 = (double *)malloc(nn * pp2 * sizeof(double));
        MakeZeroes(B12, nn, pp2);
        double *B21 = (double *)malloc(nn2 * pp * sizeof(double));
        MakeZeroes(B21, nn2, pp);
        double *B22 = (double *)malloc(nn2 * pp2 * sizeof(double));
        MakeZeroes(B22, nn2, pp2);
        double *X11 = (double *)malloc(nn * pp * sizeof(double));
        MakeZeroes(X11, nn, pp);
        double *X12 = (double *)malloc(nn * pp2 * sizeof(double));
        MakeZeroes(X12, nn, pp2);
        double *X21 = (double *)malloc(nn2 * pp * sizeof(double));
        MakeZeroes(X21, nn2, pp);
        double *X22 = (double *)malloc(nn2 * pp2 * sizeof(double));
        MakeZeroes(X22, nn2, pp2);
        
        // Initializa Axx and Bxx matrices!
        InitializeSubmatrices(matrixL, L11, L12, L21, L22, n,n);
        InitializeSubmatrices(matrixB, B11, B12, B21, B22, n,p);

        // Recurse L11 X11 = B11
        LowerTriangularSolverRecursiveReal_0(L11,B11,X11,nn,pp);
        //LowerTriangularSolverRecursiveReal(matrixL,L_n1,L_n2,matrixB,B_n1,B_p1,matrixSol,major_n,nn,pp);
        // Recurse L11 X12 = B12
        LowerTriangularSolverRecursiveReal_0(L11,B12,X12,nn,pp2);
        //LowerTriangularSolverRecursiveReal(matrixL,L_n1,L_n2,matrixB,B_n1,B_p1+pp,matrixSol,major_n,nn,pp2);
    
        // PHASE 2: CALCULATE THE NEW B's for next Phase     
        // B21' = B21 - L21 X11
        for (int i = 0; i < nn2; ++i)
        {
            for (int j = 0; j < pp; ++j)
            {
                for (int k = 0; k < nn; ++k)
                {
                    B21[i + (j*nn2)] -= (L21[i + (k*nn2)]) * (X11[ k + (j) * nn]);
                    //matrixB[B_n1 + nn + i + ((B_p1 + j) * major_n)] -= (matrixL[L_n1 + nn + i + (L_n2 + k) * major_n]) * (matrixSol[B_n1 + k + (B_p1 + j) * major_n]);
                    // A(i,k)*B(k,j)
                }
            }
        }
        // PHASE 3: RECURSE on REST of calculations with TRIANGULAR A22
        // Recurse L22 X21 = B21'
        LowerTriangularSolverRecursiveReal_0(L22,B21,X21,nn2,pp);
        //LowerTriangularSolverRecursiveReal(matrixL,L_n1+nn,L_n2+nn,matrixB,B_n1+nn,B_p1,matrixSol,major_n,nn2,pp);

        // B22' = B22 - L21 X12
        for (int i = 0; i < nn2; ++i)
        {
            for (int j = 0; j < pp2; ++j)
            {
                for (int k = 0; k < nn; ++k)
                {
                    B22[i + (j*nn2)] -= (L21[i + (k*nn2)]) * (X12[ k + (j) * nn]);
                    //matrixB[B_n1 + nn + i + ((B_p1 + pp + j) * major_n)] -= (matrixL[L_n1 + nn + i + (L_n2 + k) * major_n]) * (matrixSol[B_n1 + k + (B_p1 + pp + j) * major_n]);
                    // A(i,k)*B(k,j)
                }
            }
        }
        // Recurse L22 X22 = B22'
        LowerTriangularSolverRecursiveReal_0(L22,B22,X22,nn2,pp2);
        //LowerTriangularSolverRecursiveReal(matrixL,L_n1+nn,L_n2+nn,matrixB,B_n1+nn,B_p1+pp,matrixSol,major_n,nn2,pp2);

        // At the end Collect pieces of matrixc = matrixc + C11 + C12 + C21 + C22 and done!
        CollectSubmatrices(matrixX,X11,X12,X21,X22,n,p);
        
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

void SchurComplement(double *matrix, int N, int n)
{
    int offset = (N - n);
    int offset2 = offset*(N+1);
    // ONLY 1 matrix, rewrite A22 as S22 ! ; N is the original A size ; n is the size of A in the recursion; n-1 is size of A22
    for (int i = 0; i < (n-1); i++)
    {
        int i2=(i+offset2+1);
        for (int j = 0; j < (n-1); j++)
        {
            //This is in Column Major Order
            matrix[i2+((j+1)*N)] = matrix[(i2)+((j+1)*N)] - ((matrix[offset2+(j+1)*N]) * (matrix[i+1+(offset2)]) / matrix[offset2]) ;
        }   
    }
}
void SchurComplement2(double *matrix, int n)
{
    // ONLY 1 matrix, rewrite A22 as S22 ! ; N is the original A size ; n is the size of A in the recursion; n-1 is size of A22
    // Without offset concept just 1 rank lower
    for (int i = 1; i < (n); i++)
    {
        for (int j = 1; j < (n); j++)
        {
            //This is in Column Major Order
            matrix[(i)+(j*n)] = matrix[(i)+(j*n)] - ((matrix[j*n]) * (matrix[i])) / matrix[0] ;
        }   
    }
}
void LUdecompositionRecursive2(double *matrix, double *Lmatrix, double *Umatrix, int N, int n)
{
    //Assume square Matrix for simplicity

    int offset = (N-n);
    int offset2 = N*offset;
    int offset3 = (offset)+offset2;
    int j,i2;
    Umatrix[(offset)*(N+1)] = matrix[(offset)*(N+1)];
    Lmatrix[(offset)*(N+1)] = 1.0;

    for (int i = 1; i < n; i++)
    {
        j=i*N+offset3;
        i2=i+offset3;
        // offset*N unnecesary repeated calculations!!!!!!!!!!!!!!
        Lmatrix[i2] = matrix[i2] / matrix[offset3] ;
        //Lmatrix[j] = 0.0;  Redundant, Already Zero 
        Umatrix[j] = matrix[j];
        //Umatrix[i2] = 0.0; Redundant, Already Zero 
    }
    
    if (n==2) 
    {
        Umatrix[(offset+1)+(offset+1)*N] = matrix[(offset+1)+(offset+1)*N] - matrix[(offset+1)+(offset)*N]*matrix[(offset)+(offset+1)*N]/matrix[(offset)+(offset)*N];
        Lmatrix[(offset+1)+(offset+1)*N] = 1.0;
    }
    else
    {
        SchurComplement(matrix,N,n);
        LUdecompositionRecursive2(matrix, Lmatrix, Umatrix, N, n-1);
    }
}
void LUdecompositionRecursive3Pivot(double *Amatrix, double *Lmatrix, double *Umatrix, double *Pmatrix, int n)
{
    //  do this with NO offsets just passing temporary (m allocated) matrices in recursion loop
    double *matrixP1 = (double *)malloc(n * n * sizeof(double)); 
    double *matrixP2 = (double *)malloc(n * n * sizeof(double)); 
    double *matrixP22 = (double *)malloc((n-1) * (n-1) * sizeof(double)); 
    double *matrixS22 = (double *)malloc((n-1) * (n-1) * sizeof(double));
    double *matrixL22 = (double *)malloc((n-1) * (n-1) * sizeof(double));
    double *matrixU22 = (double *)malloc((n-1) * (n-1) * sizeof(double));

    // Assume square Matrix for simplicity
    // This version relies on Malloc of smaller matrices

    int maxrow = MaxRow(Amatrix,n); 
    MakeIdentity(matrixP1,n,n);
    // Permutation 
    SwapRow_ColMajMatrix(matrixP1,0, maxrow, n, n);
    SwapRow_ColMajMatrix(Amatrix,0, maxrow, n, n);
    // Amatrix is now Amatrixbar(permutation done)
    matrixP2[0] = 1.0;
    
    /* P2 is All Zeros, this is redundant
    for (int i = 1; i < n; i++)
    {
        matrixP2[i * n] = 0.0;
        matrixP2[i] = 0.0;
    }
    */

    Umatrix[0] = Amatrix[0];  
    Lmatrix[0] = 1.0;

    if (n == 2)
    {
        Umatrix[3] = Amatrix[3] - Amatrix[1] * Amatrix[2] / Amatrix[0];
        Lmatrix[3] = 1.0;
        Lmatrix[1] = Amatrix[1] / Amatrix[0] ;  
        //Lmatrix[2] = 0.0; Redundant, Already Zero 
        Umatrix[2] = Amatrix[2];
        //Umatrix[1] = 0.0; Redundant, Already Zero
        matrixP2[3] = 1.0;
    }
    else
    {
        SchurComplement2(Amatrix, n);
        for (int i = 0; i < n - 1; i++)
        {
            for (int j = 0; j < n - 1; j++)
            {
                matrixS22[i + j * (n-1)] = Amatrix[1 + i + (1 + j) * n];
            }
        }
        LUdecompositionRecursive3Pivot(matrixS22, matrixL22, matrixU22, matrixP22, n-1);
        for (int i = 1; i < n; i++)
        {
            //Lmatrix[i] = Amatrix[i] / Amatrix[0];
            //Lmatrix[i * n] = 0.0; Redundant, Already Zero
            Umatrix[i * n] = Amatrix[i * n];
            //Umatrix[i] = 0.0; Redundant, Already Zero
            for (int j = 1; j < n; j++)
            {
                Lmatrix[i] += matrixP22[(i-1)+(j-1)*(n-1)] * Amatrix[j] / Amatrix[0];
                matrixP2[i + j * n] = matrixP22[i - 1 + (j - 1) * (n - 1)];
                Lmatrix[i + j * n] = matrixL22[i - 1 + (j - 1) * (n - 1)];
                Umatrix[i + j * n] = matrixU22[i - 1 + (j - 1) * (n - 1)];
            }
        }
    }

    //End step is calculate the Pmatrix 
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, matrixP2, n, matrixP1, n, 0.0, Pmatrix, n);

    free(matrixL22);
    free(matrixU22);
    free(matrixS22);
    free(matrixP22);
    free(matrixP1);
    free(matrixP2);
}

void LUdecompositionRecursive4Pivot(double *Amatrix, double *Lmatrix, double *Umatrix, double *Pmatrix, int n)
{
    //  TRY do this with offsets WITHOUT MALLOC
    double *matrixP1 = (double *)malloc(n * n * sizeof(double)); 
    double *matrixP2 = (double *)malloc(n * n * sizeof(double)); 
    double *matrixP22 = (double *)malloc((n-1) * (n-1) * sizeof(double)); 
    double *matrixS22 = (double *)malloc((n-1) * (n-1) * sizeof(double));
    double *matrixL22 = (double *)malloc((n-1) * (n-1) * sizeof(double));
    double *matrixU22 = (double *)malloc((n-1) * (n-1) * sizeof(double));

    // Assume square Matrix for simplicity
    // This version relies on Malloc of smaller matrices

    int maxrow = MaxRow(Amatrix,n); 
    MakeIdentity(matrixP1,n,n);
    // Permutation 
    SwapRow_ColMajMatrix(matrixP1,0, maxrow, n, n);
    SwapRow_ColMajMatrix(Amatrix,0, maxrow, n, n);
    // Amatrix is now Amatrixbar(permutation done)
    matrixP2[0] = 1.0;
    
    /* P2 is All Zeros, this is redundant
    for (int i = 1; i < n; i++)
    {
        matrixP2[i * n] = 0.0;
        matrixP2[i] = 0.0;
    }
    */

    Umatrix[0] = Amatrix[0];  
    Lmatrix[0] = 1.0;

    if (n == 2)
    {
        Umatrix[3] = Amatrix[3] - Amatrix[1] * Amatrix[2] / Amatrix[0];
        Lmatrix[3] = 1.0;
        Lmatrix[1] = Amatrix[1] / Amatrix[0] ;  
        //Lmatrix[2] = 0.0; Redundant, Already Zero 
        Umatrix[2] = Amatrix[2];
        //Umatrix[1] = 0.0; Redundant, Already Zero
        matrixP2[3] = 1.0;
    }
    else
    {
        SchurComplement2(Amatrix, n);
        for (int i = 0; i < n - 1; i++)
        {
            for (int j = 0; j < n - 1; j++)
            {
                matrixS22[i + j * (n-1)] = Amatrix[1 + i + (1 + j) * n];
            }
        }
        LUdecompositionRecursive3Pivot(matrixS22, matrixL22, matrixU22, matrixP22, n-1);
        for (int i = 1; i < n; i++)
        {
            //Lmatrix[i] = Amatrix[i] / Amatrix[0];
            //Lmatrix[i * n] = 0.0; Redundant, Already Zero
            Umatrix[i * n] = Amatrix[i * n];
            //Umatrix[i] = 0.0; Redundant, Already Zero
            for (int j = 1; j < n; j++)
            {
                Lmatrix[i] += matrixP22[(i-1)+(j-1)*(n-1)] * Amatrix[j] / Amatrix[0];
                matrixP2[i + j * n] = matrixP22[i - 1 + (j - 1) * (n - 1)];
                Lmatrix[i + j * n] = matrixL22[i - 1 + (j - 1) * (n - 1)];
                Umatrix[i + j * n] = matrixU22[i - 1 + (j - 1) * (n - 1)];
            }
        }
    }

    //End step is calculate the Pmatrix 
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, matrixP2, n, matrixP1, n, 0.0, Pmatrix, n);

    free(matrixL22);
    free(matrixU22);
    free(matrixS22);
    free(matrixP22);
    free(matrixP1);
    free(matrixP2);
}

void ErrorCalc_Display(double *matrixA, double *matrixB, double *matrixX, long double elapsed_time,int n, int p)
{
    double *CalculatedB = (double *)malloc(n * p * sizeof(double));
    MakeZeroes(CalculatedB, n, p);
    // NaiveMatrixMultiplyCol(matrixA, matrixX, CalculatedB, n, n, p);
    // Substitute by LAPACK dGEMM 
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, matrixA, n, matrixX, n, 0.0, CalculatedB, n);
    double dist = MatrixDistance(matrixB, CalculatedB, n, p);
    cout << "\nVector Distance - Error (AX - B):----------------> : " << dist << "\n";
    cout << "Elapsed Time:------------------------------------> : " << elapsed_time << " s.\n\n";
    free(CalculatedB);
}

///     MAIN :: For the sake of simplicity we will Consider all square matrices n x n

int main ( int argc, char* argv[] ) {
    
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <filename A> <filename B> rank " << std::endl;
        return 1;
    }
    const int n = std::atoi(argv[3]); // rank

    // Timers
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time_BLAS, elapsed_time_Pivot, elapsed_time_nonPivot, elapsed_time_Solve;

    // Alloc Space for MATRICES Needed in Column Major Order
    double *matrixA = (double *)malloc(n * n * sizeof(double));
    double *matrixB = (double *)malloc(n * n * sizeof(double));
    double *matrixBPivot = (double *)malloc(n * n * sizeof(double));
    double *matrixL = (double *)malloc(n * n * sizeof(double));
    double *matrixU = (double *)malloc(n * n * sizeof(double));
    double *matrixP = (double *)malloc(n * n * sizeof(double));  //Permutation Matrix
    double *matrixY = (double *)malloc(n * n * sizeof(double));
    double *matrixX = (double *)malloc(n * n * sizeof(double));
    double *matrixB_Calc = (double *)malloc(n * n * sizeof(double));
    double *matrixA_original = (double *)malloc(n * n * sizeof(double)); // in case they get overwritten
    double *matrixB_original = (double *)malloc(n * n * sizeof(double)); // in case they get overwritten

    // Other Variables
    double AConditionNumber;
    int INFO, IPIV[n];
    //int recursion_limit = 64;

    // READ Matrix A and B from arguments and FILES
    Read_Matrix_file(matrixA, n*n, argv[1]);
    Read_Matrix_file(matrixB, n*n, argv[2]);
    
    // Backup A and B Matrices
    Write_A_over_B(matrixA,matrixA_original,n,n);
    Write_A_over_B(matrixB,matrixB_original,n,n);

    // ----------------- Calculate Condition Number of Matrix A

    AConditionNumber = ConditionNumber(matrixA,n,n);
    cout << "\nMatrix A Condition Number: " << (AConditionNumber) << "\n";
    
    // ----------------- Start PIVOTED Algorithm HERE!

    start = std::chrono::high_resolution_clock::now();
    // Recursive Implementation of LU decomposition for PA -> PIVOTED
    LUdecompositionRecursive4Pivot(matrixA, matrixL, matrixU, matrixP, n);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);  
    elapsed_time_Pivot = duration.count() * 1.e-9;

    // Now use BPivot instead of B for Solving LUX=PB -> PAX=PB -> PA=LU
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, matrixP, n, matrixB, n, 0.0, matrixBPivot, n);
    
    // Now Solve system of linear equations given AX=B given B is n x n
    // Solve PAX=PB -> LUX=BPivot -> (2) UX=Y -> (1) LY=BPivot
    // Solve (1) LY=BPivot
    start = std::chrono::high_resolution_clock::now();
    LowerTriangularSolverRecursiveReal_0(matrixL,matrixBPivot,matrixY,n,n);
    // Solve (2) UX=Y   
    UpperTriangularSolverRecursiveReal_0(matrixU,matrixY,matrixX,n,n);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);  
    elapsed_time_Solve = duration.count() * 1.e-9;

    cout << "\nCheck Accuracy and time of my AX=B (Pivoted):";
    ErrorCalc_Display(matrixA_original,matrixB_original, matrixX, elapsed_time_Pivot+elapsed_time_Solve,n,n);
    
    // Restore A and B Matrices After Calculation
    Write_A_over_B(matrixA_original,matrixA,n,n);
    Write_A_over_B(matrixB_original,matrixB,n,n); 

    // ----------------- Start Non-PIVOTED Algorithm HERE!
    // Reset Result Matrices X and Y
    MakeZeroes(matrixY,n,n);
    MakeZeroes(matrixX,n,n);
    
    start = std::chrono::high_resolution_clock::now();
    // Recursive Implementation of LU decomposition for A -> NON - PIVOTED
    LUdecompositionRecursive2(matrixA, matrixL, matrixU, n, n);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);  
    elapsed_time_nonPivot = duration.count() * 1.e-9;
    
    // Now Solve system of linear equations given AX=B given B is n x n
    // Solve AX=B -> LUX=B -> (2) UX=Y -> (1) LY=B
    // Solve (1) LY=B
    start = std::chrono::high_resolution_clock::now();
    LowerTriangularSolverRecursiveReal_0(matrixL,matrixB,matrixY,n,n);
    // Solve (2) UX=Y
    UpperTriangularSolverRecursiveReal_0(matrixU,matrixY,matrixX,n,n);
    stop = std::chrono::high_resolution_clock::now();

    cout << "Check Accuracy and time of my AX=B (non-Pivoted):";
    ErrorCalc_Display(matrixA_original,matrixB_original, matrixX, elapsed_time_nonPivot+elapsed_time_Solve,n,n);

    // Restore A and B Matrices After Calculation
    Write_A_over_B(matrixA_original,matrixA,n,n);
    Write_A_over_B(matrixB_original,matrixB,n,n);

    // ---------------- Done! Now to Show the Results and Compare with BLAS

    //   Solve BLAS and compare with my implementation
    start = std::chrono::high_resolution_clock::now();
    LAPACK_dgesv(&n,&n,matrixA,&n,IPIV,matrixB,&n,&INFO);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);  
    elapsed_time_BLAS = duration.count() * 1.e-9;
    cout << "Check Accuracy and time of BLAS (dgesv): ";
    ErrorCalc_Display(matrixA_original, matrixB_original, matrixB, elapsed_time_BLAS, n, n);
    
    cout << "Pivot LU decomposition: " << (elapsed_time_Pivot) << " s.\n";
    cout << "Non-Pivot LU decomposition: " << (elapsed_time_nonPivot) << " s.\n";
    cout << "Lower + Upper Solve: " << (elapsed_time_Solve) << " s.\n\n";
        
    cout << "Solution Calculation Speedup from BLAS to my_Pivot: " << (elapsed_time_Pivot+elapsed_time_Solve)/elapsed_time_BLAS << "x.\n\n";
    cout << "Solution Calculation Speedup from BLAS to my_nonPivot: " << (elapsed_time_nonPivot+elapsed_time_Solve)/elapsed_time_BLAS << "x.\n\n";

    return 0;
}
