#include <cblas.h>
#include <lapacke.h>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>
#include <pthread.h>
#include <thread>
#include <lapacke.h>

using namespace std;

// Problem , get a recursive algorithm for a Triangular solver that does the last step after LU factorization
// Ax=b -> LUx=b -> Ux=y // Ly=b //-> solve y from Ly=b and then solve x from Ux=y
// Case 1) left lower // Case 2) left upper
// then calculate distance (error from Ax to b)
// Compare Timing with dgesv (double real)/ zgesv (complex)
// SCOPE for REAL (double) First and leave COMPLEX case for next iteration

int recursion_count = 0;

/////////////////////////////     FUNCTIONS

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

void NaiveMatrixMultiplyCol(double *matrixa, double *matrixb, double *matrixc, int m, int n, int p)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                matrixc[i + (j * m)] += matrixa[i + (k * m)] * matrixb[k + (j * n)];
            }
        }
    }
}

void MakeZeroes(double *matrix, int m, int n)
{
    for (int i = 0; i < m * n; i++)
    {
        matrix[i] = 0.0;
    }
}

double MatrixAbsDiff(double *matrixa, double *matrixb, int m, int p)
{
    double diff = 0.0;
    for (int i = 0; i < m * p; ++i)
    {
        diff += abs(matrixa[i] - matrixb[i]);
    }
    return diff;
}

//Collect Results of Xxx to the big matrix X
void CollectSubmatrices(double *matrixc, double *C11, double *C12, double *C21, double *C22, int m, int p)
{
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

//Initialize will Create the submatrices based on the big matrix
void InitializeSubmatrices(double *matrixc, double *C11, double *C12, double *C21, double *C22, int m, int p)
{
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

void NaiveMatrixMultiplyColReferenced2(double *matrixa, int matrixa_m, int a_m1, int a_n1 ,double *matrixb, int matrixb_n, int b_n1, int b_p1, double *matrixc,int matrixc_m, int c_m1, int c_p1, int m, int n, int p)
{
    // This commented line will help debug
    // cout<<"\n SubMatrix A " << a_m1 << "," << a_n1 <<"  SubMatrix B " << b_n1 << "," << b_p1 <<"\n";
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                matrixc[c_m1 + i + ((c_p1+j)*matrixc_m)] += (matrixa[a_m1 + i +(a_n1+k)*matrixa_m])*(matrixb[b_n1 + k +(b_p1+j)*matrixb_n]); 
                // A(i,k)*B(k,j)
            }
        }
    }
}

void MMMultRecursive3(double *matrixa, int matrixa_m, int a_m1, int a_n1, double *matrixb, int matrixb_n, int b_n1, int b_p1, double *matrixc, int matrixc_m, int c_m1, int c_p1, int m, int n, int p, int recursion_limit)
{
    recursion_count++;
    // This Version will NOT Malloc Cxx Either and will write C elements on proper places

    if (m <= recursion_limit or n <= recursion_limit or p <= recursion_limit)
    {
        NaiveMatrixMultiplyColReferenced2(matrixa, matrixa_m, a_m1, a_n1, matrixb, matrixb_n, b_n1, b_p1, matrixc, matrixc_m, c_m1, c_p1, m, n, p);
    }
    else
    {
        int mm = m / 2;
        int mm2 = m - mm; // If odd numbers then mm != mm2 so we keep track of the fact
        int nn = n / 2;
        int nn2 = n - nn;
        int pp = p / 2;
        int pp2 = p - pp;

        // C11 is recurse1 and recurse2
        MMMultRecursive3(matrixa, matrixa_m, a_m1, a_n1, matrixb, matrixb_n, b_n1, b_p1, matrixc, matrixc_m, c_m1, c_p1, mm, nn, pp, recursion_limit);
        MMMultRecursive3(matrixa, matrixa_m, a_m1, a_n1 + nn, matrixb, matrixb_n, b_n1 + nn, b_p1, matrixc, matrixc_m, c_m1, c_p1, mm, nn2, pp, recursion_limit);

        // C12 is recurse3 and recurse4
        MMMultRecursive3(matrixa, matrixa_m, a_m1, a_n1, matrixb, matrixb_n, b_n1, b_p1 + pp, matrixc, matrixc_m, c_m1, c_p1 + pp, mm, nn, pp2, recursion_limit);
        MMMultRecursive3(matrixa, matrixa_m, a_m1, a_n1 + nn, matrixb, matrixb_n, b_n1 + nn, b_p1 + pp, matrixc, matrixc_m, c_m1, c_p1 + pp, mm, nn2, pp2, recursion_limit);

        // C21 is recurse5 and recurse6
        MMMultRecursive3(matrixa, matrixa_m, a_m1 + mm, a_n1, matrixb, matrixb_n, b_n1, b_p1, matrixc, matrixc_m, c_m1 + mm, c_p1, mm2, nn, pp, recursion_limit);
        MMMultRecursive3(matrixa, matrixa_m, a_m1 + mm, a_n1 + nn, matrixb, matrixb_n, b_n1 + nn, b_p1, matrixc, matrixc_m, c_m1 + mm, c_p1, mm2, nn2, pp, recursion_limit);

        // C22 is recurse7 and recurse8
        MMMultRecursive3(matrixa, matrixa_m, a_m1 + mm, a_n1, matrixb, matrixb_n, b_n1, b_p1 + pp, matrixc, matrixc_m, c_m1 + mm, c_p1 + pp, mm2, nn, pp2, recursion_limit);
        MMMultRecursive3(matrixa, matrixa_m, a_m1 + mm, a_n1 + nn, matrixb, matrixb_n, b_n1 + nn, b_p1 + pp, matrixc, matrixc_m, c_m1 + mm, c_p1 + pp, mm2, nn2, pp2, recursion_limit);
    }
}

void LowerTriangularSolverNaiveReal(double *matrixL, double *matrixB, double *matrixSol, int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            double tempval = 0;
            for (int k = 0; k < i; k++)
            {
                tempval += matrixL[i + k * n] * matrixSol[k + j * n];
            }
            matrixSol[j * n + i] = (matrixB[j * n + i] - tempval) / matrixL[i * n + i];
        }
    }
}

void UpperTriangularSolverNaiveReal(double *matrixU, double *matrixB, double *matrixSol, int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = n - 1; i >= 0; i--)
        {
            double tempval = 0;
            for (int k = n - 1; k > i; k--)
            {
                tempval += matrixU[i + k * n] * matrixSol[k + j * n];
            }
            matrixSol[j * n + i] = (matrixB[j * n + i] - tempval) / matrixU[i * n + i];
        }
    }
}

// Get value (ColMajorFormat) from orinal matrix with n rows, submatrix (0,0) starts in (n1,p1), element i,j
double GetSubMatrixValue(double *matrixa,int n,int n1,int p1, int i, int j)
{
    return matrixa[n1 + i + (p1+j)*n];
}

void UpperTriangularSolverRecursiveReal(double *matrixU, int U_n1, int U_n2, double *matrixB, int B_n1, int B_p1, double *matrixSol, int major_n, int n, int p)
{
    recursion_count++;
    // cout << " This is the iteration " << n << " x " << p << "\n"; // This Line is for debugging
    // PHASE 1: RECURSE on calculations based on TRIANGULAR U22
    if (n == 1)
    {
        for (int j = 0; j < p; j++)
        {
            matrixSol[B_n1 + (B_p1+j)*major_n] = matrixB[B_n1 + (B_p1+j)*major_n]/matrixU[U_n1 + (U_n2)*major_n];
        }
    }
    else
    {
        int nn = n / 2;
        int nn2 = n - nn; // Size of right or lower side covers for odd cases 
        int pp = (p/2); 
        int pp2 = p - pp;

        // Recurse U22 X21 = B21
        UpperTriangularSolverRecursiveReal(matrixU,U_n1+nn,U_n2+nn,matrixB,B_n1+nn,B_p1,matrixSol,major_n,nn2,pp);
        // Recurse U22 X22 = B22
        UpperTriangularSolverRecursiveReal(matrixU,U_n1+nn,U_n2+nn,matrixB,B_n1+nn,B_p1+pp,matrixSol,major_n,nn2,pp2);
    
        // PHASE 2: CALCULATE THE NEW B's for next Phase     
        // B11' = B11 - U12 X21
        for (int i = 0; i < nn; ++i)
        {
            for (int j = 0; j < pp; ++j)
            {
                for (int k = 0; k < nn2; ++k)
                {
                    matrixB[B_n1 + i + ((B_p1 + j) * major_n)] -= (matrixU[U_n1 + i + (U_n2 + nn + k) * major_n]) * (matrixSol[B_n1 + nn + k + (B_p1 + j) * major_n]);
                    // A(i,k)*B(k,j)
                }
            }
        }
        // PHASE 3: RECURSE on REST of calculations with TRIANGULAR A22
        // Recurse U11 X11 = B11'
        UpperTriangularSolverRecursiveReal(matrixU,U_n1,U_n2,matrixB,B_n1,B_p1,matrixSol,major_n,nn,pp);

        // B12' = B12 - U12 X22
        for (int i = 0; i < nn; ++i)
        {
            for (int j = 0; j < pp2; ++j)
            {
                for (int k = 0; k < nn2; ++k)
                {
                    matrixB[B_n1 + i + ((B_p1 + pp + j) * major_n)] -= (matrixU[U_n1 + i + (U_n2 + nn + k) * major_n]) * (matrixSol[B_n1 + nn + k + (B_p1 + pp + j) * major_n]);
                    // A(i,k)*B(k,j)
                }
            }
        }
        // Recurse U11 X12 = B12'
        UpperTriangularSolverRecursiveReal(matrixU,U_n1,U_n2,matrixB,B_n1,B_p1+pp,matrixSol,major_n,nn,pp2);
    }   
}

void LowerTriangularSolverRecursiveReal(double *matrixL, int L_n1, int L_n2, double *matrixB, int B_n1, int B_p1, double *matrixSol, int major_n, int n, int p)
{
    recursion_count++;
    //cout << " This is the iteration " << n << " x " << p << "\n"; // This Line is for debugging
    // PHASE 1: RECURSE on calculations based on TRIANGULAR L11
    if (n == 1)
    {   
        for (int j = 0; j < p; j++)
        {
            matrixSol[B_n1 + (B_p1 + j)*major_n] = matrixB[B_n1 + (B_p1 + j)*major_n]/matrixL[L_n1 + L_n2*major_n];  
        }
    }
    else
    {
        int nn = n / 2;
        int nn2 = n - nn; // Size of right or lower side covers for odd cases 
        int pp = (p/2); 
        int pp2 = p - pp;
        
        // Recurse L11 X11 = B11
        LowerTriangularSolverRecursiveReal(matrixL,L_n1,L_n2,matrixB,B_n1,B_p1,matrixSol,major_n,nn,pp);
        // Recurse L11 X12 = B12
        LowerTriangularSolverRecursiveReal(matrixL,L_n1,L_n2,matrixB,B_n1,B_p1+pp,matrixSol,major_n,nn,pp2);
    
        // PHASE 2: CALCULATE THE NEW B's for next Phase     
        // B21' = B21 - L21 X11
        for (int i = 0; i < nn2; ++i)
        {
            for (int j = 0; j < pp; ++j)
            {
                for (int k = 0; k < nn; ++k)
                {
                    matrixB[B_n1 + nn + i + ((B_p1 + j) * major_n)] -= (matrixL[L_n1 + nn + i + (L_n2 + k) * major_n]) * (matrixSol[B_n1 + k + (B_p1 + j) * major_n]);
                    // A(i,k)*B(k,j)
                }
            }
        }
        // PHASE 3: RECURSE on REST of calculations with TRIANGULAR A22
        // Recurse L22 X21 = B21'
        LowerTriangularSolverRecursiveReal(matrixL,L_n1+nn,L_n2+nn,matrixB,B_n1+nn,B_p1,matrixSol,major_n,nn2,pp);

        // B22' = B22 - L21 X12
        for (int i = 0; i < nn2; ++i)
        {
            for (int j = 0; j < pp2; ++j)
            {
                for (int k = 0; k < nn; ++k)
                {
                    matrixB[B_n1 + nn + i + ((B_p1 + pp + j) * major_n)] -= (matrixL[L_n1 + nn + i + (L_n2 + k) * major_n]) * (matrixSol[B_n1 + k + (B_p1 + pp + j) * major_n]);
                    // A(i,k)*B(k,j)
                }
            }
        }
        // Recurse L22 X22 = B22'
        LowerTriangularSolverRecursiveReal(matrixL,L_n1+nn,L_n2+nn,matrixB,B_n1+nn,B_p1+pp,matrixSol,major_n,nn2,pp2);
    }   
}

void UpperTriangularSolverRecursiveReal_0(double *matrixU, double *matrixB, double *matrixX, int n, int p)
{
    // This is a Naive version with Malloc and free as crutch to avoid index calculation over the original matrix
    recursion_count++;
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
    recursion_count++;
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

void Rewrite_A_over_B(double *matrixA, double *matrixB, int n,int p)
{
    for (int i = 0; i < n * p; i++)
    {
        matrixB[i] = matrixA[i];
    }
}

/////////////////////////////     MAIN
int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <filenameL> <filenameU> <filenameB> n " << std::endl;
        return 1;
    }

    const int n = std::atoi(argv[4]);               // n
    const int p = n;                                // Note we should allow P to be higher or lower than n!!!! (LATER)

    // Timers
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto stop1 = std::chrono::high_resolution_clock::now();
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    // Input Matrices in Column Major Format
    double *matrixL = (double *)malloc(n * n * sizeof(double));
    double *matrixU = (double *)malloc(n * n * sizeof(double));
    double *matrixB = (double *)malloc(n * n * sizeof(double));
    double *matrixB2 = (double *)malloc(n * n * sizeof(double));
    double *matrixB_orig = (double *)malloc(n * n * sizeof(double));
    double *matrixL_orig = (double *)malloc(n * n * sizeof(double));
    double *matrixU_orig = (double *)malloc(n * n * sizeof(double));
    // Solution Matrices
    double *LowerMatrixXsolNaive = (double *)malloc(n * n * sizeof(double));
    double *LowerMatrixXsol = (double *)malloc(n * n * sizeof(double));
    double *UpperMatrixXsolNaive = (double *)malloc(n * n * sizeof(double));
    double *UpperMatrixXsol = (double *)malloc(n * n * sizeof(double));
    double *LowerMatrixXsol2 = (double *)malloc(n * n * sizeof(double));
    double *UpperMatrixXsol2 = (double *)malloc(n * n * sizeof(double));
    double *LowerMatrixBLASsol = (double *)malloc(n * n * sizeof(double));
    double *UpperMatrixBLASsol = (double *)malloc(n * n * sizeof(double));
    // Matrices to Calculate Error
    double *LowerCalculatedBNaive = (double *)malloc(n * n * sizeof(double));
    double *UpperCalculatedBNaive = (double *)malloc(n * n * sizeof(double));
    double *LowerCalculatedB = (double *)malloc(n * n * sizeof(double));
    double *UpperCalculatedB = (double *)malloc(n * n * sizeof(double));
    double *LowerCalculatedB2 = (double *)malloc(n * n * sizeof(double));
    double *UpperCalculatedB2 = (double *)malloc(n * n * sizeof(double));
    double *LowerCalculatedBBLAS = (double *)malloc(n * n * sizeof(double));
    double *UpperCalculatedBBLAS = (double *)malloc(n * n * sizeof(double));

    // Open the binary file for reading INPUT Matrices and handle error
    std::ifstream inputA(argv[1], std::ios::binary);
    std::ifstream inputB(argv[2], std::ios::binary);
    std::ifstream inputC(argv[3], std::ios::binary);
    if (!inputA or !inputB or !inputC)
    {
        std::cerr << "Error: could not open file for reading" << std::endl;
        return 1;
    }
    // Read the binary data into the vector
    inputA.read(reinterpret_cast<char *>(matrixL), sizeof(double) * n * n);
    inputB.read(reinterpret_cast<char *>(matrixU), sizeof(double) * n * n);
    inputC.read(reinterpret_cast<char *>(matrixB), sizeof(double) * n * n);
    // Check if read was successful and handle error
    if (!inputA or !inputB or !inputC)
    {
        std::cerr << "Error: could not read file" << std::endl;
        return 1;
    }

    Rewrite_A_over_B(matrixB, matrixB_orig, n, p);
    Rewrite_A_over_B(matrixB_orig, matrixB2, n, p);
    Rewrite_A_over_B(matrixL, matrixL_orig, n, p);
    Rewrite_A_over_B(matrixU, matrixU_orig, n, p);

    // Solve Naive
    start = std::chrono::high_resolution_clock::now();
    LowerTriangularSolverNaiveReal(matrixL, matrixB, LowerMatrixXsolNaive, n);
    UpperTriangularSolverNaiveReal(matrixU, matrixB, UpperMatrixXsolNaive, n);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    cout << "Naive Triangular Solver--------------------->: " << (duration.count() * 1.e-9) << " s.\n";

    // Solve Recursive NO MALLOC
    start = std::chrono::high_resolution_clock::now();
    LowerTriangularSolverRecursiveReal(matrixL,0,0,matrixB,0,0,LowerMatrixXsol,n,n,p);
    UpperTriangularSolverRecursiveReal(matrixU,0,0,matrixB2,0,0,UpperMatrixXsol,n,n,p);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    cout << "Recursive No MALLOC Triangular Solver------->: " << (duration.count() * 1.e-9) << " s.\n";

    //Restore
    Rewrite_A_over_B(matrixB_orig, matrixB, n, p);
    Rewrite_A_over_B(matrixB_orig, matrixB2, n, p);

    // Solve Recursive WITH MALLOC
    start = std::chrono::high_resolution_clock::now();
    LowerTriangularSolverRecursiveReal_0(matrixL,matrixB,LowerMatrixXsol2,n,p);
    UpperTriangularSolverRecursiveReal_0(matrixU,matrixB2,UpperMatrixXsol2,n,p);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    cout << "Recursive WITH MALLOC Triangular Solver----->: " << (duration.count() * 1.e-9) << " s.\n";

    //Restore
    Rewrite_A_over_B(matrixB_orig, matrixB, n, p);
    Rewrite_A_over_B(matrixB_orig, matrixB2, n, p);

    //Solve BLAS
    int INFO;
    int IPIV[n];
    LAPACK_dgesv(&n,&p,matrixL,&n,IPIV,matrixB,&n,&INFO);
    Rewrite_A_over_B(matrixB, LowerMatrixBLASsol, n, p);
    Rewrite_A_over_B(matrixB_orig, matrixB, n, p);
    int INFO2;
    int IPIV2[n];
    LAPACK_dgesv(&n,&p,matrixU,&n,IPIV2,matrixB,&n,&INFO2);
    Rewrite_A_over_B(matrixB, UpperMatrixBLASsol, n, p);
    Rewrite_A_over_B(matrixB_orig, matrixB, n, p);

    // PRINT NAIVE Solutions
    //cout << "\nLower Matrix X Solution Naive:\n";
    //PrintColMatrix(LowerMatrixXsolNaive,n,n);
    //cout << "\nUpper Matrix X Solution Naive:\n";
    //PrintColMatrix(UpperMatrixXsolNaive,n,n);

    // PRINT Recurse Solutions
    //cout << "\nLower Matrix X Solution Recurse:\n";
    //PrintColMatrix(LowerMatrixXsol,n,n);
    //cout << "\nUpper Matrix X Solution Recurse:\n";
    //PrintColMatrix(UpperMatrixXsol,n,n);
    
    // ERROR CALCULATION and display
    NaiveMatrixMultiplyCol(matrixL_orig, LowerMatrixXsolNaive, LowerCalculatedBNaive, n, n, n);
    NaiveMatrixMultiplyCol(matrixU_orig, UpperMatrixXsolNaive, UpperCalculatedBNaive, n, n, n);
    NaiveMatrixMultiplyCol(matrixL_orig, LowerMatrixXsol, LowerCalculatedB, n, n, n);
    NaiveMatrixMultiplyCol(matrixU_orig, UpperMatrixXsol, UpperCalculatedB, n, n, n);
    NaiveMatrixMultiplyCol(matrixL_orig, LowerMatrixXsol2, LowerCalculatedB2, n, n, n);
    NaiveMatrixMultiplyCol(matrixU_orig, UpperMatrixXsol2, UpperCalculatedB2, n, n, n);
    NaiveMatrixMultiplyCol(matrixL_orig, LowerMatrixBLASsol, LowerCalculatedBBLAS, n, n, n);
    NaiveMatrixMultiplyCol(matrixU_orig, UpperMatrixBLASsol, UpperCalculatedBBLAS, n, n, n);

    double Ldiffnaive = MatrixAbsDiff(matrixB_orig, LowerCalculatedBNaive, n, n);
    double Udiffnaive = MatrixAbsDiff(matrixB_orig, UpperCalculatedBNaive, n, n);
    double Ldiff = MatrixAbsDiff(matrixB_orig, LowerCalculatedB, n, n);
    double Udiff = MatrixAbsDiff(matrixB_orig, UpperCalculatedB, n, n);
    double Ldiff2 = MatrixAbsDiff(matrixB_orig, LowerCalculatedB2, n, n);
    double Udiff2 = MatrixAbsDiff(matrixB_orig, UpperCalculatedB2, n, n);
    double LdiffBLAS = MatrixAbsDiff(matrixB_orig, LowerCalculatedBBLAS, n, n);
    double UdiffBLAS = MatrixAbsDiff(matrixB_orig, UpperCalculatedBBLAS, n, n);

    cout << "\nNaive Error (LX - B)-----------------> : " << Ldiffnaive << "\n";
    cout << "Naive Error (UX - B)-----------------> : " << Udiffnaive << "\n";
    cout << "\nRecursive Error (LX - B)-----------------> : " << Ldiff << "\n";
    cout << "Recursive Error (UX - B)-----------------> : " << Udiff << "\n";
    cout << "\nRecursive Error Malloc (LX - B)----------> : " << Ldiff2 << "\n";
    cout << "Recursive Error Malloc (UX - B)-----------> : " << Udiff2 << "\n";
    cout << "\nBLAS Error (LX - B)-----------------> : " << LdiffBLAS << "\n";
    cout << "BLAS Error (UX - B)-----------------> : " << UdiffBLAS << "\n";
    cout << "\nRecursion Count -----------------> : " << recursion_count << "\n";

    //    Compare with Real solver from LAPACK LAPACK_dgesv();
    //    LAPACK_zgesv() is Complex Solver
    //    You can compare your results to the *getrf() LU based solver *gesv()

   return 0;
}
