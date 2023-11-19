#include <cblas.h>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

// 	Problem , get a recursive algorithm for Matrix Matrix Multiplication of any size 
//  Matrices given in COLUMN major order : A (m,n) x B (n,p) = C (m,p)
//  Algorithm to be done in COLUMN Major 
//  USE POINTERS in this exercise double *C = (double *)malloc(m * n * sizeof(double));
//  free to avoid Memory leaks!!!!!!

/*
    //THE OPTIMAL approach would be to transpose A so rows are adjecent in A*B multiplication
    //We will forget about this fact for now and focus in the recursive aspect
    //Later we can analyze this further optimization

    ColMajor_Transpose(matrixA,m,n);
    cout << "\nMatrix A traspose Print:\n";
    PrintColMatrix(matrixA,n,m);
    ColMajor_Transpose(matrixA,n,m);
    cout << "\nMatrix A trasposed again to original format:\n";
    PrintColMatrix(matrixA,m,n);
*/

int recursion_count = 0;

/////////////////////////////     FUNCTIONS

void PrintRowMatrix(double *matrix, int m, int n)
{
    for (int i = 0; i < m * n; i++)
    {
        cout << matrix[i] << " ";
        if (i%n==n-1) {
            cout << "\n";
        }
    }  
}

void PrintColMatrix(double *matrix, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << matrix[i+(j*m)] << " ";
        }
        cout << "\n";
    }
}

void NaiveMatrixMultiplyCol(double *matrixa, double *matrixb, double *matrixc, int m, int n,int p)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                matrixc[i+(j*m)] += matrixa[i+(k*m)]*matrixb[k+(j*n)];
            }
        }
    }
}

double MatrixAbsDiff(double *matrixa, double *matrixb, int m, int p)
{
    double diff= 0.0;
    for (int i = 0; i < m*p; ++i)
    {
        diff += abs(matrixa[i] - matrixb[i]);
    }
    return diff;
}

void ColMajor_Transpose(double *matrix, int m, int n)
{
    double *tmpmatrix = (double *)malloc(m * n * sizeof(double));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            tmpmatrix[j+(i*n)] = matrix[i+(j*m)];
        }
    }
    for (int i = 0; i < m*n; i++)
    {
        matrix[i] = tmpmatrix[i];
    }
    free(tmpmatrix);
}

void MakeZeroes(double *matrix, int m, int n)
{
    for (int i = 0; i < m * n; i++)
    {
        matrix[i] = 0.0;
    }
}

//Collect will create the big matrix based on submatrices
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
    // CRFEATE this algo from scratch
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

void MMMultRecursive(double *matrixa, double *matrixb, double *matrixc, int m, int n, int p, int recursion_limit)
{
    recursion_count++;
    // if any of those is n=1 STOP??  Where to break recurse????
    // if n intermediate is small enough? if n==1? Break recurse??
    // not really!!! if m or n or p is 1 you HAVE to break recurse.
    // Then ->  do calculations by hand C = C + A*B
    // else COMPLETE RECURSE BLOCK

    if (m<=recursion_limit or n<=recursion_limit or p<=recursion_limit)
    {
        NaiveMatrixMultiplyCol(matrixa, matrixb, matrixc,m,n,p);        
    }
    else
    {
        int mm = m / 2;
        int mm2 = m - mm;
        int nn = n / 2;
        int nn2 = n - nn;
        int pp = p / 2;
        int pp2 = p - pp;
        
        double *A11 = (double *)malloc(mm * nn * sizeof(double));
        MakeZeroes(A11, mm, nn);
        double *A12 = (double *)malloc(mm * nn2 * sizeof(double));
        MakeZeroes(A12, mm, nn2);
        double *A21 = (double *)malloc(mm2 * nn * sizeof(double));
        MakeZeroes(A21, mm2, nn);
        double *A22 = (double *)malloc(mm2 * nn2 * sizeof(double));
        MakeZeroes(A22, mm2, nn2);
        double *B11 = (double *)malloc(nn * pp * sizeof(double));
        MakeZeroes(B11, nn, pp);
        double *B12 = (double *)malloc(nn * pp2 * sizeof(double));
        MakeZeroes(B12, nn, pp2);
        double *B21 = (double *)malloc(nn2 * pp * sizeof(double));
        MakeZeroes(B21, nn2, pp);
        double *B22 = (double *)malloc(nn2 * pp2 * sizeof(double));
        MakeZeroes(B22, nn2, pp2);
        double *C11 = (double *)malloc(mm * pp * sizeof(double));
        MakeZeroes(C11, mm, pp);
        double *C12 = (double *)malloc(mm * pp2 * sizeof(double));
        MakeZeroes(C12, mm, pp2);
        double *C21 = (double *)malloc(mm2 * pp * sizeof(double));
        MakeZeroes(C21, mm2, pp);
        double *C22 = (double *)malloc(mm2 * pp2 * sizeof(double));
        MakeZeroes(C22, mm2, pp2);

        // Initializa Axx and Bxx matrices!
        InitializeSubmatrices(matrixa, A11, A12, A21, A22, m,n);
        InitializeSubmatrices(matrixb, B11, B12, B21, B22, n,p);

        // C11 is recurse1 and recurse2
        MMMultRecursive(A11, B11, C11, mm, nn, pp,recursion_limit);
        MMMultRecursive(A12, B21, C11, mm, nn2, pp,recursion_limit);
                
        // C12 is recurse3 and recurse4
        MMMultRecursive(A11, B12, C12, mm,nn,pp2,recursion_limit);
        MMMultRecursive(A12, B22, C12, mm,nn2,pp2,recursion_limit);
        // C21 is recurse5 and recurse6
        MMMultRecursive(A21, B11, C21, mm2,nn,pp,recursion_limit);
        MMMultRecursive(A22, B21, C21, mm2,nn2,pp,recursion_limit);
        // C22 is recurse7 and recurse8
        MMMultRecursive(A21, B12, C22, mm2,nn,pp2,recursion_limit);
        MMMultRecursive(A22, B22, C22, mm2,nn2,pp2,recursion_limit);
        
        // At the end Collect pieces of matrixc = matrixc + C11 + C12 + C21 + C22 and done!
        CollectSubmatrices(matrixc,C11,C12,C21,C22,m,p);
        
        free(A11);
        free(A12);
        free(A21);
        free(A22);
        free(B11);
        free(B12);
        free(B21);
        free(B22);
        free(C11);
        free(C12);
        free(C21);
        free(C22);
    }
}

/////////////////////////////     MAIN
int main ( int argc, char* argv[] ) {

    if (argc != 7)
    {
        std::cerr << "Usage: " << argv[0] << " <filenameA> <filenameB> m n p recursion_limit" << std::endl;
        return 1;
    }
    
    const int m = std::atoi(argv[3]); // m
    const int n = std::atoi(argv[4]); // n
    const int p = std::atoi(argv[5]); // p
    const int recursion_limit = std::atoi(argv[6]); // depth of recursion
    
    // Create a vector to store the Column Major matrix data and another for the Row Major
    double *matrixA = (double *)malloc(m * n * sizeof(double));
    double *matrixB = (double *)malloc(n * p * sizeof(double));
    double *matrixC = (double *)malloc(m * p * sizeof(double));
    double *matrixC_test = (double *)malloc(m * p * sizeof(double));

    // Open the binary file for reading and handle error
    std::ifstream inputA(argv[1], std::ios::binary);
    std::ifstream inputB(argv[2], std::ios::binary);
    std::ifstream input("matrix_random.bin", std::ios::binary);
    if (!inputA or !inputB){std::cerr << "Error: could not open file for reading" << std::endl; return 1;}
    // Read the binary data into the vector
    inputA.read(reinterpret_cast<char *>(matrixA), sizeof(double) * m * n);
    inputB.read(reinterpret_cast<char *>(matrixB), sizeof(double) * n * p);
    // Check if read was successful and handle error 
    if (!inputA or !inputB) {std::cerr << "Error: could not read file" << std::endl; return 1;}
    
    // Print the matrix elements
    cout << "\nMatrix A Print:\n";
    PrintColMatrix(matrixA,m,n);
    cout << "\nMatrix B Print:\n";
    PrintColMatrix(matrixB,n,p);
    
    // Multiply
    MMMultRecursive(matrixA,matrixB,matrixC,m,n,p,recursion_limit);

    cout << "\nMatrix C Recursive:\n";
    PrintColMatrix(matrixC,m,p);
   
    // Naive Calculation of product to compare with recursive version
    NaiveMatrixMultiplyCol(matrixA,matrixB,matrixC_test,m,n,p);
    //cout << "\nMatrix Ctest Naive:" << std::endl;
    //PrintColMatrix (matrixC_test,m,p);

    double diff=MatrixAbsDiff(matrixC_test,matrixC,m,p);

    cout << "\nABS (C - Ctest)-----------------> : " << diff << "\n";
    cout << "\nRecursion Count -----------------> : " << recursion_count << "\n";

    free(matrixA);
    free(matrixB);
    free(matrixC);
    free(matrixC_test);
    return 0;
}
