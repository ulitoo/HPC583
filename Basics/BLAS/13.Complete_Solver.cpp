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
//  0. Read Random matrix A and B
//  1.Random Matrix -> get a LU decomposition recursively
//  2.Solve with LUX=B -> LY=B -> when you solve for Y -> UX=Y -> Solve  for X
//  3.Evaluate Error and Time
//  4.Implement PIVOTing and see change in error?
//  5.Implement threading? -> See change in timing

/////////////////////////////     FUNCTIONS
int recursion_count = 0;

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
    std::cout << "File " << filename << " read correctly!\n" << std::endl;
    return 0;
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
double MatrixAbsDiff(double *matrixa, double *matrixb, int m, int n)
{
    double diff = 0.0;
    for (int i = 0; i < m * n; ++i)
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
            tmpmatrix[j + (i * n)] = matrix[i + (j * m)];
        }
    }
    for (int i = 0; i < m * n; i++)
    {
        matrix[i] = tmpmatrix[i];
    }
    free(tmpmatrix);
}
void Write_A_over_B(double *matrixA, double *matrixB, int m,int n)
{
    for (int i = 0; i < m * n; i++)
    {
        matrixB[i] = matrixA[i];
    }
}
void SwapCol_ColMajMatrix(double *matrix,int from, int towards, int m, int n)
{
    double tmpval;
    for (int i = 0; i < m; i++)
    {
        tmpval = matrix[i+towards*m];
        matrix[i+towards*m] = matrix[i+from*m];
        matrix[i+from*m] = tmpval;
    }
}
void SwapRow_ColMajMatrix(double *matrix,int from, int towards, int m, int n)
{
    double tmpval;
    for (int i = 0; i < n; i++)
    {
        tmpval = matrix[towards+i*m];
        matrix[towards+i*m] = matrix[from+i*m];
        matrix[from+i*m] = tmpval;
    }
}
void TransposeColMajor(double *matrix, int m, int n)
{
    double tmpvalue;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < i; j++)
        {
            tmpvalue = matrix[j+(i*m)];
            matrix[j+(i*m)] = matrix[i+(j*m)];
            matrix[i+(j*m)] = tmpvalue;
        }
    }
}
void SchurComplement(double *matrix, int N, int n)
{
    int offset = (N - n);
    // ONLY 1 matrix, rewrite A22 as S22 ! ; N is the original A size ; n is the size of A in the recursion; n-1 is size of A22
    for (int i = 0; i < (n-1); i++)
    {
        for (int j = 0; j < (n-1); j++)
        {
            //This is in Column Major Order
            matrix[(i+offset+1)+((j+offset+1)*N)] = matrix[(i+offset+1)+((j+offset+1)*N)] - ((matrix[offset+(j+offset+1)*N]) * (matrix[i+offset+1+(offset*N)]) / matrix[(offset)+(offset)*N]) ;
        }   
    }
}
void LUdecompositionRecursive(double *matrix, double *LUmatrix, int N, int n)
{
    //Assume square Matrix for simplicity
    int offset = (N-n);

    LUmatrix[(offset)+(offset)*N]=matrix[(offset)+(offset)*N];
    
    for (int i = 1; i < n; i++)
    {
        LUmatrix[i+offset+(offset*N)] = matrix[i+offset+(offset*N)] / matrix[(offset)+(offset)*N] ;
        LUmatrix[offset+(i+offset)*N] = matrix[offset+(i+offset)*N];
    }
    
    if (n==2) 
    {
        LUmatrix[(offset+1)+(offset+1)*N] = matrix[(offset+1)+(offset+1)*N] - matrix[(offset+1)+(offset)*N]*matrix[(offset)+(offset+1)*N]/matrix[(offset)+(offset)*N];    
    }
    else
    {
        SchurComplement(matrix,N,n);
        LUdecompositionRecursive(matrix, LUmatrix, N, n-1);
    }
}
void LUdecompositionRecursive2(double *matrix, double *Lmatrix, double *Umatrix, int N, int n)
{
    //Assume square Matrix for simplicity
    int offset = (N-n);

    Umatrix[(offset)+(offset)*N] = matrix[(offset)+(offset)*N];
    Lmatrix[(offset)+(offset)*N] = 1.0;

    for (int i = 1; i < n; i++)
    {
        Lmatrix[i+offset+(offset*N)] = matrix[i+offset+(offset*N)] / matrix[(offset)+(offset)*N] ;
        Lmatrix[offset+(i+offset)*N] = 0.0; 
        Umatrix[offset+(i+offset)*N] = matrix[offset+(i+offset)*N];
        Umatrix[i+offset+(offset*N)] = 0.0;
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
void MMMultRecursive3Threaded(double *matrixa, int matrixa_m, int a_m1, int a_n1, double *matrixb, int matrixb_n, int b_n1, int b_p1, double *matrixc, int matrixc_m, int c_m1, int c_p1, int m, int n, int p, int recursion_limit)
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

        // C11 is recurse1 
        std::thread threadC11_1(MMMultRecursive3Threaded,matrixa, matrixa_m, a_m1, a_n1, matrixb, matrixb_n, b_n1, b_p1, matrixc, matrixc_m, c_m1, c_p1, mm, nn, pp, recursion_limit);
        // C12 is recurse3 
        std::thread threadC12_1(MMMultRecursive3Threaded,matrixa, matrixa_m, a_m1, a_n1, matrixb, matrixb_n, b_n1, b_p1 + pp, matrixc, matrixc_m, c_m1, c_p1 + pp, mm, nn, pp2, recursion_limit);
        // C21 is recurse5 
        std::thread threadC21_1(MMMultRecursive3Threaded,matrixa, matrixa_m, a_m1 + mm, a_n1, matrixb, matrixb_n, b_n1, b_p1, matrixc, matrixc_m, c_m1 + mm, c_p1, mm2, nn, pp, recursion_limit);
        // C22 is recurse7 
        std::thread threadC22_1(MMMultRecursive3Threaded,matrixa, matrixa_m, a_m1 + mm, a_n1, matrixb, matrixb_n, b_n1, b_p1 + pp, matrixc, matrixc_m, c_m1 + mm, c_p1 + pp, mm2, nn, pp2, recursion_limit);

        threadC11_1.join();
        threadC12_1.join();
        threadC21_1.join();
        threadC22_1.join();

        std::thread threadC11_2(MMMultRecursive3Threaded,matrixa, matrixa_m, a_m1, a_n1 + nn, matrixb, matrixb_n, b_n1 + nn, b_p1, matrixc, matrixc_m, c_m1, c_p1, mm, nn2, pp, recursion_limit);
        std::thread threadC12_2(MMMultRecursive3Threaded,matrixa, matrixa_m, a_m1, a_n1 + nn, matrixb, matrixb_n, b_n1 + nn, b_p1 + pp, matrixc, matrixc_m, c_m1, c_p1 + pp, mm, nn2, pp2, recursion_limit);
        std::thread threadC21_2(MMMultRecursive3Threaded,matrixa, matrixa_m, a_m1 + mm, a_n1 + nn, matrixb, matrixb_n, b_n1 + nn, b_p1, matrixc, matrixc_m, c_m1 + mm, c_p1, mm2, nn2, pp, recursion_limit);
        std::thread threadC22_2(MMMultRecursive3Threaded,matrixa, matrixa_m, a_m1 + mm, a_n1 + nn, matrixb, matrixb_n, b_n1 + nn, b_p1 + pp, matrixc, matrixc_m, c_m1 + mm, c_p1 + pp, mm2, nn2, pp2, recursion_limit);

        threadC11_2.join();
        threadC12_2.join();
        threadC21_2.join();
        threadC22_2.join();
    }
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
void ErrorCalc_Display(double *matrixA, double *matrixB, double *matrixX, long double elapsed_time,int n, int p, int recursion_limit)
{
    double *CalculatedB = (double *)malloc(n * p * sizeof(double));   
    MakeZeroes(CalculatedB,n,p);
    //NaiveMatrixMultiplyCol(matrixA, matrixX, CalculatedB, n, n, p);
    MMMultRecursive3Threaded(matrixA,n,0,0 ,matrixX,n,0,0, CalculatedB,n,0,0,n,n,n,recursion_limit);
    double diff = MatrixAbsDiff(matrixB, CalculatedB, n, p);

    cout << "\nError (AX - B):----------------> : " << diff << "\n";
    cout << "Elapsed Time:------------------> : " << elapsed_time << " s.\n\n";
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
    auto stop1 = std::chrono::high_resolution_clock::now();
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double Speedup;

    // Alloc Space for Matrices Needed in Column Major Order
    double *matrixA = (double *)malloc(n * n * sizeof(double));
    double *matrixB = (double *)malloc(n * n * sizeof(double));
    double *matrixL = (double *)malloc(n * n * sizeof(double));
    double *matrixU = (double *)malloc(n * n * sizeof(double));
    double *matrixY = (double *)malloc(n * n * sizeof(double));
    double *matrixX = (double *)malloc(n * n * sizeof(double));
    double *matrixB_Calc = (double *)malloc(n * n * sizeof(double));
    double *matrixA_original = (double *)malloc(n * n * sizeof(double)); // in case they get overwritten
    double *matrixB_original = (double *)malloc(n * n * sizeof(double)); // in case they get overwritten
    int recursion_limit = 64;

    // Read Matrix A and B from arguments and files
    Read_Matrix_file(matrixA, n*n, argv[1]);
    Read_Matrix_file(matrixB, n*n, argv[2]);
    
    // Backup
    Write_A_over_B(matrixA,matrixA_original,n,n);
    Write_A_over_B(matrixB,matrixB_original,n,n);
    
    // ----------------- Start Algorithm HERE!

    start = std::chrono::high_resolution_clock::now();
    // Recursive Implementation of LU decomposition for A
    LUdecompositionRecursive2(matrixA, matrixL, matrixU, n, n);
    // We might need LATER LUdecompositionRecursive to have JUST one Matrix LU
    stop1 = std::chrono::high_resolution_clock::now();
    
    // Now Solve system of linear equations given AX=B given B is n x n
    // Solve AX=B -> LUX=B -> (2) UX=Y -> (1) LY=B
    // Solve (1) LY=B
    LowerTriangularSolverRecursiveReal_0(matrixL,matrixB,matrixY,n,n);
    stop2 = std::chrono::high_resolution_clock::now();
    // Solve (2) UX=Y
    UpperTriangularSolverRecursiveReal_0(matrixU,matrixY,matrixX,n,n);
    stop = std::chrono::high_resolution_clock::now();
    
    // ---------------- Done! Now to Show the Results and Compare with BLAS

    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop1 - start);
    cout << "LU decomposition: " << (duration.count() * 1.e-9) << " s.\n";
    
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop2 - stop1);
    cout << "Lower Solve: " << (duration.count() * 1.e-9) << " s.\n";
    
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - stop2);
    cout << "Upper Solve: " << (duration.count() * 1.e-9) << " s.\n";
        
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    Speedup = duration.count();
    cout << "\nCheck Accuracy and time of my AX=B:";
    ErrorCalc_Display(matrixA_original,matrixB_original, matrixX, duration.count() * 1.e-9,n,n,recursion_limit);

    Write_A_over_B(matrixA_original,matrixA,n,n);
    Write_A_over_B(matrixB_original,matrixB,n,n);
    
    //Solve BLAS and compare with my implementation
    int INFO;
    int IPIV[n];
    start = std::chrono::high_resolution_clock::now();
    LAPACK_dgesv(&n,&n,matrixA,&n,IPIV,matrixB,&n,&INFO);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);  
    cout << "Check Accuracy and time of BLAS (dgesv): ";
    ErrorCalc_Display(matrixA_original, matrixB_original, matrixB, duration.count() * 1.e-9, n, n,recursion_limit);
    
    cout << "Solution Calculation Speedup from BLAS to mine: " << Speedup/duration.count() << "x.\n\n";

    return 0;
}
