#include <cblas.h>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>
#include <pthread.h>
#include <thread>

using namespace std;

int recursion_count = 0;

// 	Problem:
//  0. Read Random matrix A and B
//  0.1 Naive implementation of ALL
//  0.2 Evaluate Error and Time

//  1.Random Matrix -> get a LU decomposition recursively
//  2.Solve with LUX=B -> LY=B -> when you solve for Y -> UX=Y -> Solve  for X
//  4.Implement PIVOT and see change in error
//  5.Implement threading? -> See change in timing

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
void ErrorCalc_Display(double *matrixA, double *matrixB, double *matrixX, long double elapsed_time,int m, int n)
{
    double *CalculatedB = (double *)malloc(m * n * sizeof(double));   
    MakeZeroes(CalculatedB,m,n);
    NaiveMatrixMultiplyCol(matrixA, matrixX, CalculatedB, m, m, n);
    double diff = MatrixAbsDiff(matrixB, CalculatedB, m, n);

    cout << "\nError (AX - B)-----------------> : " << diff << "\n";
    std::cout << "Elapsed Time----------------> : " << elapsed_time << " s.\n\n";
    free(CalculatedB);
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


///     MAIN :: For the sake of simplicity we will Consider all square matrices n x n

int main ( int argc, char* argv[] ) {
    
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <filename A> <filename B> rank " << std::endl;
        return 1;
    }
    const int n = std::atoi(argv[3]); // rank
    // Alloc Space for Matrices Needed in Column Major Order
    double *matrixA = (double *)malloc(n * n * sizeof(double));
    double *matrixB = (double *)malloc(n * n * sizeof(double));
    double *matrixLU = (double *)malloc(n * n * sizeof(double));
    double *matrixL = (double *)malloc(n * n * sizeof(double));
    double *matrixU = (double *)malloc(n * n * sizeof(double));
    double *matrixA_original = (double *)malloc(n * n * sizeof(double)); // in case they get overwritten
    int recursion_limit = 64;
    // Read Matrix A and B from arguments and files
    Read_Matrix_file(matrixA, n*n, argv[1]);
    Read_Matrix_file(matrixB, n*n, argv[2]);

    cout << "Matrix A:\n";
    PrintColMatrix(matrixA,n,n);
    Write_A_over_B(matrixA,matrixA_original,n,n);

    // Recursive Implementation of LU decomposition for A
    LUdecompositionRecursive2(matrixA, matrixL, matrixU, n, n);
    // We might need LUdecompositionRecursive to have JUST one Matrix LU
    
    MMMultRecursive3Threaded(matrixL,n,0,0 ,matrixU,n,0,0, matrixLU,n,0,0,n,n,n,recursion_limit);
    cout << "\nDiff between A and L*U:" << (MatrixAbsDiff(matrixA_original,matrixLU,n,n)) << "\n";

    //Now Solve system of linear equations given AX=B given B is n x n
    // Solve AX=B -> LUX=B -> (2) UX=Y -> (1) LY=B

    // Solve (1) LY=B

    // Solve (2) Ux=y

    
    return 0;
}
