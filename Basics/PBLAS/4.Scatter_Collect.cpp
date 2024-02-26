#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <cblas.h>
#include <cmath>
#include "scalapack.h"

#define COL_MAJOR_INDEX(row, col, numRows) (row + col * numRows)

using namespace std;

void PrintColMatrix(double *matrix, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (matrix[i + (j * m)] < 0.00000000000000000001)
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
void printMatrix(double *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void ScatterMatrix(int context, double *globalMatrix, int M, int N, int Mb, int Nb, double *localMatrix, int localrows, int localcols, int myprow, int mypcol, int nprow, int npcol)
{
    // Acknowledgements to Matt Hurliman https://github.com/mhurliman
    // https://github.com/mhurliman/uw-capstone/blob/57362cb207f58b6463d21f4e4decb8c759826d69/utilities/DistributedMatrix.inl#L19
    // Pg Dims = process grid dimensions (npcol nprows)
    // desc is matrix descriptor PROBLEM parameters with context,M,N,Mb,Nb
    // Global is location of global matrix/// Local is location of local matrix
    // localDim (localrows, localcols) is local dimension of the destination local matrix , that is calculated as M x N divided into pxq blocks.....!!!!
    // id is current process identifier id.row// id.col  // id.isRoot // it is the ip x iq ->myprow,mypcol

    // This is sending BLOCKS of nr x nc from ROOT global matrix to sendr,sendc process as receivers of the nr x nc BLOCKS
    // nr x nc can be MbXNb as maximum value OR the remainder which will be a smaller block (M-r) x (N-c)
    // In Ken PDF example we talk about 3x2 blocks of CONTIGUOS Data in Block cyclic
    // r and c are the GLOBAL indeces of the upper right corner of the (nr x nc) block to move
    // sendR and sendC are current process ip and iq that can come back again and again in this loop (mod processdim size)
    // recvr and recvc are the element in local matrix where the block will be written

    // This has to be done in conjuction with the size of the process grid in example 2x3 (pxq) in this code pgdims.row x pgdims.col
    // (nprow,npcol)

    int sendr = 0;
    int sendc = 0;
    int recvr = 0;
    int recvc = 0;

    for (int r = 0; r < M; r += Mb, sendr = (sendr + 1) % nprow) // go jumping global matrix block by block (row)
    {
        sendc = 0;

        int nr = std::min(Mb, M - r); // nr is row block size or smaller remainder size

        for (int c = 0; c < N; c += Nb, sendc = (sendc + 1) % npcol) // go jumping global matrix block by block (col)
        {
            int nc = std::min(Nb, N - c); // nc is col block size or smaller remainder size

            if (myprow == 0 && mypcol == 0) // if rank = 0 , root process. send from global matrix to all other processes (sendr,sendc)
            {
                // void Cdgesd2d(int context, int M, int N, const double* A, int lda, int rdest, int cdest);
                Cdgesd2d(context, nr, nc, &globalMatrix[COL_MAJOR_INDEX(r, c, M)], M, sendr, sendc); // sendr,sendc are processes to send to
            }

            if (myprow == sendr && mypcol == sendc) // if caller to scatter is the corresponding process  (sendr,sendc) then receive
            {
                // void Cdgerv2d(int context, int M, int N, double* A, int lda, int rsrc, int csrc);
                Cdgerv2d(context, nr, nc, &localMatrix[COL_MAJOR_INDEX(recvr, recvc, localrows)], localrows, 0, 0); // receive from root 0,0
                                                                                                                    // upper left corner of receiving local matrix is 0,0 and will move 1 block....
                recvc = (recvc + nc)% localcols;                                                                   // if there is another block to be written in the same process grid,...
                                                                                                                    //  move the receiving upper left corner by a block
                                                                                                                    // WHY % localDims.col is needed here??? -> IT IS NEEDED
                                                                                                                    // because there might be more rows, hence col has to reset to 0
            }
        }

        if (myprow == sendr) // if caller to scatter is in the row of the corresponding process, move 1 block down
                             // for local matrix to receive properly
        {
            recvr = (recvr + nr)% localrows;     // WHY % localrows is needed here???? -> It is not really needed here
        }
    }
}

void CollectMatrix(int context, double *globalMatrix, int M, int N, int Mb, int Nb, double *localMatrix, int localrows, int localcols, int myprow, int mypcol, int nprow, int npcol)
{
    int sendr = 0;
    int sendc = 0;
    int recvr = 0;
    int recvc = 0;

    for (int r = 0; r < M; r += Mb, sendr = (sendr + 1) % nprow)
    {
        sendc = 0;

        int nr = std::min(Mb, M - r);

        for (int c = 0; c < N; c += Nb, sendc = (sendc + 1) % npcol)
        {
            int nc = std::min(Nb, N - c);

            if (myprow == sendr && mypcol == sendc)
            {
                Cdgesd2d(context, nr, nc, &localMatrix[COL_MAJOR_INDEX(recvr, recvc, localrows)], localrows, 0, 0);
                recvc = (recvc + nc)% localcols;
            }

            if (myprow == 0 && mypcol == 0)
            {
                Cdgerv2d(context, nr, nc, &globalMatrix[COL_MAJOR_INDEX(r, c, M)], M, sendr, sendc);
            }
        }

        if (myprow == sendr)
            recvr = (recvr + nr)% localrows;
    }
}

// ***********************  TEST TO SCATTER MATRIX

int main(int argc, char **argv)
{
     if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << "N (Dimension of Matrix)  NB (Dimension of Block) (dime) what procesor to check locally:"  << std::endl; 
        return 1;
    }
    // Define matrix size and block size
    int N = std::atoi(argv[1]);  // Matrix size (N x N)
    int M = N;
    int NB = std::atoi(argv[2]); // Matrix block (NB x NB)
    int MB = NB;
    int dime = std::atoi(argv[3]); 
    int zero = 0;
    char ColMajor[10] = "Col-major";

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize BLACS context
    int context;
    Cblacs_pinfo(&rank, &size);
    Cblacs_get(-1, 0, &context);

    // Number of rows and columns in the process grid
    int nprow;
    int npcol;
    nprow = static_cast<int>(sqrt(size));
    npcol = size / nprow;

    // Process grid coordinates
    int myprow, mypcol; // current process grid coordinates
    Cblacs_gridinit(&context, ColMajor, nprow, npcol);
    Cblacs_gridinfo(context, &nprow, &npcol, &myprow, &mypcol);

    // Local matrix dimensions
    int localrows = numroc_(&M, &MB, &myprow, &zero, &nprow);
    int localcols = numroc_(&N, &NB, &mypcol, &zero, &npcol);

    // Allocate memory for the local portion of the matrix
    double *A_local = new double[localrows * localcols];

    // Allocate memory for the global matrix on root process

    double *A_global = nullptr;
    double *Collect_global = nullptr;

    if (rank == 0)
    {
        // Reserve Memory only in ROOT Process
        A_global = new double[M * N];
        Collect_global = new double[M * N];

        // Initialize the global matrix (for demonstration)
        for (int i = 0; i < M * N; ++i)
        {
            A_global[i] = i + 1; // Each element is set to its index + 1
        }
        std::cout << "Global Matrix:" << std::endl;
        PrintColMatrix(A_global, M, N);
    }

    //cout << "\n Memory in Global Address space:" << &A_global[0]; // There is a global matrix reserved for every procesor but only Rank0 uses it
    // NON EFFICIENT!!!!!!!!!!!!!!!!! 
    // (BUUUUT you can only reserve memory in root!!!!!!!!!!!)

    // Scatter the global matrix to the grid
    ScatterMatrix(context, A_global, M, N, MB, NB, A_local, localrows, localcols, myprow, mypcol, nprow, npcol);

    if (rank == dime)
    {
        cout << "\nIn RANK: " << rank << " , nprow:" << nprow << " npcol:" << npcol << ":: Also :: localrows:" << localrows << " and localcols:" << localcols << " myrow:" << myprow << ", mycol:" << mypcol << " \n";
        std::cout << "Scattered Local Matrix:" << std::endl;
        PrintColMatrix(A_local, localrows, localcols);
    }

    CollectMatrix(context, Collect_global, M, N, MB, NB, A_local, localrows, localcols, myprow, mypcol, nprow, npcol);
    if (rank == 0)
    {
        std::cout << "Collected Matrix:" << std::endl;
        PrintColMatrix(Collect_global, M, N);
    }

    // Clean up
    delete[] A_global;
    delete[] Collect_global;
    delete[] A_local;
    Cblacs_gridexit(context);
    MPI_Finalize();

    return 0;
}
