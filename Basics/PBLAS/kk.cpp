#include <iostream>
#include <vector>
#include "scalapack.h" // Assuming you have the Scalapack header file

// REDISTRIBUTE MATRIXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX



      /* scatter of the matrix A from 1 processor to a P*Q grid        
         Cpdgemr2d(m, n,
                   Aseq, ia, ja, &descA_1x1,
                   Apar, ib, jb, &descA_PxQ, gcontext);     
       
      /* computation of the system solution 
         Cpdgesv( m, n, 
                  Apar , 1, 1, &descA_PxQ, ipiv , 
                  Cpar, 1, 1, &descC_PxQ, &info);
     
      /* gather of the solution matrix C on 1 processor 
         Cpdgemr2d(m, n,
                   Cpar, ia, ja, &descC_PxQ,
                   Cseq, ib, jb, &descC_1x1, gcontext);
*/


extern "C" {
    // External Scalapack functions
    extern void Cblacs_get(int*, int*, int*);
    extern void Cblacs_gridinit(int*, const char*, int, int);
    extern void Cblacs_gridinfo(int, int*, int*, int*, int*);
    extern void Cblacs_exit(int);
    extern void Cblacs_gridexit(int);
    extern void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
    extern void pdgemr2d_(int*, int*, double*, int*, int*, int*, double*, int*, int*, int*, int*);
}

int main() {
    int myrank, numprocs, info;
    int ctxt, nprow, npcol, myrow, mycol;
    int n, nb, mloc, nloc;
    std::vector<double> global_A;
    std::vector<double> local_A;

    // Initialize MPI
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    // Initialize Scalapack
    Cblacs_get(0, 0, &ctxt);
    nprow = ...; // Number of process rows
    npcol = ...; // Number of process columns
    Cblacs_gridinit(&ctxt, "Row-major", nprow, npcol);
    Cblacs_gridinfo(ctxt, &nprow, &npcol, &myrow, &mycol);

    // Define global matrix size (N) and block size (NB)
    n = ...;
    nb = ...;

    // Determine local dimensions
    mloc = iloc(n, nb, myrow, 0, nprow);
    nloc = iloc(n, nb, mycol, 0, npcol);

    // Allocate memory for local matrices
    local_A.resize(mloc * nloc);

    // Distribute global matrix among processes
    if (myrank == 0)
    {
        global_A.resize(n * n);
        // Fill global_A with data
        // ...
    }

    // Distribute global_A to local_A
    int desc_a[9];
    descinit_(desc_a, &n, &n, &nb, &nb, &myrow, &mycol, &ctxt, &mloc, &info);
    pdgemr2d_(&n, &n, global_A.data(), &n, &n, desc_a, local_A.data(), &nloc, &nloc, desc_a, &ctxt);

    // Perform computations on local_A
    // ...

    // Finalize Scalapack
    Cblacs_gridexit(ctxt);
    Cblacs_exit(0);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}

// Function to calculate local dimensions
int iloc(int n, int nb, int iproc, int isrcproc, int nprocs)
{
    int nblocks = n / nb;
    int leftovers = n % nb;
    int loc_dim = (iproc < leftovers) ? nblocks + 1 : nblocks;
    return loc_dim;
}

{

    MPI_Init(&argc, &argv);

    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set up the grid for Scalapack
    Cblacs_pinfo(&rank, &size);
    Cblacs_get(-1, 0, &context);

    // Determine the number of processes in each dimension of the grid
    nprow = static_cast<int>(sqrt(size));
    npcol = size / nprow;

    Cblacs_gridinit(&context, (char *)"Col-major", nprow, npcol);
    Cblacs_gridinfo(context, &nprow, &npcol, &myrow, &mycol);

    // Determine local matrix dimensions
    localrows = numroc_(&N, &NB, &myrow, &zero, &nprow);
    localcols = numroc_(&N, &NB, &mycol, &zero, &npcol);
    double *A_local = new double[localrows * localcols];
    double *A_global = new double[N * N];

    descinit_(descA_local, &N, &N, &NB, &NB, &myrow, &mycol, &context, &localrows, &info);
    descinit_(descA_global, &N, &N, &NB, &NB, &zero, &zero, &context, &N, &info);

    pdgemr2d_(&N, &N, A_global, &N, &N, descA_global, A_local, &localrows, &localcols, descA_local, &context);
}


{
void ScatterMatrix(int2 id, int2 pgDims, const T* global, const MatDesc& desc, T* local, int2 localDims)
// Pg Dims = process grid dimensions
// desc is matrix descriptor PROBLEM parameters with context,M,N,Mb,Nb
// Global is location of global matrix/// Local is location of local matrix
// localDim is local dimension of the destination local matrix , that is calculated as M x N divided into pxq blocks.....!!!!
// id is current process identifier id.row// id.col  // id.isRoot // it is the ip x iq 
{
    const int M = desc.M;
    const int N = desc.N;
    const int Mb = desc.Mb;
    const int Nb = desc.Nb;
    const int ctxt = desc.ctxt;

    int sendr = 0;
    int sendc = 0;
    int recvr = 0; 
    int recvc = 0;

    // This is sending BLOCKS of nr x nc from ROOT global matrix to sendr,sendc process as receivers of the nr x nc BLOCKS
    // nr x nc can be MbXNb as maximum value OR the remainder which will be a smaller block (M-r) x (N-c)
    // In Ken PDF example we talk about 3x2 blocks of CONTIGUOS Data in Block cyclic 
    // r and c are the GLOBAL indeces of the upper right corner of the (nr x nc) block to move
    // sendR and sendC are current process ip and iq ??????????????????

    // This has to be done in conjuction with the size of the process grid in example 2x3 (pxq) in this code pgdims.row x pgdims.col  

    for (int r = 0; r < M; r += Mb, sendr = (sendr + 1) % pgDims.row) 
    {
        sendc = 0;

        int nr = std::min(Mb, M - r);

        for (int c = 0; c < N; c += Nb, sendc = (sendc + 1) % pgDims.col) 
        {
            int nc = std::min(Nb, N - c);

            if (id.IsRoot()) 
            {
                ops::CvGESD2D(ctxt, nr, nc, &global[CMIDX(r, c, M)], M, sendr, sendc);
            }

            if (id.row == sendr && id.col == sendc) 
            {
                ops::CvGERV2D(ctxt, nr, nc, &local[CMIDX(recvr, recvc, localDims.row)], localDims.row, 0, 0);
                recvc = (recvc + nc) % localDims.col;
            }
        }

        if (id.row == sendr)
            recvr = (recvr + nr) % localDims.row;
    }
}



}

{

template <typename U>
DistributedMatrix<T> DistributedMatrix<T>::Initialized(int context, int2 blockSize, const LocalMatrix<U>& data)
{
    auto A = Uninitialized<U>(context, blockSize, data);

    LocalMatrix<T> m = LocalMatrix<T>::Initialized(data);
    ScatterMatrix<T>(A.ProcGridId(), A.ProcGridDims(), m.Data(), A.Desc(), A.Data(), A.m_localDims);

    return A;
}

}