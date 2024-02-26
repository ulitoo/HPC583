#include <iostream>
#include <vector>
#include "scalapack.h" // Assuming you have the Scalapack header file

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
    // Acknowledgements to Matt Hurliman https://github.com/mhurliman
    // https://github.com/mhurliman/uw-capstone/blob/57362cb207f58b6463d21f4e4decb8c759826d69/utilities/DistributedMatrix.inl#L19
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
                recvc = (recvc + nc) % localcols;
            }

            if (myprow == 0 && mypcol == 0)
            {
                Cdgerv2d(context, nr, nc, &globalMatrix[COL_MAJOR_INDEX(r, c, M)], M, sendr, sendc);
            }
        }

        if (myprow == sendr)
            recvr = (recvr + nr) % localrows;
    }
}

//ORIGINAL MATT CODE.....
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
    // sendR and sendC are current process ip and iq that can come back again and again in this loop (mod processdim size)
    // recvr and recvc are the element in local matrix where the block will be written   

    // This has to be done in conjuction with the size of the process grid in example 2x3 (pxq) in this code pgdims.row x pgdims.col  

    for (int r = 0; r < M; r += Mb, sendr = (sendr + 1) % pgDims.row)  // go jumping global matrix block by block (row)
    {
        sendc = 0;

        int nr = std::min(Mb, M - r);       // nr is row block size or smaller remainder size

        for (int c = 0; c < N; c += Nb, sendc = (sendc + 1) % pgDims.col) // go jumping global matrix block by block (col)
        {
            int nc = std::min(Nb, N - c);   // nc is col block size or smaller remainder size

            if (id.IsRoot())  //if rank = 0 , root process. send from global matrix to all other processes (sendr,sendc)
            {
                ops::CvGESD2D(ctxt, nr, nc, &global[CMIDX(r, c, M)], M, sendr, sendc); // sendr,sendc are processes to send to
            }

            if (id.row == sendr && id.col == sendc) // if caller to scatter is the corresponding process  (sendr,sendc) then receive            
            {
                ops::CvGERV2D(ctxt, nr, nc, &local[CMIDX(recvr, recvc, localDims.row)], localDims.row, 0, 0); // receive from root 0,0
                                                        // upper left corner of receiving local matrix is 0,0 and will move 1 block....
                recvc = (recvc + nc) % localDims.col; // if there is another block to be written in the same process grid,...
                                                    //  move the receiving upper left corner by a block 
                                                    // I dont think % localDims.col is needed here
            }
        }

        if (id.row == sendr)                        // if caller to scatter is in the row of the corresponding process, move 1 block down
                                                    // for local matrix to receive properly
            recvr = (recvr + nr) % localDims.row; // I dont think % localDims.row is needed here
    }
}



}

{
int numroc_(const int* n, const int* nb, const int* iproc, const int* srcproc, const int* nprocs );
m_localDims.row = numroc_(&dims.row, &blockSize.row, &m_PGridId.row, &zero, &m_PGridDims.row);
m_localDims.col = numroc_(&dims.col, &blockSize.col, &m_PGridId.col, &zero, &m_PGridDims.col);

template <typename U>
DistributedMatrix<T> DistributedMatrix<T>::Initialized(int context, int2 blockSize, const LocalMatrix<U>& data)
{
    auto A = Uninitialized<U>(context, blockSize, data);

    LocalMatrix<T> m = LocalMatrix<T>::Initialized(data);
    ScatterMatrix<T>(A.ProcGridId(), A.ProcGridDims(), m.Data(), A.Desc(), A.Data(), A.m_localDims);

    return A;
}

}