#include <iostream>
#include <vector>
#include "scalapack.h" // Assuming you have the Scalapack header file



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