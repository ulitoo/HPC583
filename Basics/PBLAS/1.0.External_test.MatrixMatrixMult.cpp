#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <cblas.h>
#include <thread>
#include <lapacke.h>
#include <fstream>
#include <random>
#include <iomanip>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <sstream>

//#include "scalapack.h"
#include "JBG_BLAS.h"

using namespace std;

extern "C"
{
    /* Cblacs declarations */
    void Cblacs_pinfo(int *, int *);
    void Cblacs_get(int, int, int *);
    void Cblacs_gridinit(int *, const char *, int, int);
    void Cblacs_pcoord(int, int, int *, int *);
    void Cblacs_gridexit(int);
    void Cblacs_barrier(int, const char *);
    void Cdgerv2d(int, int, int, double *, int, int, int);
    void Cdgesd2d(int, int, int, double *, int, int, int);

    int numroc_(int *, int *, int *, int *, int *);
    void descinit_(int *,  int *,  int *,  int *,  int *,  int *,  int *,  int *,  int *, int *);
    void pdgemm_(char *, char *, int *, int *, int *, double *, double *, int *, int *, int *, double *, int *, int *, int *, double *, double *, int *, int *, int *);

}


int main(int argc, char **argv)
{
    
    int seed = 13;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    int mpirank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    bool mpiroot = (mpirank == 0);

    /* Helping vars */
    int iZERO = 0;

    if (argc < 3)
    {
        if (mpiroot)
            cerr << "Usage: Nb N " << endl;
        MPI_Finalize();
        return 1;
    }

    int N, M, Nb, Mb;
    double *A_glob = NULL, *A2_glob = NULL, *A2_glob_kk = NULL, *A2_glob_err = NULL, *A_loc = NULL, *A2_loc = NULL;

    /* Parse command line arguments */
    if (mpiroot)
    {
        std::cout << "\nMultiplication Matrix Matrix Paralel vs LAPACK/BLAS " << std::endl;
        /* Read command line arguments */
        stringstream stream;
        stream << argv[2] << " " << argv[2] << " " << argv[1] << " " << argv[1];
        stream >> N >> M >> Nb >> Mb;

        /* Reserve space and read matrix (with transposition!) */
        A_glob = new double[N * M];
        A2_glob = new double[N * M];
        A2_glob_kk = new double[N * M];
        A2_glob_err = new double[N * M];
        //string fname(argv[1]);
        //ifstream file(fname.c_str());

        for (int r = 0; r < N; ++r)
        {
            for (int c = 0; c < M; ++c)
            {
                //file >> *(A_glob + N * c + r);
                A_glob[N * c + r] = dist(rng) - 0.5;
            }
        }
    }

    /* Begin Cblas context */
    /* We assume that we have 4 processes and place them in a 2-by-2 grid */
    int ctxt, myid, myrow, mycol, numproc;
    int procrows = 2, proccols = 2;
    Cblacs_pinfo(&myid, &numproc);
    Cblacs_get(0, 0, &ctxt);
    Cblacs_gridinit(&ctxt, "Row-major", procrows, proccols);
    Cblacs_pcoord(ctxt, myid, &myrow, &mycol);

    /* Print grid pattern */
    if (myid == 0){}
            //cout << "Processes grid pattern:" << endl;
    for (int r = 0; r < procrows; ++r)
    {
        for (int c = 0; c < proccols; ++c)
        {
            Cblacs_barrier(ctxt, "All");
            if (myrow == r && mycol == c)
            {
                //cout << myid << " " << flush;
            }   
        }
        Cblacs_barrier(ctxt, "All");
        if (myid == 0){}
            //cout << endl;
    }

    /*****************************************
     * HERE BEGINS THE MOST INTERESTING PART *
     *****************************************/

    /* Broadcast of the matrix dimensions */
    int dimensions[4];
    if (mpiroot)
    {
        dimensions[0] = N;
        dimensions[1] = M;
        dimensions[2] = Nb;
        dimensions[3] = Mb;
    }
    MPI_Bcast(dimensions, 4, MPI_INT, 0, MPI_COMM_WORLD);
    N = dimensions[0];
    M = dimensions[1];
    Nb = dimensions[2];
    Mb = dimensions[3];

    /* Reserve space for local matrices */
    // Number of rows and cols owned by the current process
    int nrows = numroc_(&N, &Nb, &myrow, &iZERO, &procrows);
    int ncols = numroc_(&M, &Mb, &mycol, &iZERO, &proccols);
    int descA_local[9], descA2_local[9],info;
    int zero = 0;
    int uno = 1;
    double alpha = 1.0;
    double beta = 0.0;
    char transa = 'N';
    char transb = 'N';
    double MPIelapsed,MPIelapsed2;
    double MPIt2;
    double MPIt1;

    for (int id = 0; id < numproc; ++id)
    {
        Cblacs_barrier(ctxt, "All");
    }
    A_loc = new double[nrows * ncols];
    A2_loc = new double[nrows * ncols];
    for (int i = 0; i < nrows * ncols; ++i)
        {
            *(A_loc + i) = 0.;
            *(A2_loc + i) = 0.;
        }

    /* Scatter matrix */
    int sendr = 0, sendc = 0, recvr = 0, recvc = 0;
    for (int r = 0; r < N; r += Nb, sendr = (sendr + 1) % procrows)
    {
        sendc = 0;
        // Number of rows to be sent
        // Is this the last row block?
        int nr = Nb;
        if (N - r < Nb)
            nr = N - r;

        for (int c = 0; c < M; c += Mb, sendc = (sendc + 1) % proccols)
        {
            // Number of cols to be sent
            // Is this the last col block?
            int nc = Mb;
            if (M - c < Mb)
                nc = M - c;

            if (mpiroot)
            {
                // Send a nr-by-nc submatrix to process (sendr, sendc)
                Cdgesd2d(ctxt, nr, nc, A_glob + N * c + r, N, sendr, sendc);
            }

            if (myrow == sendr && mycol == sendc)
            {
                // Receive the same data
                // The leading dimension of the local matrix is nrows!
                Cdgerv2d(ctxt, nr, nc, A_loc + nrows * recvc + recvr, nrows, 0, 0);
                recvc = (recvc + nc) % ncols;
            }
        }

        if (myrow == sendr)
            recvr = (recvr + nr) % nrows;
    }

    /* Print local matrices */
    for (int id = 0; id < numproc; ++id)
    {
        if (id == myid)
        {
            cout << "\nA_loc on node " << myid << endl;
            /*
            for (int r = 0; r < nrows; ++r)
            {
                for (int c = 0; c < ncols; ++c)
                    cout << setw(3) << *(A_loc + nrows * c + r) << " ";
                cout << endl;
            }
            cout << endl;
            */
        }
        Cblacs_barrier(ctxt, "All");
    }

    ///////////////////////////////////////////// MULTIPLY LOCALS

    descinit_(descA_local, &N, &N, &Nb, &Nb, &zero, &zero, &ctxt, &nrows, &info);
    descinit_(descA2_local, &N, &N, &Nb, &Nb, &zero, &zero, &ctxt, &nrows, &info);

    MPIt1 = MPI_Wtime();
    
    pdgemm_(&transa, &transb, &N, &N, &N, &alpha, A_loc, &uno, &uno, descA_local, A_loc, &uno, &uno, descA_local, &beta, A2_loc, &uno, &uno, descA2_local);

    for (int id = 0; id < numproc; ++id)
    {
        Cblacs_barrier(ctxt, "All");
    }

    MPIt2 = MPI_Wtime();
    MPIelapsed = MPIt2 - MPIt1;

    if (mpiroot)
    {
        std::cout << "\nPDGEMM MPI Run Time: " << MPIelapsed << " s." << std::endl;

        if (info == 0)
        {
            std::cout << "SUCCESS" << std::endl;
        }
        if (info < 0)
        {

            cout << "info < 0:  If the i-th argument is an array and the j-entry had an illegal value, then INFO = -(i*100+j), if the i-th argument is a scalar and had an illegal value, then INFO = -i. " << endl;
            cout << "info = " << info << endl;
        }
        if (info > 0)
        {
            std::cout << "matrix is not positve definte" << std::endl;
        }
    }

    /* Gather matrix */
    sendr = 0;
    for (int r = 0; r < N; r += Nb, sendr = (sendr + 1) % procrows)
    {
        sendc = 0;
        // Number of rows to be sent
        // Is this the last row block?
        int nr = Nb;
        if (N - r < Nb)
            nr = N - r;

        for (int c = 0; c < M; c += Mb, sendc = (sendc + 1) % proccols)
        {
            // Number of cols to be sent
            // Is this the last col block?
            int nc = Mb;
            if (M - c < Mb)
                nc = M - c;

            if (myrow == sendr && mycol == sendc)
            {
                // Send a nr-by-nc submatrix to process (sendr, sendc)
                //Cdgesd2d(ctxt, nr, nc, A2_loc + nrows * recvc + recvr, nrows, 0, 0);
                Cdgesd2d(ctxt, nr, nc, A2_loc + nrows * recvc + recvr, nrows, 0, 0);
                recvc = (recvc + nc) % ncols;
            }

            if (mpiroot)
            {
                // Receive the same data
                // The leading dimension of the local matrix is nrows!
                Cdgerv2d(ctxt, nr, nc, A2_glob + N * c + r, N, sendr, sendc);
            }
        }

        if (myrow == sendr)
            recvr = (recvr + nr) % nrows;
    }

    /* Print result matrix */
    if (mpiroot)
    {
        MPIt1 = MPI_Wtime();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A_glob, N, A_glob, N, 0.0, A2_glob_kk, N);
        
        MPIt2 = MPI_Wtime();
        MPIelapsed2 = MPIt2 - MPIt1;
        std::cout << "\nDGEMM MPI Run Time: " << MPIelapsed2 << " s." << std::endl;
        std::cout << "\nParalell Speedup: " << MPIelapsed2/MPIelapsed << " x" << std::endl;

        double cumm_error = 0.0;
        for (int r = 0; r < N; ++r)
        {
            for (int c = 0; c < M; ++c)
            {
                A2_glob_err[N * c + r] =  *(A2_glob_kk + N * c + r) - *(A2_glob + N * c + r);
                cumm_error += abs(A2_glob_err[N * c + r]);
            }
            //cout << endl;
        }
        cout << "\nMatrix Cumulative Error:" << cumm_error << " ";
        cout << "\nMatrix Error per element:" << cumm_error/N << "\n\n";
    }

    /************************************
     * END OF THE MOST INTERESTING PART *
     ************************************/

    /* Release resources */
    delete[] A_glob;
    delete[] A2_glob;
    delete[] A_loc;
    delete[] A2_loc;
    Cblacs_gridexit(ctxt);
    MPI_Finalize();
}