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
#include <iostream>
#include <stdlib.h>
#include <sstream>

// #include "scalapack.h"
#include "JBG_BLAS.h"

using namespace std;

extern "C"
{
  /* Cblacs declarations */
  void Cblacs_pinfo(int *, int *);
  void Cblacs_get(int, int, int *);
  void Cblacs_gridinit(int *, const char *, int, int);
  void Cblacs_gridinfo(int, int *, int *, int *, int *);
  void Cblacs_pcoord(int, int, int *, int *);
  void Cblacs_gridexit(int);
  void Cblacs_barrier(int, const char *);
  void Cdgerv2d(int, int, int, double *, int, int, int);
  void Cdgesd2d(int, int, int, double *, int, int, int);

  int numroc_(int *, int *, int *, int *, int *);

  void pdsyev_(char *, char *, int *, double *, int *, int *, int *, double *, double *, int *, int *, int *,
               double *, int *, int *);
  void descinit_(int *, int *, int *, int *, int *, int *, int *,
                 int *, int *, int *);
}

int main(int argc, char **argv)
{

  int seed = 13;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  /* MPI */
  int mpirank, nprocs;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  double MPIelapsed;
  double MPIt2;
  double MPIt1;

  /* Helping vars */
  int iZERO = 0;
  int iONE = 1;
  int verbose = 1;
  bool mpiroot = (mpirank == 0);

  if (argc < 7)
  {
    if (mpiroot)
      cerr << "Usage: N M Nb Mb procrows proccols"
           << endl
           << " N = Rows , M = Cols , Nb = Row Blocks , Mb = Col Blocks "
           << endl;

    MPI_Finalize();
    return 1;
  }

  /* Scalapack / Blacs Vars */
  int N, M, Nb, Mb, procrows, proccols;
  int descA[9], descZ[9];
  int info = 0;

  double *A = NULL, *A_loc = NULL;
  double *w = NULL, *Z = NULL;
  double *work = NULL;

  /* Parse command line arguments */
  if (mpiroot)
  {
    /* Read command line arguments */
    stringstream stream;
    stream << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4] << " " << argv[5] << " " << argv[6];
    stream >> N >> M >> Nb >> Mb >> procrows >> proccols;

    /* Reserve space and read matrix (with transposition!) */
    A = new double[N * M];
    // string fname(argv[1]);
    // ifstream file(fname.c_str());
    for (int r = 0; r < N; ++r)
    {
      for (int c = 0; c < M; ++c)
      {
        // file >> *(A + N*c + r);
        A[N * c + r] = dist(rng) - 0.5;
      }
    }

    //    /* Print matrix */
    //
    //    if(verbose == 1) {
    //        cout << "Matrix A:\n";
    //        for (int r = 0; r < N; ++r) {
    //          for (int c = 0; c < M; ++c) {
    //            cout << setw(3) << *(A + N*c + r) << " ";
    //          }
    //          cout << "\n";
    //        }
    //        cout << endl;
    //    }
    // cout << *A << endl;
  }

  MPI_Bcast(&procrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&proccols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* Begin Cblas context */
  int ctxt, myid, myrow, mycol, numproc;
  // int procrows = 2, proccols = 2;
  Cblacs_pinfo(&myid, &numproc);
  Cblacs_get(0, 0, &ctxt);
  Cblacs_gridinit(&ctxt, "Row-major", procrows, proccols);
  Cblacs_gridinfo(ctxt, &procrows, &proccols, &myrow, &mycol);
  /* process coordinates for the process grid */
  // Cblacs_pcoord(ctxt, myid, &myrow, &mycol);

  /* Broadcast of the matrix dimensions */
  int dimensions[4];
  if (mpiroot)
  {
    dimensions[0] = N;  // Global Rows
    dimensions[1] = M;  // Global Cols
    dimensions[2] = Nb; // Local Rows
    dimensions[3] = Mb; // Local Cols
  }
  MPI_Bcast(dimensions, 4, MPI_INT, 0, MPI_COMM_WORLD);
  N = dimensions[0];
  M = dimensions[1];
  Nb = dimensions[2];
  Mb = dimensions[3];

  int nrows = numroc_(&N, &Nb, &myrow, &iZERO, &procrows);
  int ncols = numroc_(&M, &Mb, &mycol, &iZERO, &proccols);

  int lda = max(1, nrows);

  MPI_Bcast(&lda, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* Print grid pattern
 if (myid == 0)
   cout << "Processes grid pattern:" << endl;
 for (int r = 0; r < procrows; ++r) {
   for (int c = 0; c < proccols; ++c) {
     Cblacs_barrier(ctxt, "All");
     if (myrow == r && mycol == c) {
   cout << myid << " " << flush;
     }
   }
   Cblacs_barrier(ctxt, "All");
   if (myid == 0)
     cout << endl;
 }
 */

  if (myid == 0)
  {
    cout << "Run Parameters For EIGENSOLVER:\n" << endl;
    cout << "Global Rows = " << M << endl;
    cout << "Global Cols = " << N << endl;
    cout << "Local Block Rows = " << Mb << endl;
    cout << "Local Block Cols = " << Nb << endl;
    cout << "nrows = " << nrows << endl;
    cout << "ncols = " << ncols << endl;
    cout << "lda = " << lda << endl;
    // cout <<"Order = "<<ord<<endl;
  }

  for (int id = 0; id < numproc; ++id)
  {
    Cblacs_barrier(ctxt, "All");
  }
  A_loc = new double[nrows * ncols];
  for (int i = 0; i < nrows * ncols; ++i)
    *(A_loc + i) = 0.;

  /* Scatter matrix */
  int sendr = 0, sendc = 0, recvr = 0, recvc = 0;
  for (int r = 0; r < N; r += Nb, sendr = (sendr + 1) % procrows)
  {
    sendc = 0;
    int nr = Nb;
    if (N - r < Nb)
      nr = N - r;

    for (int c = 0; c < M; c += Mb, sendc = (sendc + 1) % proccols)
    {
      int nc = Mb;
      if (M - c < Mb)
        nc = M - c;

      if (mpiroot)
      {
        Cdgesd2d(ctxt, nr, nc, A + N * c + r, N, sendr, sendc);
      }

      if (myrow == sendr && mycol == sendc)
      {
        Cdgerv2d(ctxt, nr, nc, A_loc + nrows * recvc + recvr, nrows, 0, 0);
        recvc = (recvc + nc) % ncols;
      }
    }

    if (myrow == sendr)
      recvr = (recvr + nr) % nrows;
  }

  /* Print local matrices */
  //  if(verbose == 1) {
  //  for (int id = 0; id < numproc; ++id) {
  //    if (id == myid) {
  //    cout << "A_loc on node " << myid << endl;
  //    for (int r = 0; r < nrows; ++r) {
  //      for (int c = 0; c < ncols; ++c)
  //        cout << setw(3) << *(A_loc+nrows*c+r) << " ";
  //      cout << endl;
  //    }
  //    cout << endl;
  //      }
  //      Cblacs_barrier(ctxt, "All");
  //    }
  //  }

  for (int id = 0; id < numproc; ++id)
  {
    Cblacs_barrier(ctxt, "All");
  }

  /* DescInit */
  info = 0;
  descinit_(descA, &N, &M, &Nb, &Mb, &iZERO, &iZERO, &ctxt, &lda, &info);
  descinit_(descZ, &N, &M, &Nb, &Mb, &iZERO, &iZERO, &ctxt, &lda, &info);

  if (mpiroot)
  {
    if (verbose == 1)
    {
      if (info == 0)
      {
        cout << "\nDescription init sucesss!" << endl;
      }
      if (info < 0)
      {
        cout << "Error Info < 0: if INFO = -i, the i-th argument had an illegal value" << endl
             << "Info = " << info << endl;
      }
    }
    // Cblacs_barrier(ctxt, "All");
  }

  /* PDSYEV HERE */
  info = 0;
  MPIt1 = MPI_Wtime();

  int lwork;
  double wkopt;
  //  int IA = 1;
  //  int JA = 1;
  //  int IZ = 1;
  //  int JZ = 1;

  w = new double[N];
  Z = new double[N * M];
  char VVV = 'V';
  char UUU = 'U';

  lwork = -1;
  pdsyev_(&VVV, &UUU, &N, A_loc, &iONE, &iONE, descA, w, Z, &iONE, &iONE, descZ, &wkopt, &lwork, &info);

  lwork = (int)wkopt;
  work = new double[lwork];
  pdsyev_(&VVV, &UUU, &N, A_loc, &iONE, &iONE, descA, w, Z, &iONE, &iONE, descZ, work, &lwork, &info);

  delete[] work;

  for (int id = 0; id < numproc; ++id)
  {
    Cblacs_barrier(ctxt, "All");
  }

  MPIt2 = MPI_Wtime();
  MPIelapsed = MPIt2 - MPIt1;
  if (mpiroot)
  {
    std::cout << "\nPDSYEVD MPI Run Time: " << MPIelapsed << std::endl;

    if (info == 0)
    {
      std::cout << "" << std::endl;
      //std::cout << "SUCCESS" << std::endl;
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
        Cdgesd2d(ctxt, nr, nc, Z + nrows * recvc + recvr, nrows, 0, 0);
        recvc = (recvc + nc) % ncols;
      }

      if (mpiroot)
      {
        // Receive the same data
        // The leading dimension of the local matrix is nrows!
        Cdgerv2d(ctxt, nr, nc, Z + N * c + r, N, sendr, sendc);
      }
    }
    if (myrow == sendr)
      recvr = (recvr + nr) % nrows;
  }
  /* Print test matrix */
  //  if (mpiroot) {
  //    if(verbose == 1){
  //      cout << "Eigenvectors:\n";
  //      for (int r = 0; r < N; ++r) {
  //    for (int c = 0; c < M; ++c) {
  //      cout << setw(3) << *(Z + N*c+r) << "   ";
  //    }
  //    cout << endl;
  //      }
  //    }
  //  }

  if (mpiroot)
  {
    double *w2 = new double[N];
    double MPIelapsed2;
    MPIt1 = MPI_Wtime();
    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, VVV, UUU, N, A, N, w2);
    MPIt2 = MPI_Wtime();
    MPIelapsed2 = MPIt2 - MPIt1;
    std::cout << "DSYEVD Run Time: " << MPIelapsed2 << std::endl;
    std::cout << "\nParalell Speedup: " << MPIelapsed2 / MPIelapsed << " x\n"
              << std::endl;
    double Eerror = 0.0;
    for (int i = 0; i < N; i++)
    {
      Eerror += abs(w[i] - w2[i]);
      // cout << w[i] << endl;
    }
    cout << "Eigenvalue Error: " << Eerror << "\n";
    cout << "Eigenvalue Error per element: " << Eerror/N << "\n\n";
  }

  /* Release resources */
  delete[] A;
  delete[] A_loc;
  delete[] w;
  delete[] Z;
  Cblacs_gridexit(ctxt);
  MPI_Finalize();
}