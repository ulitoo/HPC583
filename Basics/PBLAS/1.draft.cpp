#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include "scalapack.h"

static int MAX(int a, int b)
{
    if (a > b)
        return (a);
    else
        return (b);
}
int main(int argc, char **argv)
{
    // Useful constants for ScaLapack
    const int i_one = 1, i_negone = -1, i_zero = 0;
    double zero = 0.0E+0, one = 1.0E+0;

    // used for MPI init
    int iam = 0, nprocs = 0;
    int myrank_mpi, nprocs_mpi;
    // Used for BLACS grid
    int nprow = 2, npcol = 2, myrow = -1, mycol = -1;
    // dimension for global matrix, Row and col blocking params
    int M, N, mb, nb, LWORK, INFO = 0, ictxt = 0;
    int descA_distr[9], descA[9];
    int lld = 0;
    double *A;
    // dimension for local matrix
    int mp, nq, lld_distr;
    double *A_distr;
    double *TAU, *WORK;
    // counters, seeds
    int i = 0, itemp, seed;

    // call MPI init commands
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

    // Input your parameters: m, n - matrix dimensions, mb, nb - blocking parameters,
    // nprow, npcol - grid dimensions
    // Want 2x2 grid, will change in future
    nprow = 2;
    npcol = 2;
    // Matrix Size
    M = 16;
    N = 16;
    // Matrix blocks
    mb = 2;
    nb = 2;
    // for local work
    LWORK = nb * M;

    // Part with invoking of ScaLAPACK routines. Initialize process grid, first
    /*
    //Cblacs routines, do not work
    Cblacs_pinfo( &iam, &nprocs ) ;
    Cblacs_get( -1, 0, &ictxt );
    Cblacs_gridinit( &ictxt, "Row", nprow, npcol );
    Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );

    printf("Thread=%d, Cblacs_pinfo, iam=%d,nprocs=%d\n",myrank_mpi,iam,nprocs);
    printf("Thread=%d, After info, nprow=%d, npcol=%d, myrow=%d, mycol=%d\n",myrank_mpi,nprow,npcol,myrow,mycol);
    */

    // blacs Method, does not work
    printf("Thread %d: Before init, nprow=%d, npcol=%d, myrow=%d, mycol=%d\n", myrank_mpi, nprow, npcol, myrow, mycol);
    // printf("before, ictxt=%d\n",ictxt);
    Cblacs_get(&i_zero, &i_zero, &ictxt);
    Cblacs_gridinit(&ictxt, "R", &nprow, &npcol);
    Cblacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol);
    printf("Thread %d: After init, nprow=%d, npcol=%d, myrow=%d, mycol=%d\n", myrank_mpi, nprow, npcol, myrow, mycol);
    // Generate Data
    if (myrow == 0 && mycol == 0)
    {
        A = malloc(M * N * sizeof(double));
        // just generate some random samples for now,
for(i =0; i {
            A = rand() % 1000000;
}
// FOR DEBUG
/*A[0]=1.0;
A[1]=2.0;
A[2]=3.0;
A[3]=4.0;
A[4]=4.0;
A[5]=3.0;
A[6]=2.0;
A[7]=1.0;
A[8]=1.0;
A[9]=2.0;
A[10]=3.0;
A[11]=4.0;
A[12]=4.0;
A[13]=3.0;
A[14]=2.0;
A[15]=1.0;*/
    }
    else
    {
        A = NULL;
        // other processes don't contain parts of A
    }

    // Compute dimensions of local part of distributed matrix A_distr
    /*
    * DEBUG
    printf("M=%d\n",M);
    printf("mb=%d\n",mb);
    printf("myrow=%d\n",myrow);
    printf("izero=%d\n",i_zero);
    printf("nprow=%d\n",nprow);
    * */

    mp = numroc_(&M, &mb, &myrow, &i_zero, &nprow);
    nq = numroc_(&N, &nb, &mycol, &i_zero, &npcol);
    printf("Thread %d: After mp=%d, np=%d\n", myrank_mpi, mp, nq);

    A_distr = malloc(mp * nq * sizeof(double));
    WORK = (double *)malloc(N * sizeof(double));
    TAU = (double *)malloc(N * sizeof(double));

    // Initialize discriptors (local matrix A is considered as distributed with blocking parameters
    // m, n, i.e. there is only one block - whole matrix A - which is located on process (0,0) )
    lld = MAX(numroc_(&N, &N, &myrow, &i_zero, &nprow), 1);
    descinit_(descA, &M, &N, &M, &N, &i_zero, &i_zero, &ictxt, &lld, &INFO);
    lld_distr = MAX(mp, 1);
    descinit_(descA_distr, &M, &N, &mb, &nb, &i_zero, &i_zero, &ictxt, &lld_distr, &INFO);

    // Call pdgeadd_ to distribute matrix (i.e. copy A into A_distr)
    pdgeadd_("N", &M, &N, &one, A, &i_one, &i_one, descA, &zero, A_distr, &i_one, &i_one, descA_distr);

    // Now call ScaLAPACK routines
    pdgeqrf_(&M, &N, A_distr, &i_one, &i_one, descA_distr, TAU, WORK, &LWORK, &INFO);

    // Copy result into local matrix
    pdgeadd_("N", &M, &N, &one, A_distr, &i_one, &i_one, descA_distr, &zero, A, &i_one, &i_one, descA);

    free(A_distr);
    if (myrow == 0 && mycol == 0)
    {
        free(A);
    }

    // End of ScaLAPACK part. Exit process grid.
    blacs_gridexit_(&ictxt);
    blacs_exit_(&i_zero);
    // finalize MPI grid
    //  MPI_Finalize();
    exit(0);
}