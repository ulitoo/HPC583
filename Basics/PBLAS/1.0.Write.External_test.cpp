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

// #include "scalapack.h"
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
}

int main(int argc, char **argv)
{

    int seed = 13;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    if (argc < 4)
    {
        cerr << "Usage:  matrixfile N M" << endl;
        MPI_Finalize();
        return 1;
    }

    int RW, N, M;
    double *A_glob = NULL;

    /* Read command line arguments */
    stringstream stream;
    stream << argv[2] << " " << argv[3];
    stream >> N >> M;

    /* Reserve space and read matrix (with transposition!) */
    A_glob = new double[N * M];

    string fname(argv[1]);

    ofstream filew(fname.c_str());
    for (int r = 0; r < N; ++r)
    {
        for (int c = 0; c < M; ++c)
        {
            A_glob[N * c + r] = dist(rng) - 0.5;
            filew << *(A_glob + N * c + r);
        }
    }
    return 0;
}