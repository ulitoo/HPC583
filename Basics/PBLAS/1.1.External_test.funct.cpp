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

#include "scalapack.h"
#include "JBG_BLAS.h"

using namespace std;

void dscatter(
    const int& context,                // [IN]
    const double* const& GlobalMatrix, // [IN] Only relevant for root
    double*& LocalMatrix,              // [OUT] The pointer changes
    int& GlobalRows,                   // [IN (root) / OUT (other)]
    int& GlobalCols,                   // [IN (root) / OUT (other)]
    int& BlockRows,                    // [IN (root) / OUT (other)]
    int& BlockCols,                    // [IN (root) / OUT (other)]
    int& LocalRows,                    // [OUT]
    int& LocalCols,                    // [OUT]
    const int& root = 0                // [IN]
) {
    /* Helper variables */
    int iZERO = 0, iONE = 1, iSIX = 6;
 
    int myid, myrow, mycol, procrows, proccols, procnum, rootrow, rootcol;
    blacs_pinfo_(&myid, &procnum);
    blacs_gridinfo_(&context, &procrows, &proccols, &myrow, &mycol);
    bool iamroot = (myid == root);
 
    /* Broadcast matrix info */
    int minfo[6];
    if (iamroot) {
        minfo[0] = GlobalRows;
        minfo[1] = GlobalCols;
        minfo[2] = BlockRows;
        minfo[3] = BlockCols;
        minfo[4] = myrow;
        minfo[5] = mycol;
        igebs2d_(&context, "All", " ", &iSIX, &iONE, minfo, &iSIX);
    } else {
        igebr2d_(&context, "All", " ", &iSIX, &iONE, minfo, &iSIX,
                 &iZERO, &iZERO);
    }
 
    GlobalRows = minfo[0];
    GlobalCols = minfo[1];
    BlockRows  = minfo[2];
    BlockCols  = minfo[3];
    rootrow    = minfo[4];
    rootcol    = minfo[5];
 
    /* Reserve space */
    LocalRows = numroc_(&GlobalRows, &BlockRows, &myrow, &iZERO, &procrows);
    LocalCols = numroc_(&GlobalCols, &BlockCols, &mycol, &iZERO, &proccols);
    LocalMatrix = new double[LocalRows*LocalCols];
 
    /* Scatter matrix */
    int destr = 0, destc = 0;
    int SendRows, SendCols;
    int RecvRow = 0, RecvCol = 0;
    for (int r = 0; r < GlobalRows; r += BlockRows, destr=(destr+1)%procrows) {
        destc = 0;
 
        // Is this the last row bloc?
        SendRows = BlockRows;
        if (GlobalRows-r < BlockRows)
            SendRows = GlobalRows-r;
 
        for (int c=0; c<GlobalCols; c+=BlockCols, destc=(destc+1)%proccols) {
            // Is this the last column block?
            SendCols = BlockCols;
            if (GlobalCols-c < BlockCols)
                SendCols = GlobalCols-c;
 
            // Send data
            if (iamroot) {
                dgesd2d_(&context, &SendRows, &SendCols,
                         GlobalMatrix + GlobalRows*c + r,
                         &GlobalRows, &destr, &destc
                );
            }
 
            // Rerceive data
            if (myrow == destr && mycol == destc) {
                dgerv2d_(&context, &SendRows, &SendCols,
                         LocalMatrix+LocalRows*RecvCol+RecvRow,
                         &LocalRows, &rootrow, &rootcol
                );
 
                // Adjust the next starting column
                RecvCol = (RecvCol + SendCols) % LocalCols;
            }
        }
 
        // Adjust the next starting row
        if (myrow == destr)
            RecvRow = (RecvRow + SendRows) % LocalRows;
 
    }
 
}