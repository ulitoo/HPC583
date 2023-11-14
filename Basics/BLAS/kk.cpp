#include <cblas.h>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>
#include "gnuplot-iostream.h" // Gnuplot C++ interface

using namespace std;

// 	Problem , get a LU decomposition recursively
//  Algorithm to be done in Row Major 

/////////////////////////////     FUNCTIONS


void PrintVector(int n, std::vector<double> vector)
{
    for (int i = 0; i < n; i++)
    {
        cout << vector[i] << " ";
    }
    cout << "\n";
}

std::vector<double> InitVector(int n)
{   
    std::vector<double> vector(n);
    for (int i = 0; i < n; i++)
    {
        vector[i] = (double)(i+1);
    }
    return vector;
}

int main()
{

    int n =5;

    //Now Solve system of linear equations given Ax=b given b=( 1 2 3 4 ... n)

    std::vector<double> vectorb;
    vectorb = InitVector(n);
    PrintVector(n,vectorb);


    return 0;
}
