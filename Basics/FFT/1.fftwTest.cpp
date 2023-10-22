#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <fftw3.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>


/*
int main()
{
    // Define the lattice size
    int N = 128;
    fftw_complex *in, *out;
    fftw_plan p;
    //...
    in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    //...
    fftw_execute(p); // repeat as needed 
    //...
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    return 0;
} 
*/

int main() {
    // Define the size of the input signal
    int N = 128;
    printf("DFT Hola");

    // Create an array to hold the input signal
    //double* in = (double*) fftw_malloc(sizeof(double) * N);
    double* in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);

    // Initialize the input signal (example values)
    for (int i = 0; i < N; i++) {
        in[i] = i;
    }

    // Create arrays to hold the real and imaginary parts of the output
    //double* out = (double*) fftw_malloc(sizeof(double) * N);
    double* out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    
    // Create a plan for the forward DFT
    //fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, (fftw_complex*)out, FFTW_ESTIMATE);
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute the plan to compute the DFT
    fftw_execute(plan);

    // Output the results
    for (int i = 0; i < N; i++) {
    //    printf("DFT[%d] = %f + %fi\n", i, out[i][0], out[i][1]);
    }

    // Clean up
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return 0;
}
