#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <fftw3.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main() {
    // Define the size of the input signal. 2024
    int N = 128;
    printf("DFT Hola\n");
    fftw_complex *in, *out;
    fftw_plan p;

    // Create an array to hold the input signal
    // Create arrays to hold the real and imaginary parts of the output
    //double* in = (double*) fftw_malloc(sizeof(double) * N);
    //double* out = (double*) fftw_malloc(sizeof(double) * N);

    in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);

    // Initialize the input signal (example values)
    for (int i = 0; i < N; i++) {
        in[i][0] = i;
        in[i][1] = 0;
    }

    // Create a plan for the forward DFT
    //fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, (fftw_complex*)out, FFTW_ESTIMATE);
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute the plan to compute the DFT
    fftw_execute(plan);

    // Output the results
    for (int i = 0; i < N; i++) {
        printf("DFT[%d] = %f + %fi => abs:%f\n", i, out[i][0], out[i][1], sqrt(out[i][0]*out[i][0]+out[i][1]*out[i][1]));
    }

    // Clean up
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return 0;
}
