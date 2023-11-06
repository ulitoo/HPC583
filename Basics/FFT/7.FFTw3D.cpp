#include <iostream>
#include <iomanip>
#include <complex>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <fftw3.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "gnuplot-iostream.h" // Gnuplot C++ interface

using namespace std::complex_literals;

// HOMEWORK 3D transform of a 3d gaussian 3xsigmas -> Add a global phase (Imaginary constant)
// 3D lattice.
// FWD transform
// Reverse into DIFFERENT array
// ESCALE!
// Compare Norms -> errors in each [NORM of error] OR SUMS of [ABS errors] and divide by Volume. - error per point average -
// Calculation of precision.

// 3D Gaussian Function centered in (x0,y0,z0) and with different sigmas per dimension.
double gaussian3Ddetailed(double x, double x0, double sigmax, double y, double y0, double sigmay, double z, double z0, double sigmaz)
{
    return exp(-(((x - x0) * (x - x0) / (2.0 * sigmax * sigmax)) + ((y - y0) * (y - y0) / (2.0 * sigmay * sigmay)) + ((z - z0) * (z - z0) / (2.0 * sigmaz * sigmaz))));
}
// Simplified Gaussian Function, Centered in (0,0,0) and only 1 Sigma
double gaussian3d(double x, double y, double z, double sigma)
{
    return exp(-((x * x + y * y + z * z) / (2.0 * sigma * sigma)));
}

void normalize(fftw_complex *output, int size)
{
    for (int i = 0; i < size; i++)
    {
        output[i][0] /= size;
        output[i][1] /= size;
    }
}

int main(int argc, char **argv)
{
    // Timers
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time;
    int N = 32;
    
    if (argc == 2)
    {
        N = 1 << atoi(argv[1]);
    }
    
    int totalN = N * N * N;

    fftw_complex *in, *out, *in_back;
    fftw_plan plan, plan_back;

    // Create an array to hold the input signal
    // Create arrays to hold the real and imaginary parts of the output

    in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * N * N);
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * N * N);
    in_back = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * N * N);

    // Create a Plan
    plan = fftw_plan_dft_3d(N, N, N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Initialize the input signal (with 3d gaussian values).
    // Note that we will move the Gaussian N/2 in each dimension so the peak will be in N/2 (Center in N/2)
    // Real is populated , Imaginary is 0.0
    double sigma = 3.0; // Standard deviation of the Gaussian

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                double x = (i - N / 2);
                double y = (j - N / 2);
                double z = (k - N / 2);
                in[k + N * j + N * N * i][0] = gaussian3d(x, y, z, sigma);
                in[k + N * j + N * N * i][1] = 0.0;
            }
        }
    }
    // fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, (fftw_complex*)out, FFTW_ESTIMATE);
    start = std::chrono::high_resolution_clock::now();

    // Calculate the 3D FFT
    // Execute the plan to compute the DFT
    fftw_execute(plan);

    // PRINT Time ellapsed
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time = (duration.count() * 1.e-9);
    std::cout << "\nFFT Fwd for "<< (N) <<"^3 points Time: " << elapsed_time << " s.\n"
              << std::endl;

    // Normalize output
    normalize(out, totalN);

    start = std::chrono::high_resolution_clock::now();
    plan_back = fftw_plan_dft_3d(N, N, N, out, in_back, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan_back);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time = (duration.count() * 1.e-9);
    std::cout << "\nFFT Back for "<< (N) <<"^3 points Time: " << elapsed_time << " s.\n"
              << std::endl;

    // Normalize in_back
    normalize(in_back, totalN);

    double errorabs = 0;
    double errorsqr = 0;
    double error_Real = 0;

    // Calculate Error
    for (int i = 0; i < N*N*N; i++)
    {
        // Error as difference of im and Re squared
        errorsqr += ((out[i][0]-in_back[i][0])*(out[i][0]-in_back[i][0]) + (out[i][1]-in_back[i][1])*(out[i][1]-in_back[i][1]));
        // Error as difference absoute values
        errorabs += abs(sqrt(out[i][0]*out[i][0] + out[i][1]*out[i][1]) - sqrt(in_back[i][0]*in_back[i][0] + in_back[i][1]*in_back[i][1]));
        // Error only as cummulative of the real parts
        error_Real += abs(out[i][0]-in_back[i][0]);
    }

    std::cout << "\nFFT Sqr Error: " << sqrt(errorsqr) <<"\n";
    std::cout << "FFT Abs Error: " << errorabs <<"\n";
    std::cout << "FFT Real Error: " << error_Real <<"\n";

    // Clean up
    fftw_destroy_plan(plan);
    fftw_destroy_plan(plan_back);
    fftw_free(in);
    fftw_free(out);
    fftw_free(in_back);

    return 0;
}
