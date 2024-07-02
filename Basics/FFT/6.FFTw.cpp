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

int main(int argc, char **argv)
{
    // Timers
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    int N = 128 ;
    // Initialize Gnuplot
    Gnuplot gp,gp1;
    fftw_complex *in, *out;
    fftw_plan p;
    //printf("FFTW Hola\n");
    if (argc == 2)
    {
        N = 1 << atoi(argv[1]);
    }

    // Create an array to hold the input signal
    // Create arrays to hold the real and imaginary parts of the output
    //double* in = (double*) fftw_malloc(sizeof(double) * N);
    //double* out = (double*) fftw_malloc(sizeof(double) * N);

    in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);

    // Generate and plot a simple function (e.g., sine function)
    std::vector<std::pair<double, double>> signal;
    std::vector<std::pair<double, double>> Fsignal;
    std::vector<double> xsignal, ysignal, xFsignal, yFsignal;
    std::vector<std::complex<double>> FFTsignal;

    // Initialize the input signal (example values)
    double x = 0.0;
    double y = 0.0;
    for (int i = 0; i < N ; i++)
    {
        xsignal.push_back(x);
        y = (10.0) * std::sin(2.0 * M_PI * x / N) + (3.0) * std::sin(5.0 * 2.0 * M_PI * x / N) + (2.0) * std::sin(10.0 * 2.0 * M_PI * x / N);
        FFTsignal.push_back(y);
        signal.push_back(std::make_pair(x, y));
        in[i][0] = y;
        in[i][1] = 0;
        x += 3;
    }

    // Create a plan for the forward DFT
    //fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, (fftw_complex*)out, FFTW_ESTIMATE);
    //start = std::chrono::high_resolution_clock::now();
    
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    start = std::chrono::high_resolution_clock::now();
    
    // Calculate the FFT
    // Execute the plan to compute the DFT
    
    fftw_execute(plan);

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time = (duration.count() * 1.e-9);
    std::cout << "NOPlan+FFT Time: " << elapsed_time << " s.\n" << std::endl;

    // Print the results
    for (int k = 0; k < N; k++)
    {
        //std::cout << "DFT[" << k << "] = " << abs(FFTsignal[k]) << std::endl;
    }

    // Prepare PLOT
    x = 0.0;
    for (int i = 0; i < N ; i++)
    {
        yFsignal.push_back( (sqrt(out[i][0]*out[i][0]+out[i][1]*out[i][1])) / N );
        Fsignal.push_back(std::make_pair(x, yFsignal [i]));
        x += 1;
    }

    gp << "set terminal wxt size 1024,768 position 0,0\n"; // Set the terminal (window size)
    gp << "set grid\n";
    gp1 << "set terminal wxt size 1024,768 position 1024,0\n"; // Set the terminal (window size)
    gp1 << "set grid\n";
    //gp << "set multiplot layout 2,1 rowsfirst\n";
    //gp << "plot '-' with lines title 'sin(x)', '-' with lines title 'DFT'\n";
    gp << "plot '-' with lines title 'sin(x)'\n";
    gp.send(signal); // Send the data to Gnuplot
    gp1 << "plot '-' with lines title 'DFT'\n";
    gp1.send(Fsignal);

    for (int i = 0; i < N ; i++)
    {
        //    std::cout << "(" << signal[i].first << "," << signal[i].second << ")-";
    }

    // Keep the plot window open
    std::cout << "Press enter to exit." << std::endl;
    std::cin.get();
    
    // Clean up
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return 0;
}

