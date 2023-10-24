#include <iostream>
#include <iomanip>
#include <complex>
#include <vector>
#include <cmath>
#include "gnuplot-iostream.h" // Gnuplot C++ interface

using namespace std::complex_literals;

// Cooley-Tukey FFT algorithm
void fft(std::vector<std::complex<double>> &a)
{
    int n = a.size();
    if (n <= 1)
    {
        return;
    }

    std::vector<std::complex<double>> even(n / 2);
    std::vector<std::complex<double>> odd(n / 2);

    for (int i = 0; 2 * i < n; ++i)
    {
        even[i] = a[2 * i];
        odd[i] = a[2 * i + 1];
    }

    fft(even);
    fft(odd);

    for (int i = 0; 2 * i < n; ++i)
    {
        std::complex<double> t = std::polar(1.0, -2 * M_PI * i / n) * odd[i];
        a[i] = even[i] + t;
        a[i + n / 2] = even[i] - t;
    }
}

int main(int argc, char **argv)
{
    // Timers
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    int N = 8 ;
    // Initialize Gnuplot
    Gnuplot gp,gp1;


    if (argc == 2)
    {
        N = 1 << atoi(argv[1]);
    }

    // Generate and plot a simple function (e.g., sine function)
    std::vector<std::pair<double, double>> signal;
    std::vector<std::pair<double, double>> Fsignal;
    std::vector<double> xsignal, ysignal, xFsignal, yFsignal;
    std::vector<std::complex<double>> FFTsignal;

    double x = 0.0;
    double y = 0.0;
    for (int i = 0; i < N ; i++)
    {
        xsignal.push_back(x);
        y = (10.0) * std::sin(2.0 * M_PI * x / N) + (3.0) * std::sin(5.0 * 2.0 * M_PI * x / N) + (2.0) * std::sin(10.0 * 2.0 * M_PI * x / N);
        FFTsignal.push_back(y);
        signal.push_back(std::make_pair(x, y));
        x += 3;
        //ysignal.push_back(y);
    }

    // Calculate the FFT
    start = std::chrono::high_resolution_clock::now();
    
    fft(FFTsignal);
    
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time = (duration.count() * 1.e-9);
    std::cout << "Time: " << elapsed_time << " s." << std::endl;

    // Print the results
    for (int k = 0; k < N; k++)
    {
        //std::cout << "DFT[" << k << "] = " << abs(FFTsignal[k]) << std::endl;
    }

    x = 0.0;
    for (int i = 0; i < N - 1; i++)
    {
        yFsignal.push_back( std::abs(FFTsignal[i]) / N );
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

    return 0;
}

