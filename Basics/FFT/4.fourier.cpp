#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include "gnuplot-iostream.h" // Gnuplot C++ interface

std::vector<std::complex<double>> calculateDFT(const std::vector<double> &signal)
{
    int N = signal.size();
    std::vector<std::complex<double>> dft(N);

    for (int k = 0; k < N; k++)
    {
        dft[k] = {0.0, 0.0};
        for (int n = 0; n < N; n++)
        {
            double angle = 2.0 * M_PI * k * n / N;
            std::complex<double> term(std::cos(angle), -std::sin(angle));
            dft[k] += signal[n] * term;
        }
    }

    return dft;
}

int main(int argc, char **argv)
{
    // Timers
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    
    // Initialize Gnuplot
    Gnuplot gp,gp1;
    int N = 1 << atoi(argv[1]);

    // Generate and plot a simple function (e.g., sine function)
    std::vector<std::pair<double, double>> signal;
    std::vector<std::pair<double, double>> Fsignal;
    std::vector<double> xsignal, ysignal, xFsignal, yFsignal;

    double x = 0.0;
    double y = 0.0;
    for (int i = 0; i < N - 1; i++)
    {
        xsignal.push_back(x);
        y = (10.0) * std::sin(2.0 * M_PI * x / N) + (3.0) * std::sin(5.0 * 2.0 * M_PI * x / N) + (2.0) * std::sin(10.0 * 2.0 * M_PI * x / N);
        ysignal.push_back(y);
        signal.push_back(std::make_pair(x, y));
        x += 3;
    }

    // Calculate the DFT
    start = std::chrono::high_resolution_clock::now();

    std::vector<std::complex<double>> dft = calculateDFT(ysignal);
    
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time = (duration.count() * 1.e-9);
    std::cout << "Time: " << elapsed_time << " s." << std::endl;


    // Print the results
    for (int k = 0; k < N; k++)
    {
       // std::cout << "DFT[" << k << "] = " << dft[k] << std::endl;
    }

    x = 0.0;
    for (int i = 0; i < N - 1; i++)
    {
        yFsignal.push_back( std::abs(dft[i]) / N );
        Fsignal.push_back(std::make_pair(x, yFsignal [i]));
        x += 1;
    }

    // ***** PLOT OPTIONS ****  gp << "plot '-' with linespoints linewidth 1 linecolor black linetype 4 title 'sin(x)'\n"; // Plot the data
    // ***** PLOT OPTIONS ****  gp << "set grid linewidth 2\n";
    //                          gp << "set multiplot\n";

    gp << "set terminal wxt size 1024,768 position 0,0\n"; // Set the terminal (window size)
    gp << "set grid\n";
    gp1 << "set terminal wxt size 1024,768 position 1024,0\n"; // Set the terminal (window size)
    gp1 << "set grid\n";
    //gp << "set multiplot layout 2,1 rowsfirst\n";
    //gp << "plot '-' with lines title 'sin(x)', '-' with lines title 'DFT'\n";
    gp << "plot '-' with lines title 'sin(x)'\n";
    gp.send(signal); // Send the data to Gnuplot
    //std::cout << "Press enter to continue." << std::endl;
    //std::cin.get();
    gp1 << "plot '-' with lines title 'DFT'\n";
    gp1.send(Fsignal);
    //gp << "unset multiplot\n";

    for (int i = 0; i < N - 1; i++)
    {
        //    std::cout << "(" << signal[i].first << "," << signal[i].second << ")-";
    }

    // Keep the plot window open
    std::cout << "Press enter to exit." << std::endl;
    std::cin.get();

    return 0;
}
