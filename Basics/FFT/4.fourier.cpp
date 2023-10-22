#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include "gnuplot-iostream.h" // Gnuplot C++ interface

int main() {
    // Initialize Gnuplot
    Gnuplot gp;
    int N = 1<<11;

    // Generate and plot a simple function (e.g., sine function)
    std::vector<std::pair<double, double>> signal;
    std::vector<std::pair<double, double>> Fsignal;
    std::vector<double> xsignal,ysignal,xFsignal,yFsignal;

    double x = 0.0;
    double y = 0.0;
    for (int i = 0; i < N-1 ; i++)
    {
        xsignal.push_back(x);
        y= (10.0)*std::sin(x)+(3.0)*std::sin(5.0*x)+(2.0)*std::sin(10.0*x);
        ysignal.push_back(y);
        signal.push_back(std::make_pair(x,y));
        x+=0.01;
    }



    // ***** PLOT OPTIONS ****  gp << "plot '-' with linespoints linewidth 1 linecolor black linetype 4 title 'sin(x)'\n"; // Plot the data
    // ***** PLOT OPTIONS ****  gp << "set grid linewidth 2\n";
    
    gp << "set terminal wxt size 1024,768\n"; // Set the terminal (window size)
    gp << "set grid\n";
    gp << "plot '-' with lines title 'sin(x)'\n";
    gp.send(signal); // Send the data to Gnuplot
    

    for (int i = 0; i < N-1 ; i++)
    {
        //std::cout << "(" << signal[i].first << "," << signal[i].second << ")-";
    }

    // Keep the plot window open
    std::cout << "Press enter to exit." << std::endl;
    std::cin.get();

    return 0;
}
