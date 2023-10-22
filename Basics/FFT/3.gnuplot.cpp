#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include "gnuplot-iostream.h" // Gnuplot C++ interface

int main() {
    // Initialize Gnuplot
    Gnuplot gp;

    // Generate and plot a simple function (e.g., sine function)
    std::vector<std::pair<double, double>> data;
    for (double x = 0.0; x < 10.0; x += .1) {
        data.push_back(std::make_pair(x, std::sin(x)));
    }

    // ***** PLOT OPTIONS ****  gp << "plot '-' with linespoints linewidth 1 linecolor black linetype 4 title 'sin(x)'\n"; // Plot the data
    // ***** PLOT OPTIONS ****  gp << "set grid linewidth 2\n";
    gp << "set terminal wxt size 1024,768\n"; // Set the terminal (window size)
    gp << "set grid\n";
    gp << "plot '-' with lines title 'sin(x)'\n";
    gp.send(data); // Send the data to Gnuplot

    // Keep the plot window open
    std::cout << "Press enter to exit." << std::endl;
    std::cin.get();

    return 0;
}
