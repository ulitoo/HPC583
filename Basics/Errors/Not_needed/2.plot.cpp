

#include <iostream>
#include <cstdio>

int main() {
    // Given values for the first series
    double x_values[] = {2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384};
    double y_values[] = {0.190089,0.358521,0.179477,0.103592,0.186909,0.201387,0.264768,0.270892,0.390271,0.545655,0.925044,1.24821,1.66097,1.98549};
    double y_values2[] = {0.0475224,0.209137,0.134608,0.100283,0.190789,0.217254,0.277831,0.265266,0.475646,0.728851,1.23754,1.7446,2.72584,3.0};


    // Open a pipe to Gnuplot
    FILE* gnuplotPipe = popen("gnuplot -persist", "w");

    if (gnuplotPipe != nullptr) {
        // Set up multiplot layout
        //fprintf(gnuplotPipe, "set multiplot layout 1,1\n");
        //fprintf(gnuplotPipe, "set key autotitle columnheader\n");
        fprintf(gnuplotPipe, "set logscale x\n");
        // Send Gnuplot command to plot the first series
        fprintf(gnuplotPipe, "plot '-' with linespoints title 'Series 1' linecolor rgb 'blue'\n");
        for (int i = 0; i < sizeof(x_values) / sizeof(x_values[0]) ; ++i) {
            fprintf(gnuplotPipe, "%f %f\n", x_values[i], y_values[i]);
        }
        // Signal the end of data for the first series
        fprintf(gnuplotPipe, "\ne\n");

        // Send Gnuplot command to plot the second series
        fprintf(gnuplotPipe, "plot '-'  with linespoints title 'Series 2' linecolor rgb 'red'\n");
        for (int i = 0; i <sizeof(x_values) / sizeof(x_values[0]); ++i) {
            fprintf(gnuplotPipe, "%f %f\n", x_values[i], y_values2[i]);
        }
        // Signal the end of data for the second series
        fprintf(gnuplotPipe, "e\n");

        // Close the pipe
        pclose(gnuplotPipe);
    } else {
        std::cerr << "Error opening Gnuplot pipe." << std::endl;
    }

    return 0;
}
