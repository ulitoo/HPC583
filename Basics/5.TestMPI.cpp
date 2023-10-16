#include <mpi.h>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <cmath>
#include <vector>
#include <iomanip>

using namespace std;

double f(double x)
{
    //return sqrt(1-x*x);
    return std::sin(x); // function to integrate
}

double riemann_sum(double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 0; i < n; ++i)

    {
        double x = a + i * h;
        sum += f(x);
    }
    sum *= h;
    return sum;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int ip, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &ip);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    double a = 0.0, b = 1.0;
    int n = 100000000;
    double local_a = a + ip * (b - a) / np;
    double local_b = a + (ip + 1) * (b - a) / np;
    int local_n = n / np;
    double local_sum = riemann_sum(local_a, local_b, local_n);
    double global_sum=0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (ip == 0)
    {
        std::cout << "Result: " << a << " to " << b << " = " << global_sum << std::endl;
    }
    MPI_Finalize();
    return 0;
}