#include <mutex>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <cmath>
#include <vector>
#include <iomanip>

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


void compute_partial_sum(double &sum, std::mutex &sum_mutex, double a, double h, int n, int num_threads, int i)
{
    double partial_sum = 0.0;
    int start = (n / num_threads) * i;
    int end = (i == num_threads - 1) ? n : (n / num_threads) * (i + 1);
    for (int j = start; j < end; ++j)
    {
        double x = a + j * h;
        partial_sum += f(x);
    }
    partial_sum *= h;
    // Lock the mutex and update the sum variable
    sum_mutex.lock();
    sum += partial_sum;
    sum_mutex.unlock();
}


double parallel_riemann_sum(double a, double b, int n, int num_threads)
{
    double h = (b - a) / n;
    double sum = 0.0;
    std::vector<std::thread> threads(num_threads);
    // Mutex for protecting the sum variable 
    std::mutex sum_mutex;
    // Spawn threads to compute partial sums
    for (int i = 0; i < num_threads; ++i)
    {
        threads[i] = std::thread(compute_partial_sum, std::ref(sum), std::ref(sum_mutex), a, h, n, num_threads, i);
    }
    // Wait for threads to finish
    for (int i = 0; i < num_threads; ++i)
    {
        threads[i].join();
    }
    return sum;
}



int main(int argc, char *argv[])
{
    const double a = 0.0;
    const double b = 1.0;
    const int n = 100000000;
    // timer foo
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " nthreads " << std::endl;
        return 1;
    }
    std::cout << std::setprecision(20);
    const int num_threads = std::atoi(argv[1]);
    // Compute Riemann sum sequentially
    start = std::chrono::high_resolution_clock::now();
    double seq_sum = riemann_sum(a, b, n);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time = (duration.count() * 1.e-9);
    std::cout << "PI: \t\t" << M_PI << std::endl;
    std::cout << "Sequential: \t" << 4*seq_sum << "\ttime: " << elapsed_time << std::endl;

    // Compute Riemann sum in parallel
    start = std::chrono::high_resolution_clock::now();
    double par_sum = parallel_riemann_sum(a, b, n, num_threads);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    elapsed_time = (duration.count() * 1.e-9);
    std::cout << "Parallel: \t" << 4*par_sum << "\ttime: " << elapsed_time << std::endl;
    return 0;
}
