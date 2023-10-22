#include <iostream>
#include <complex>
#include <vector>
#include <cmath>

const double PI = 3.14159265358979323846;

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
        std::complex<double> t = std::polar(1.0, -2 * PI * i / n) * odd[i];
        a[i] = even[i] + t;
        a[i + n / 2] = even[i] - t;
    }
}

int main()
{
    std::vector<std::complex<double>> signal = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};

    fft(signal);

    std::cout << "FFT Results:" << std::endl;
    for (const auto &element : signal)
    {
        std::cout << element << " ";
    }
    
    std::cout << std::endl;
    return 0;
}
