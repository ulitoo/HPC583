#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <fftw3.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// Function to initialize a 2D Gaussian on a lattice
void initializeGaussian(fftw_complex *data, int N, double sigma)
{
    double norm = 1.0 / (2.0 * M_PI * sigma * sigma);
    double sigmaSq = sigma * sigma;

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int index = i * N + j;

            double x = i - N / 2;
            double y = j - N / 2;
            double exponent = -(x * x + y * y) / (2.0 * sigmaSq);

            data[index][0] = norm * exp(exponent);
            data[index][1] = 0.0; // Initialize imaginary part to 0
        }
    }
}

int main()
{
    // Define the lattice size
    int N = 128;

    // Allocate memory for input and output data
    fftw_complex *inputData = reinterpret_cast<fftw_complex *>(fftw_malloc(sizeof(fftw_complex) * N * N));
    fftw_complex *outputData = reinterpret_cast<fftw_complex *>(fftw_malloc(sizeof(fftw_complex) * N * N));

    // Create FFTW plans
    fftw_plan forwardPlan = fftw_plan_dft_2d(N, N, inputData, outputData, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan reversePlan = fftw_plan_dft_2d(N, N, outputData, inputData, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Initialize input data with a Gaussian
    // double sigma = N / 6.0;
    double sigma = 3.;
    initializeGaussian(inputData, N, sigma);

    // Write input data to file
    std::ofstream inputFile("input_data.txt");
    for (int i = 0; i < N * N; ++i)
    {
        inputFile << inputData[i][0] << " " << inputData[i][1] << std::endl;
    }
    inputFile.close();

    // Perform forward Fourier transform
    fftw_execute(forwardPlan);

    // Write output data to file
    std::ofstream outputFile("output_data.txt");
    for (int i = 0; i < N * N; ++i)
    {
        outputFile << outputData[i][0] << " " << outputData[i][1] << std::endl;
    }
    // for (int i = 0; i < N * N; ++i)
    //{
    //     outputFile << outputData[i][0] << " " << outputData[i][1] << std::endl;
    // }
    outputFile.close();

    // Clean up FFTW resources
    fftw_destroy_plan(forwardPlan);
    fftw_destroy_plan(reversePlan);
    fftw_free(inputData);
    fftw_free(outputData);

    // Fork a child process to execute the Python script
    pid_t pid = fork();
    if (pid == -1)
    {
        perror("fork");
        exit(EXIT_FAILURE);
    }
    else if (pid == 0)
    {
        // Child process: execute the Python script using execvp
        const char *python_args[] = {"python3", "plot_data.py", nullptr};
        execvp("python3", const_cast<char *const *>(python_args));
        // If execvp returns, there was an error
        perror("execvp");
        exit(EXIT_FAILURE);
    }
    else
    {
        // Parent process: wait for the child to exit
        int status;
        if (waitpid(pid, &status, 0) == -1)
        {
            perror("waitpid");
            exit(EXIT_FAILURE);
        }
        // Check the exit status of the child process
        if (WIFEXITED(status))
        {
            int exit_status = WEXITSTATUS(status);
            if (exit_status != 0)
            {
                fprintf(stderr, "Python script exited with status %d\n", exit_status);
            }
        }
        else if (WIFSIGNALED(status))
        {
            int signal_number = WTERMSIG(status);
            fprintf(stderr, "Python script was killed by signal %d\n", signal_number);
        }
    }

    std::cout << "FFT completed successfully!" << std::endl;

    return 0;
}
