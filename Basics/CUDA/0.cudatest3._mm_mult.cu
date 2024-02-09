#include <iostream>

// Matrix multiplication kernel
__global__ void matrixMultiply(double *a, double *b, double *c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("N is:%d , thread x: %d and thread y: %d \n",N,threadIdx.x,threadIdx.y);
    //printf("N is:%d , col %d and row: %d \n",N,col,row);
    if (row < N && col < N)
    {
        double sum = 0;
        for (int i = 0; i < N; ++i)
        {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

void matrixPrint(double *a, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << a[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int getMaxBlockSize()
{
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMultiply, 0, 0);
    return blockSize;
}


int main(int argc, char *argv[])
{

    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " n (Dimension of Matrix)" << std::endl;
        return 1;
    }

    const int N = std::atoi(argv[1]); // Matrix size (N x N)

    // Matrix dimensions
    const int matrixSize = N * N * sizeof(double);

    // Host matrices
    double *h_A, *h_B, *h_C;

    // Allocate memory for host matrices
    h_A = new double[N * N];
    h_B = new double[N * N];
    h_C = new double[N * N];

    // Initialize host matrices
    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = (double)i+(double)1.0/7.0;
        h_B[i] = (double)i-(double)1.0/3.0;
    }

    // Allocate memory on the device (GPU)
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, matrixSize);
    cudaMalloc((void **)&d_B, matrixSize);
    cudaMalloc((void **)&d_C, matrixSize);

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);
    
    int maxBL1D = getMaxBlockSize();
    int maxBL2D = static_cast<int>(sqrt(maxBL1D));

    std::cout<< "1D MAXBLOCKSIZE: " << maxBL1D << "\n";
    std::cout<< "2D MAXBLOCKSIZE , each Dimension Max: " << maxBL2D << "\n";

    // Define grid and block dimensions
    //dim3 blockSize(32,32);

    dim3 blockSize(maxBL2D,maxBL2D);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);


    // Launch matrix multiplication kernel
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);

    // Print the a matrices
    std::cout << "A Matrix:" << std::endl;
    //matrixPrint(h_A, N);
    std::cout << "B Matrix:" << std::endl;
    //matrixPrint(h_B, N);
    std::cout << "Result Matrix:" << std::endl;
    //matrixPrint(h_C, N);

    // Free allocated memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
