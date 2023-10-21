#include <mpi.h>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <cmath>
#include <vector>
#include <iomanip>

using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //const int array_size = 6;
    const int N = 4;
    
    std::vector<int> sendbuf(N);

    for (int i = 0; i < N; i++)
    {
        sendbuf[i] = rank*N+i;
    }

    std::vector<int> recvbuf(N * size);
    MPI_Gather(&sendbuf[0], N, MPI_INT, &recvbuf[0], N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "Rank " << rank << " received: ";
        // Initialize the array on the root process
        for (int i = 0; i < N * size; i++)
        {
            std::cout << " " << recvbuf[i];
        }
        std::cout << std::endl;
    }

    /* SCATTER
    std::vector<int> sendbuf(array_size);
    if (rank == 0)
    {   
        // Initialize the array on the root process
        for (int i = 0; i < array_size; i++)
        {
            sendbuf[i] = i;
        }
    }
    std::vector<int> recvbuf(array_size / size);
    MPI_Scatter(sendbuf.data(), array_size / size, MPI_INT, recvbuf.data(), array_size / size, MPI_INT, 0, MPI_COMM_WORLD);
    std::cout << "Process " << rank << " received: ";
    for (int i = 0; i < array_size / size; i++)
    {
        std::cout << recvbuf[i] << " ";
    }

    */

    std::cout << std::endl;
    MPI_Finalize();
    return 0;
}