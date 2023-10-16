#include <iostream>
#include <mpi.h>

using namespace std;
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int value = 0;
    if (rank == 0)
    {
        value = 21232;
        
    }
    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::cout << "Process " << rank << ": value = "<< value << std::endl;
    MPI_Finalize();
    return 0;
}