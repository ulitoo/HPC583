#include <iostream>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Message buffer
    int msg = 0;

    std::cout << "Process " << rank << std::endl;

    if (rank < 6)
    {
        MPI_Finalize();
        return 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Sender process
    if (rank == 3)
    {
        msg = 123321; // bc I like it
        // Send a message to process 5
        MPI_Request req;
        MPI_Isend(&msg, 1, MPI_INT, 5, 0, MPI_COMM_WORLD, &req);

        std::cout << "Process " << rank << " sent message " << msg << std::endl;

        // Wait for the send operation to complete
        MPI_Status status;
        MPI_Wait(&req, &status);
    }

    // Receiver process
    if (rank == 5)
    {
        // Receive a message from process 3
        MPI_Request req;
        MPI_Irecv(&msg, 1, MPI_INT, 3, 0, MPI_COMM_WORLD, &req);

        // Wait for the receive operation to complete
        MPI_Status status;
        MPI_Wait(&req, &status);

        std::cout << "Process " << rank << " received message " << msg << std::endl;
    }

    MPI_Finalize();
    return 0;
}
