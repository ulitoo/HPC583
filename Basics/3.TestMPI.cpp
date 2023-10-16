#include <iostream>
#include <mpi.h>

using namespace std;
int main(int argc, char *argv[])
{
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int size, rank, len, tag;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name, &len);
    
    if (rank == 3) {
        int message = 4222;
        MPI_Send(&message,1,MPI_INT,1,tag,MPI_COMM_WORLD);
        std::cout<<"Process 3 Sent Message: "<< message<<std::endl;
    }
    else if (rank == 1) {
        int message;
        MPI_Recv(&message,1,MPI_INT,3,tag,MPI_COMM_WORLD,&status);
        std::cout<<"Process 1 Received Message: "<< message<<std::endl;
    }
    
    //cout << "Hello world from process " << rank << " of " << size << " on " << processor_name << endl;
    MPI_Finalize();
    return 0;
}