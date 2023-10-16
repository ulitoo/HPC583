#include <iostream>
#include <mpi.h>

using namespace std;
int main(int argc, char *argv[])
{
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int size, rank, len;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name, &len);
    cout << "Hello world from process " << rank << " of " << size << " on " << processor_name << endl;
    //cout << "Length " << argv[4] << endl;
    MPI_Finalize();
    return 0;
}