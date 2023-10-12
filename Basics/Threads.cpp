#include <iostream>
#include <thread>

void threadFunc(int id)
{
    std::cout << "Thread " << id << " started " << std::endl;
    std::cout << "Thread " << id << " finito " << std::endl;
}

int main()
{
    const int numThreads = 4;
    std::thread threads[numThreads];

    for (int i = 0; i < numThreads; ++i)
    {
        threads[i] = std::thread(threadFunc, i);
    }

    for (int i = 0; i < numThreads; ++i)
    {
        threads[i].join();
    }

    std::cout << "All Finito" << std::endl;
}