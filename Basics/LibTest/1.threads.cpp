#include <iostream>
#include <string>
#include <thread>
void threadFunc(int id)
{
    std::cout << "Thread " << id << " started." << std::endl;
    // Do some work...
    int cpujbg;
    cpujbg=sched_getcpu();  
    std::cout<<"cpu: "<<(cpujbg)<<"!\n";
    std::cout << "Thread " << id << " finished." << std::endl;
}

int main()
{
    const int numThreads = 100;
    std::thread threads[numThreads];
    // Spawn threads
    for (int i = 0; i < numThreads; ++i)
    {
        threads[i] = std::thread(threadFunc, i);
    }

    // Wait for threads to finish
    for (int i = 0; i < numThreads; ++i)
    {
        threads[i].join();
    }
    std::cout << "All threads finished." << std::endl;
    return 0;
}