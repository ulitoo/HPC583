#include <mpi.h>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <cmath>
#include <vector>
#include <iomanip>
#include <chrono>
using namespace std;

void function1()
{
    for (int i = 0; i < 15; ++i)
    {
        cout << "Thread 1 executing:"<<i<<"\n";
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}

void function2()
{
    for (int i = 0; i < 15; ++i)
    {
        cout << "Thread 2 executing:"<<i<<"\n";
        this_thread::sleep_for(chrono::milliseconds(20));
    }
}

int main()
{
    cout << "Main thread executing\n";
    thread t1(function1);
    thread t2(function2);
    t1.detach();
    t2.detach();
    cout << "Main thread continues... going to sleep\n";
    // wait for the detached threads to finish executing
    this_thread::sleep_for(chrono::milliseconds(2000));
    return 0;
}