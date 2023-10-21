
#include <iostream>
#include <thread>
int count = 0; // global variable

void increment()
{
    for (int i = 0; i < 100000; ++i)
    {
    
        count++;
    }
}

int main()
{
    std::thread t1(increment);
    std::thread t2(increment);
    t1.join();
    t2.join();

    std::cout << "Expected: " << 2* 100000 << " Count: "<< count << std::endl;
    return 0;
}