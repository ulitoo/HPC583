#include <mutex>
#include <condition_variable>
#include <iostream>
#include <thread>

//std::mutex cout_mutex;
std::mutex count_mutex;
std::condition_variable cv;
int count = 0; // global variable

void increment()
{
    for (int i = 0; i < 100000; ++i)
    {
        std::unique_lock<std::mutex> lock(count_mutex);
        ++count;
    }
    cv.notify_one();
}

int main()
{
    std::thread t1(increment);
    std::thread t2(increment);
    // Wait for the threads to finish and update the count variable
    {
        std::unique_lock<std::mutex> lock(count_mutex);
        cv.wait(lock, []
                { return count == 200000; });
    }

    t1.join();
    t2.join();

    std::cout << "Expected: " << 2 * 100000 << " Count: " << count << std::endl;
    return 0;
}