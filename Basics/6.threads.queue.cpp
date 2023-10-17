#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>


template <typename T>
class boundedQueue
{
public:
    boundedQueue(size_t capacity) : capacity_(capacity) {}
    void put(T item)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]()
                 { return queue_.size() < capacity_; });
        queue_.push(std::move(item));
        cv_.notify_all();
    }
    T get()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]()
                 { return queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        cv_.notify_all();
        return item;
    }

private:
    std::queue<T> queue_;
    size_t capacity_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

boundedQueue<int> bounded_queue(5);
std::mutex cout_mutex;
void producer()
{
    for (int i = 0; i < 10; ++i)
    {
        bounded_queue.put(i);
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Producer put " << i << std::endl;
        }
    }
}
void consumer()
{
    for (int i = 0; i < 10; ++i)
    {
        int item = bounded_queue.get();
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Consumer got " << item << std::endl;
        }
    }
}

int main()
{
    std::thread t1(producer);
    std::thread t2(consumer);
    t1.join();
    t2.join();
    return 0;
}