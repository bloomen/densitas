#include "densitas/thread_pool.hpp"
#include <chrono>


namespace densitas {
namespace core {


condition_variable::condition_variable()
: flag_{false}, cond_var_{}, mutex_{}
{}

condition_variable::~condition_variable()
{}

void condition_variable::notify_one()
{
    {
        std::lock_guard<std::mutex> lock{mutex_};
        flag_ = true;
    }
    cond_var_.notify_one();
}

void condition_variable::wait()
{
    std::unique_lock<std::mutex> lock{mutex_};
    cond_var_.wait(lock, [this]() { return this->flag_; });
    flag_ = false;
}


thread_pool::thread_pool(int max_threads)
: max_threads_{static_cast<size_t>(max_threads<1 ? 1 : max_threads)}, threads_{}, cond_var_{}
{}

thread_pool::~thread_pool()
{
    for (auto& thread : threads_) {
        thread.second.join();
    }
}

void thread_pool::wait_for_slot()
{
    while (threads_.size() >= max_threads_) {
        cond_var_.wait();
        for (auto it=threads_.begin(); it!=threads_.cend();) {
            if (it->first->load()) {
                it->second.join();
                threads_.erase(it++);
            } else {
                ++it;
            }
        }
    }
}


} // core
} // densitas
