#include "densitas/thread_pool.hpp"
#include <chrono>


namespace densitas {
namespace core {


runner::runner(std::shared_ptr<std::atomic_bool> done)
: done_(std::move(done))
{}


thread_pool::~thread_pool()
{
    for (auto& thread : threads_) {
        thread.second.join();
    }
}

thread_pool::thread_pool(int max_threads, int check_interval_ms)
: max_threads_(max_threads<1 ? 1 : max_threads),
  check_interval_ms_(check_interval_ms<0 ? 0 : check_interval_ms), threads_{}
{}

void thread_pool::wait_for_slot()
{
    while (threads_.size() >= max_threads_) {
        for (auto it=threads_.begin(); it!=threads_.cend();) {
            if (it->first->load()) {
                it->second.join();
                threads_.erase(it++);
            } else {
                ++it;
            }
        }
        if (threads_.size() >= max_threads_)
            std::this_thread::sleep_for(std::chrono::milliseconds(check_interval_ms_));
    }
}


} // core
} // densitas
