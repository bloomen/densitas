#include "densitas/thread_pool.hpp"
#include <vector>


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

thread_pool::thread_pool(int max_threads)
    : max_threads_(max_threads<1 ? 1 : max_threads), threads_()
{}

void thread_pool::wait_for_threads()
{
    while (threads_.size() >= max_threads_-1) {
        for (auto it = threads_.begin(); it != threads_.cend();) {
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
