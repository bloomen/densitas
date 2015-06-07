#include "densitas/thread_pool.hpp"


namespace densitas {
namespace core {


runner::runner(std::shared_ptr<std::atomic_bool> done)
    : done_(std::move(done))
{}


thread_pool::~thread_pool()
{
    for (auto& thread : threads_) {
        thread.second.second.join();
    }
}

thread_pool::thread_pool(int max_threads)
    : max_threads_(max_threads<1 ? 1 : max_threads), threads_()
{}

void thread_pool::wait_for_threads()
{
    while (threads_.size() >= max_threads_-1) {
        for (auto& thread : threads_) {
            if (thread.second.first->load()) {
                thread.second.second.join();
                threads_.erase(thread.first);
            }
        }
    }
}


} // core
} // densitas
