#include "densitas/thread_pool.hpp"


namespace densitas {
namespace core {


runner::runner(std::shared_ptr<std::atomic_bool> done)
    : done_(std::move(done))
{}


thread::~thread()
{
    if (thread_.get_id() != std::thread::id())
        thread_.join();
}

std::shared_ptr<std::atomic_bool> thread::done()
{
    return done_;
}


thread_pool::thread_pool(int max_threads)
    : max_threads_(max_threads<1 ? 1 : max_threads), n_running_(0), threads_()
{}

thread_pool::~thread_pool()
{}

void thread_pool::wait_for_threads()
{
    while (n_running_ >= max_threads_-1) {
        for (size_t i=0; i<threads_.size(); ++i) {
            if (threads_[i].done())
                threads_.erase(i);
        }
        n_running_ = threads_.size();
    }
}


} // core
} // densitas
