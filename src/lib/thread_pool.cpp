#include "densitas/thread_pool.hpp"


namespace densitas {
namespace core {


thread_pool::thread_pool(int max_threads)
    : max_threads_(max_threads<1 ? 1 : max_threads), n_running_(0), futures_()
{}

thread_pool::~thread_pool()
{
    for (auto& future : futures_) future.get();
}

void thread_pool::wait_for_futures()
{
    while (n_running_ >= max_threads_) {
        for (int i=futures_.size()-1; i>=0; --i) {
            if (futures_[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                futures_[i].get();
                futures_.erase(futures_.begin() + i);
            }
        }
        n_running_ = futures_.size();
    }
}


} // core
} // densitas
