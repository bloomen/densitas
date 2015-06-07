#pragma once
#include <future>
#include <vector>


namespace densitas {
namespace core {


class thread_pool {
public:

    explicit
    thread_pool(int max_threads);

    virtual ~thread_pool();

    template<typename Functor, typename... Args>
    void launch_new(Functor&& functor, Args&&... args)
    {
        wait_for_futures();
        futures_.push_back(std::async(std::launch::async, std::forward<Functor>(functor), std::forward<Args>(args)...));
        ++n_running_;
    }

    thread_pool(const thread_pool&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;
    thread_pool(thread_pool&&) = delete;
    thread_pool& operator=(thread_pool&&) = delete;

protected:
    const size_t max_threads_;
    size_t n_running_;
    std::vector<std::future<void>> futures_;

    void wait_for_futures();
};


} // core
} // densitas
