#pragma once
#include <thread>
#include <memory>
#include <atomic>
#include <list>


namespace densitas {
namespace core {


class runner {
public:

    explicit
    runner(std::shared_ptr<std::atomic_bool> done);

    template<typename Functor, typename... Args>
    void operator()(Functor&& functor, Args&&... args)
    {
        functor(std::forward<Args>(args)...);
        done_->store(true);
    }

private:
    std::shared_ptr<std::atomic_bool> done_;
};


class thread_pool {
public:

    thread_pool(int max_threads, std::size_t check_interval_ms);

    virtual ~thread_pool();

    void wait_for_slot();

    template<typename Functor, typename... Args>
    void launch_new(Functor&& functor, Args&&... args)
    {
        wait_for_slot();
        auto done = std::make_shared<std::atomic_bool>(false);
        std::thread thread(densitas::core::runner(done), std::forward<Functor>(functor), std::forward<Args>(args)...);
        threads_.push_back(std::make_pair(done, std::move(thread)));
    }

    thread_pool(const thread_pool&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;
    thread_pool(thread_pool&&) = delete;
    thread_pool& operator=(thread_pool&&) = delete;

protected:
    const std::size_t max_threads_;
    const std::size_t check_interval_ms_;
    std::list<std::pair<std::shared_ptr<std::atomic_bool>, std::thread>> threads_;
};


} // core
} // densitas
