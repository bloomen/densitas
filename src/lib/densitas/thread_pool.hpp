#pragma once
#include <thread>
#include <unordered_map>
#include <memory>
#include <atomic>


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

    explicit
    thread_pool(int max_threads);

    virtual ~thread_pool();

    template<typename Functor, typename... Args>
    void launch_new(Functor&& functor, Args&&... args)
    {
        wait_for_threads();
        auto done = std::make_shared<std::atomic_bool>(false);
        std::thread thread(densitas::core::runner(done), std::forward<Functor>(functor), std::forward<Args>(args)...);
        threads_.emplace(threads_.size(), std::make_pair(done, std::move(thread)));
    }

    thread_pool(const thread_pool&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;
    thread_pool(thread_pool&&) = delete;
    thread_pool& operator=(thread_pool&&) = delete;

protected:
    const size_t max_threads_;
    std::unordered_map<size_t, std::pair<std::shared_ptr<std::atomic_bool>, std::thread>> threads_;

    void wait_for_threads();
};


} // core
} // densitas
