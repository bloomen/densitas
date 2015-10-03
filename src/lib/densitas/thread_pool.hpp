#pragma once
#include <thread>
#include <memory>
#include <atomic>
#include <list>
#include <mutex>
#include <condition_variable>


namespace densitas {
namespace core {


class condition_variable {
public:

    condition_variable();

    virtual ~condition_variable();

    void notify_one();

    void wait();

    condition_variable(const condition_variable&) = delete;
    condition_variable& operator=(const condition_variable&) = delete;
    condition_variable(condition_variable&&) = delete;
    condition_variable& operator=(condition_variable&&) = delete;

protected:
    bool flag_;
    std::condition_variable cond_var_;
    std::mutex mutex_;
};


template<typename ConditionVariable=densitas::core::condition_variable>
class functor_runner {
public:

    functor_runner(std::shared_ptr<std::atomic_bool> done, ConditionVariable& cond_var)
    : done_{done}, cond_var_(cond_var)
    {}

    template<typename Functor, typename... Args>
    void operator()(Functor&& functor, Args&&... args)
    {
        std::forward<Functor>(functor)(std::forward<Args>(args)...);
        *done_ = true;
        cond_var_.notify_one();
    }

private:
    std::shared_ptr<std::atomic_bool> done_;
    ConditionVariable& cond_var_;
};


class thread_pool {
public:

    thread_pool(int max_threads);

    virtual ~thread_pool();

    void wait_for_slot();

    template<typename Functor, typename... Args>
    void launch_new(Functor&& functor, Args&&... args)
    {
        wait_for_slot();
        auto done = std::make_shared<std::atomic_bool>(false);
        threads_.emplace_back(done, std::thread{densitas::core::functor_runner<>{done, cond_var_},
                                                std::forward<Functor>(functor), std::forward<Args>(args)...});
    }

    thread_pool(const thread_pool&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;
    thread_pool(thread_pool&&) = delete;
    thread_pool& operator=(thread_pool&&) = delete;

protected:
    const std::size_t max_threads_;
    std::list<std::pair<std::shared_ptr<std::atomic_bool>, std::thread>> threads_;
    densitas::core::condition_variable cond_var_;
};


} // core
} // densitas
