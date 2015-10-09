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


struct task {
    task(std::shared_ptr<std::atomic_bool> done, std::thread&& thread);
    std::shared_ptr<std::atomic_bool> done;
    std::thread thread;
};


class task_manager {
public:

    task_manager(int max_tasks);

    virtual ~task_manager();

    void wait_for_slot();

    template<typename Functor, typename... Args>
    void launch_new(Functor&& functor, Args&&... args)
    {
        wait_for_slot();
        auto done = std::make_shared<std::atomic_bool>(false);
        auto runner = densitas::core::functor_runner<>{done, cond_var_};
        tasks_.emplace_back(done, std::thread{std::move(runner), std::forward<Functor>(functor), std::forward<Args>(args)...});
    }

    task_manager(const task_manager&) = delete;
    task_manager& operator=(const task_manager&) = delete;
    task_manager(task_manager&&) = delete;
    task_manager& operator=(task_manager&&) = delete;

protected:
    const std::size_t max_tasks_;
    std::list<densitas::core::task> tasks_;
    densitas::core::condition_variable cond_var_;
};


} // core
} // densitas
