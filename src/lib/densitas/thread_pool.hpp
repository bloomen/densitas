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


class thread {
public:

    thread()
        : done_(std::make_shared<std::atomic_bool>(false)), thread_()
    {}

    template<typename Functor, typename... Args>
    explicit
    thread(Functor&& functor, Args&&... args)
        : thread()
    {
        densitas::core::runner runner(done_);
        thread_ = std::thread(std::move(runner), std::forward<Functor>(functor), std::forward<Args>(args)...);
    }

    ~thread();

    std::shared_ptr<std::atomic_bool> done();

private:
    std::shared_ptr<std::atomic_bool> done_;
    std::thread thread_;
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
        threads_.emplace(std::piecewise_construct,
                         std::forward_as_tuple(threads_.size()),
                         std::forward_as_tuple(std::forward<Functor>(functor), std::forward<Args>(args)...));
        ++n_running_;
    }

    thread_pool(const thread_pool&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;
    thread_pool(thread_pool&&) = delete;
    thread_pool& operator=(thread_pool&&) = delete;

protected:
    const size_t max_threads_;
    size_t n_running_;
    std::unordered_map<size_t, densitas::core::thread> threads_;

    void wait_for_threads();
};


} // core
} // densitas
