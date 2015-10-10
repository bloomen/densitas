#include "densitas/task_manager.hpp"


namespace densitas {
namespace core {


condition_variable::condition_variable()
: flag_{false}, cond_var_{}, mutex_{}
{}

condition_variable::~condition_variable()
{}

void condition_variable::notify_one()
{
    {
        std::lock_guard<std::mutex> lock{mutex_};
        flag_ = true;
    }
    cond_var_.notify_one();
}

void condition_variable::wait()
{
    std::unique_lock<std::mutex> lock{mutex_};
    cond_var_.wait(lock, [this]() { return this->flag_; });
    flag_ = false;
}


task::task(std::shared_ptr<std::atomic_bool> done, std::thread&& thread)
: done{done}, thread{std::move(thread)}
{}


task_manager::task_manager(int max_tasks)
: max_tasks_{static_cast<std::size_t>(max_tasks<1 ? 1 : max_tasks)}, tasks_{}, cond_var_{}
{}

task_manager::~task_manager()
{
    for (auto& task : tasks_) {
        task.thread.join();
    }
}

void task_manager::wait_for_slot()
{
    while (tasks_.size() >= max_tasks_) {
        cond_var_.wait();
        for (auto it=tasks_.begin(); it!=tasks_.cend();) {
            if (it->done->load()) {
                it->thread.join();
                tasks_.erase(it++);
            } else {
                ++it;
            }
        }
    }
}


} // core
} // densitas
