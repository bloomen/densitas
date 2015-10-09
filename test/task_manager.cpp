#include "utils.hpp"


struct tman : densitas::core::task_manager {

    tman(int max_threads)
    : densitas::core::task_manager{max_threads}
    {}

    std::size_t get_max_tasks()
    {
        return max_tasks_;
    }

    std::list<densitas::core::task>& get_tasks()
    {
        return tasks_;
    }

};


template<typename... Args>
struct functor {

    bool called;

    functor()
    : called{false}
    {}

    void operator()(Args...)
    {
        called = true;
    }

};


COLLECTION(task_manager) {

TEST(test_construct) {
    tman tm(42);
    assert_equal(42u, tm.get_max_tasks(), SPOT);
    assert_equal(0, tm.get_tasks().size(), SPOT);
}

TEST(test_construct_with_weird_param) {
    tman tm(-3);
    assert_equal(1u, tm.get_max_tasks(), SPOT);
}

TEST(test_launch_new_without_args) {
    functor<> func;
    {
        tman tm(3);
        tm.launch_new(std::ref(func));
        assert_equal(1, tm.get_tasks().size(), SPOT);
    }
    assert_true(func.called, SPOT);
}

TEST(test_launch_new_with_some_args) {
    functor<int, double> func;
    {
        tman tm(5);
        tm.launch_new(std::ref(func), 1, 5.);
        assert_equal(1, tm.get_tasks().size(), SPOT);
    }
    assert_true(func.called, SPOT);
}

TEST(test_launch_many) {
    functor<int, double> func;
    {
        tman tm(4);
        tm.launch_new(functor<>());
        tm.launch_new(functor<>());
        tm.launch_new(functor<>());
        tm.launch_new(functor<>());
        tm.launch_new(functor<>());
        tm.launch_new(functor<>());
        tm.launch_new(std::ref(func), 1, 5.);
    }
    assert_true(func.called, SPOT);
}

struct cond_var_mock {

    bool called;

    cond_var_mock()
    : called{false}
    {}

    void notify_one()
    {
        called = true;
    }

};

COLLECTION(functor_runner) {

struct fixture {

    std::shared_ptr<std::atomic_bool> done_;
    cond_var_mock cond_var_;
    densitas::core::functor_runner<cond_var_mock> runner_;
    bool called_;

    fixture()
    : done_(std::make_shared<std::atomic_bool>(false)),
      cond_var_{}, runner_{done_, cond_var_}, called_{false}
    {}

    virtual ~fixture() UNITTEST_NOEXCEPT_FALSE
    {
        assert_true(called_, SPOT);
        assert_true(*done_, SPOT);
        assert_true(cond_var_.called, SPOT);
    }

};

TEST_FIXTURE(fixture, test_with_no_args) {
    runner_([this](){ called_ = true; });
}

TEST_FIXTURE(fixture, test_with_multiple_args) {
    runner_([this](int, double){ called_ = true; }, 42, 1.3);
}

}

struct cv : densitas::core::condition_variable {

    bool get_flag()
    {
        return flag_;
    }

};

COLLECTION(condition_variable) {

TEST(test_constructor) {
    cv cond_var;
    assert_false(cond_var.get_flag(), SPOT);
}

TEST(test_notify_one) {
    cv cond_var;
    cond_var.notify_one();
    assert_true(cond_var.get_flag(), SPOT);
}

TEST(test_notify_one_and_wait) {
    cv cond_var;
    cond_var.notify_one();
    cond_var.wait();
    assert_false(cond_var.get_flag(), SPOT);
}

TEST(test_notify_one_and_wait_in_separate_threads) {
    cv cond_var;
    std::thread([&cond_var]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        cond_var.notify_one();
    }).detach();
    cond_var.wait();
    assert_false(cond_var.get_flag(), SPOT);
}

}
}
