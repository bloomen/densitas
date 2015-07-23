#include "utils.hpp"


struct tpool : densitas::core::thread_pool {

    using densitas::core::thread_pool::thread_pool;

    std::size_t get_max_threads()
    {
        return max_threads_;
    }

    std::size_t get_check_interval_ms()
    {
        return check_interval_ms_;
    }

    std::list<std::pair<std::shared_ptr<std::atomic_bool>, std::thread>>& get_threads()
    {
        return threads_;
    }

};


template<typename... Args>
struct functor {

    bool called;

    functor()
        : called(false)
    {}

    void operator()(Args...)
    {
        called = true;
    }

};


COLLECTION(thread_pool) {

TEST(test_construct) {
    tpool tp(42, 20);
    assert_equal(42u, tp.get_max_threads(), SPOT);
    assert_equal(20u, tp.get_check_interval_ms(), SPOT);
    assert_equal(0, tp.get_threads().size(), SPOT);
}

TEST(test_construct_with_weird_param) {
    tpool tp(-3, 0);
    assert_equal(1u, tp.get_max_threads(), SPOT);
}

TEST(test_launch_new_without_args) {
    functor<> func;
    {
        tpool tp(3, 0);
        tp.launch_new(std::ref(func));
        assert_equal(1, tp.get_threads().size(), SPOT);
    }
    assert_true(func.called, SPOT);
}

TEST(test_launch_new_with_some_args) {
    functor<int, double> func;
    {
        tpool tp(5, 0);
        tp.launch_new(std::ref(func), 1, 5.);
        assert_equal(1, tp.get_threads().size(), SPOT);
    }
    assert_true(func.called, SPOT);
}

TEST(test_launch_many) {
    functor<int, double> func;
    {
        tpool tp(4, 0);
        tp.launch_new(functor<>());
        tp.launch_new(functor<>());
        tp.launch_new(functor<>());
        tp.launch_new(functor<>());
        tp.launch_new(functor<>());
        tp.launch_new(functor<>());
        tp.launch_new(std::ref(func), 1, 5.);
    }
    assert_true(func.called, SPOT);
}

}
