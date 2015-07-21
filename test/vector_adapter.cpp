#include "utils.hpp"


class mock_vector {
public:

    const std::size_t size_;
    std::size_t index_used_;
    double value_;

    explicit
    mock_vector(std::size_t size)
        : size_(size), index_used_(0), value_(0)
    {}

    std::size_t size() const
    {
        return size_;
    }

    double operator()(std::size_t) const
    {
        return value_;
    }

    double& operator()(std::size_t index)
    {
        index_used_ = index;
        return value_;
    }

};


COLLECTION(vector_adapter) {

TEST(test_construct_uninitialized) {
    const auto size = 3u;
    const auto vector = densitas::vector_adapter::construct_uninitialized<mock_vector>(size);
    assert_equal(size, vector.size_, SPOT);
}

TEST(test_n_elements) {
    const auto size = 3u;
    const auto vector = densitas::vector_adapter::construct_uninitialized<mock_vector>(size);
    const auto n_elem = densitas::vector_adapter::n_elements(vector);
    assert_equal(size, n_elem, SPOT);
}

TEST(test_get_element) {
    const auto size = 3u;
    auto vector = densitas::vector_adapter::construct_uninitialized<mock_vector>(size);
    vector.value_ = 14.3;
    const auto element = densitas::vector_adapter::get_element<double>(vector, 2);
    assert_equal(vector.value_, element, SPOT);
}

TEST(test_set_element) {
    const auto size = 3u;
    auto vector = densitas::vector_adapter::construct_uninitialized<mock_vector>(size);
    const auto index = 2u;
    const auto value = 14.3;
    densitas::vector_adapter::set_element<double>(vector, index, value);
    assert_equal(value, vector.value_, SPOT);
    assert_equal(index, vector.index_used_, SPOT);
}

}
