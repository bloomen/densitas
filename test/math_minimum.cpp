#include "utils.hpp"


COLLECTION(math_minimum) {

auto function = densitas::math::minimum<double, vector_t>;

TEST(test_happy_path) {
    const auto vector = mkcol({2, 1, 3});
    const auto minimum = function(vector);
    assert_equal(1u, minimum, SPOT);
}

TEST(test_with_vector_of_size_one) {
    const auto vector = mkcol({2});
    const auto minimum = function(vector);
    assert_equal(2u, minimum, SPOT);
}

TEST(test_with_vector_of_size_zero) {
    const auto vector = vector_t();
    assert_throw<densitas::densitas_error>([&]() { function(vector); });
}

}
