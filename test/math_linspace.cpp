#include "utils.hpp"


COLLECTION(math_linspace) {

auto function = densitas::math::linspace<vector_t, double>;

TEST(test_happy_path) {
    const double start = 0;
    const double end = 1;
    const std::size_t n = 5;
    const auto linspace = function(start, end, n);
    const auto expected = vector_t{0, 0.25, 0.5, 0.75, 1};
    assert_equal_containers(expected, linspace, SPOT);
}

TEST(test_for_end_not_larger_than_start) {
    const double start = 1;
    const double end = 1;
    const std::size_t n = 5;
    assert_throw<densitas::densitas_error>([&]() { function(start, end, n); }, SPOT);
}

TEST(test_for_n_not_larger_than_one) {
    const double start = 0;
    const double end = 1;
    const std::size_t n = 1;
    assert_throw<densitas::densitas_error>([&]() { function(start, end, n); }, SPOT);
}

}
