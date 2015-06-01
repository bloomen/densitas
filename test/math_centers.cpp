#include "utils.hpp"


COLLECTION(math_centers) {

auto function = densitas::math::centers<double, vector_t>;

TEST(test_happy_path) {
    const auto data = vector_t{0, 0.2, 0.8, 1, 1.5, 2};
    const auto quantiles = vector_t{0, 1, 2};
    const auto centers = function(data, quantiles);
    const auto expected = vector_t{0.5, 1.5};
    assert_equal_containers(expected, centers, SPOT);
}

TEST(test_only_one_value) {
    const auto data = vector_t{0, 0.2, 1};
    const auto quantiles = vector_t{1};
    assert_throw<densitas::densitas_error>([&]() { function(data, quantiles); });
}

TEST(test_no_value) {
    const auto data = vector_t{0, 0.2, 1};
    const auto quantiles = vector_t();
    assert_throw<densitas::densitas_error>([&]() { function(data, quantiles); });
}

TEST(test_zero_size_data) {
    const auto data = vector_t{};
    const auto quantiles = vector_t{0, 1, 2};
    assert_throw<densitas::densitas_error>([&]() { function(data, quantiles); });
}

}
