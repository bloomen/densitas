#include "utils.hpp"


COLLECTION(math_centers) {

auto function = densitas::math::centers<double, vector_t>;

TEST(test_happy_path) {
    const auto vector = vector_t{0, 1, 2};
    const auto centers = function(vector);
    const auto expected = vector_t{0.5, 1.5};
    assert_equal_containers(expected, centers, SPOT);
}

TEST(test_only_one_value) {
    const auto vector = vector_t{1};
    assert_throw<densitas::densitas_error>([&]() { function(vector); });
}

TEST(test_no_value) {
    const auto vector = vector_t();
    assert_throw<densitas::densitas_error>([&]() { function(vector); });
}

}
