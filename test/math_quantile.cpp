#include "utils.hpp"


COLLECTION(math_quantile) {

auto function = densitas::math::quantile<double>;

TEST(test_happy_path) {
    auto data = std::vector<double>{1, 1.5, 2, 2.7, 3, 3.1, 4, 4.7, 5};
    const auto eps = 1e-15;
    assert_approx_equal(1, function(data, 0.), eps, SPOT);
    assert_approx_equal(1, function(data, 0.1), eps, SPOT);
    assert_approx_equal(1.4, function(data, 0.2), eps, SPOT);
    assert_approx_equal(1.85, function(data, 0.3), eps, SPOT);
    assert_approx_equal(2.42, function(data, 0.4), eps, SPOT);
    assert_approx_equal(2.85, function(data, 0.5), eps, SPOT);
    assert_approx_equal(3.04, function(data, 0.6), eps, SPOT);
    assert_approx_equal(3.37, function(data, 0.7), eps, SPOT);
    assert_approx_equal(4.14, function(data, 0.8), eps, SPOT);
    assert_approx_equal(4.73, function(data, 0.9), eps, SPOT);
    assert_approx_equal(5, function(data, 1.), eps, SPOT);
}

TEST(test_vector_with_no_contents) {
    auto data = std::vector<double>();
    assert_throw<densitas::densitas_error>([&]() { function(data, 1); }, SPOT);
}

TEST(test_vector_with_one_value) {
    auto data = std::vector<double>{3.7};
    const auto eps = 1e-15;
    assert_approx_equal(3.7, function(data, 0.5), eps, SPOT);
}

TEST(test_proba_too_big) {
    auto data = std::vector<double>{1, 2, 3};
    assert_throw<densitas::densitas_error>([&]() { function(data, 1.1); }, SPOT);
}

TEST(test_proba_too_small) {
    auto data = std::vector<double>{1, 2, 3};
    assert_throw<densitas::densitas_error>([&]() { function(data, -0.1); }, SPOT);
}

}
