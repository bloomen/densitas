#include "utils.hpp"


COLLECTION(math_quantiles_weighted) {

auto function = densitas::math::quantiles_weighted<double, vector_t>;
const double default_accuracy = 1e-2;

TEST(test_happy_path) {
    const auto data = vector_t{1, 2, 3};
    const auto weights = vector_t{1, 0.5, 0.5};
    const auto probas = vector_t{0, 0.8};
    const auto quantiles = function(data, weights, probas, default_accuracy);
    const auto eps = 1e-15;
    const auto expected = vector_t{1, 2.2};
    assert_approx_equal_containers(expected, quantiles, eps, SPOT);
}

TEST(test_vector_and_weights_different_size) {
    const auto data = vector_t{1, 2, 3};
    const auto weights = vector_t{1, 0.5};
    const auto probas = vector_t{0, 0.8};
    assert_throw<densitas::densitas_error>([&]() { function(data, weights, probas, default_accuracy); });
}

TEST(test_all_weights_zero) {
    const auto data = vector_t{1, 2, 3};
    const auto weights = vector_t{0, 0, 0};
    const auto probas = vector_t{0, 0.8};
    const auto quantiles = function(data, weights, probas, default_accuracy);
    const auto eps = 1e-15;
    const auto expected = vector_t{1, 2.4};
    assert_approx_equal_containers(expected, quantiles, eps, SPOT);
}

TEST(test_accuracy_too_big) {
    const auto data = vector_t{1, 2, 3};
    const auto weights = vector_t{1, 0.5, 0.5};
    const auto probas = vector_t{0, 0.8};
    assert_throw<densitas::densitas_error>([&]() { function(data, weights, probas, 1); });
}

TEST(test_accuracy_too_small) {
    const auto data = vector_t{1, 2, 3};
    const auto weights = vector_t{1, 0.5, 0.5};
    const auto probas = vector_t{0, 0.8};
    assert_throw<densitas::densitas_error>([&]() { function(data, weights, probas, 0); });
}

}
