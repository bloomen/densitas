#include "utils.hpp"


COLLECTION(math_quantiles) {

auto function = densitas::math::quantiles<double, vector_t>;

TEST(test_happy_path) {
    const auto data = vector_t{1, 1.5, 2, 2.7, 3, 3.1, 4, 4.7, 5};
    const auto probas = vector_t{0, 0.1, 0.2};
    const auto quantiles = function(data, probas);
    const auto eps = 1e-15;
    const auto expected = vector_t{1, 1, 1.4};
    assert_approx_equal_containers(expected, quantiles, eps, SPOT);
}

TEST(test_with_no_probas) {
    const auto data = vector_t{1, 2, 3};
    const auto probas = vector_t{};
    const auto quantiles = function(data, probas);
    const auto expected = vector_t{};
    assert_equal_containers(expected, quantiles, SPOT);
}

}
