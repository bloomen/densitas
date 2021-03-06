#include "utils.hpp"


COLLECTION(math_make_classification_target) {

auto function = densitas::math::make_classification_target<mock_model, double, vector_t>;

TEST(test_happy_path) {
    const auto y = mkcol({1, 2, 3, 4});
    const auto lower = 1.5;
    const auto upper = 3.5;
    const auto target = function(y, lower, upper);
    const auto expected = mkcol({dno, dyes, dyes, dno});
    assert_equal_containers(expected, target, SPOT);
}

TEST(test_with_zero_length_vector) {
    const auto y = mkcol({});
    const auto lower = 1.5;
    const auto upper = 3.5;
    const auto target = function(y, lower, upper);
    const auto expected = mkcol({});
    assert_equal_containers(expected, target, SPOT);
}

TEST(test_all_no) {
    const auto y = mkcol({1, 2, 3, 4});
    const auto lower = 4.5;
    const auto upper = 9.;
    const auto target = function(y, lower, upper);
    const auto expected = mkcol({dno, dno, dno, dno});
    assert_equal_containers(expected, target, SPOT);
}

TEST(test_all_yes) {
    const auto y = mkcol({1, 2, 3, 4});
    const auto lower = 1.;
    const auto upper = 4.;
    const auto target = function(y, lower, upper);
    const auto expected = mkcol({dyes, dyes, dyes, dyes});
    assert_equal_containers(expected, target, SPOT);
}

}
