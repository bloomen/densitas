#include "utils.hpp"


COLLECTION(manipulation_predict_proba_for_row) {

auto function = densitas::core::predict_proba_for_row<double, vector_t, matrix_t, mock_model>;

TEST(test_happy_path) {
    auto model = mock_model();
    model.prediction = mkcol({0.666});
    const auto row1 = mkrow({1, 2, 3});
    const auto row2 = mkrow({10, 20, 30});
    auto X = matrix_t(2, 3);
    X.row(0) = row1;
    X.row(1) = row2;
    const auto proba = function(model, X, 1);
    assert_equal(0.666, proba, SPOT);
}

TEST(test_row_index_too_big) {
    auto model = mock_model();
    const auto X = matrix_t(2, 3);
    assert_throw<densitas::densitas_error>([&]() { function(model, X, 2); });
}

}
