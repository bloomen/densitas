#include "utils.hpp"


COLLECTION(manipulation_assign_vector_to_row) {

auto function = densitas::core::assign_vector_to_row<double, matrix_t, vector_t>;

TEST(test_happy_path) {
    const auto row1 = mkrow({1, 2, 3});
    const auto row2 = mkrow({10, 20, 30});
    auto matrix = matrix_t(2, 3);
    matrix.row(0) = row1;
    matrix.row(1) = row2;
    const auto vector = mkcol({-1, -2, -3});
    function(matrix, 1, vector);
    assert_equal(2u, matrix.n_rows, SPOT);
    assert_equal(3u, matrix.n_cols, SPOT);
    assert_equal_containers(row1, arma::rowvec(matrix.row(0)), SPOT);
    assert_equal_containers(vector, vector_t(arma::trans(matrix.row(1))), SPOT);
}

TEST(test_row_index_too_big) {
    auto matrix = matrix_t(2, 3);
    const auto vector = mkcol({-1, -2, -3});
    assert_throw<densitas::densitas_error>([&]() { function(matrix, 2, vector); });
}

TEST(test_vector_not_matching_matrix) {
    auto matrix = matrix_t(2, 3);
    const auto vector = mkcol({-1, -2});
    assert_throw<densitas::densitas_error>([&]() { function(matrix, 1, vector); });
}

}
