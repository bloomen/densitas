#include "utils.hpp"


class mock_matrix {
public:

    const std::size_t n_rows_;
    const std::size_t n_cols_;
    std::size_t row_index_used_;
    std::size_t col_index_used_;
    double value_;

    mock_matrix(std::size_t n_rows, std::size_t n_cols)
        : n_rows_(n_rows), n_cols_(n_cols),
          row_index_used_(0), col_index_used_(0), value_(0)
    {}

    std::size_t n_rows() const
    {
        return n_rows_;
    }

    std::size_t n_cols() const
    {
        return n_cols_;
    }

    double operator()(std::size_t, std::size_t) const
    {
        return value_;
    }

    double& operator()(std::size_t i, std::size_t j)
    {
        row_index_used_ = i;
        col_index_used_ = j;
        return value_;
    }

};


COLLECTION(matrix_adapter) {

TEST(test_construct_uninitialized) {
    const auto n_rows = 2u;
    const auto n_cols = 3u;
    const auto matrix = densitas::matrix_adapter::construct_uninitialized<mock_matrix>(n_rows, n_cols);
    assert_equal(n_rows, matrix.n_rows_, SPOT);
    assert_equal(n_cols, matrix.n_cols_, SPOT);
}

TEST(test_n_rows) {
    const auto n_rows = 2u;
    const auto n_cols = 3u;
    const auto matrix = densitas::matrix_adapter::construct_uninitialized<mock_matrix>(n_rows, n_cols);
    const auto act_n_rows = densitas::matrix_adapter::n_rows(matrix);
    assert_equal(n_rows, act_n_rows, SPOT);
}

TEST(test_n_columns) {
    const auto n_rows = 2u;
    const auto n_cols = 3u;
    const auto matrix = densitas::matrix_adapter::construct_uninitialized<mock_matrix>(n_rows, n_cols);
    const auto act_n_cols = densitas::matrix_adapter::n_columns(matrix);
    assert_equal(n_cols, act_n_cols, SPOT);
}

TEST(test_get_element) {
    auto matrix = densitas::matrix_adapter::construct_uninitialized<mock_matrix>(2, 3);
    matrix.value_ = 14.3;
    const auto element = densitas::matrix_adapter::get_element<double>(matrix, 1, 2);
    assert_equal(matrix.value_, element, SPOT);
}

TEST(test_set_element) {
    auto matrix = densitas::matrix_adapter::construct_uninitialized<mock_matrix>(2, 3);
    const auto value = 14.3;
    const auto row = 1;
    const auto col = 2;
    densitas::matrix_adapter::set_element<double>(matrix, row, col, value);
    assert_equal(value, matrix.value_, SPOT);
    assert_equal(row, matrix.row_index_used_, SPOT);
    assert_equal(col, matrix.col_index_used_, SPOT);
}

}
