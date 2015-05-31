#pragma once
#include <libunittest/all.hpp>
#include <densitas/all.hpp>
#include <armadillo>
using namespace unittest::assertions;


using vector_t = arma::vec;

using matrix_t = arma::mat;


struct mock_model {

    vector_t prediction;
    vector_t train_y;
    matrix_t train_X;

    mock_model()
        : prediction(), train_y(), train_X()
    {}

    void train(matrix_t& X, vector_t& y)
    {
        train_y = y;
        train_X = X;
    }

    vector_t predict_proba(matrix_t&) const
    {
        return prediction;
    }

};


constexpr int no = densitas::model_adapter::no<mock_model>();

constexpr int yes = densitas::model_adapter::yes<mock_model>();


namespace densitas {
namespace matrix_adapter {

template<>
inline
size_t n_rows(const matrix_t& matrix)
{
    return matrix.n_rows;
}

template<>
inline
size_t n_columns(const matrix_t& matrix)
{
    return matrix.n_cols;
}

} // matrix_adapter

namespace vector_adapter {

template<>
inline
size_t n_elements(const vector_t& vector)
{
    return vector.n_elem;
}

} // vector_adapter
} // densitas


namespace unittest {
namespace assertions {

template<typename... Args>
void
assert_equal_containers(const matrix_t& expected,
                        const matrix_t& actual,
                        const Args&... message)
{
    assert_equal(expected.n_rows, actual.n_rows, "n_rows don't match! ", message...);
    assert_equal(expected.n_cols, actual.n_cols, "n_cols don't match! ", message...);
    for (size_t i=0; i<expected.n_rows; ++i) {
        if (!unittest::core::is_containers_equal(vector_t(expected.row(i).t()), vector_t(actual.row(i).t()))) {
            const std::string text = "matrices are not equal";
            unittest::fail(UNITTEST_FUNC, text, message...);
        }
    }
}

template<typename... Args>
void
assert_approx_equal_containers(const matrix_t& expected,
                               const matrix_t& actual,
                               const double eps,
                               const Args&... message)
{
    assert_equal(expected.n_rows, actual.n_rows, "n_rows don't match! ", message...);
    assert_equal(expected.n_cols, actual.n_cols, "n_cols don't match! ", message...);
    for (size_t i=0; i<expected.n_rows; ++i) {
        if (!unittest::core::is_containers_approx_equal(vector_t(expected.row(i).t()), vector_t(actual.row(i).t()), eps)) {
            const std::string text = "matrices are not approx. equal";
            unittest::fail(UNITTEST_FUNC, text, message...);
        }
    }
}

} // unittest
} // assertions
