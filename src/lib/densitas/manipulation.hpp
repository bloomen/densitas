#pragma once
#include "matrix_adapter.hpp"
#include "vector_adapter.hpp"
#include "densitas_error.hpp"


namespace densitas {
namespace core {


template<typename ElementType, typename MatrixType, typename VectorType>
void assign_vector_to_row(MatrixType& matrix, size_t row_index, const VectorType& vector)
{
    densitas::core::check_element_type<ElementType>();
    const auto n_rows = densitas::matrix_adapter::n_rows(matrix);
    if (row_index > n_rows-1)
        throw densitas::densitas_error("row index larger than rows in matrix: " + std::to_string(row_index));
    const auto n_elem = densitas::vector_adapter::n_elements(vector);
    const auto n_cols = densitas::matrix_adapter::n_columns(matrix);
    if (n_cols != n_elem)
        throw densitas::densitas_error("size of vector not matching number of columns in matrix");
    for (size_t i=0; i<n_cols; ++i) {
        const auto value = densitas::vector_adapter::get_element<ElementType>(vector, i);
        densitas::matrix_adapter::set_element<ElementType>(matrix, row_index, i, value);
    }
}


template<typename ElementType, typename VectorType, typename MatrixType>
VectorType extract_row(const MatrixType& matrix, size_t row_index)
{
    densitas::core::check_element_type<ElementType>();
    const auto n_rows = densitas::matrix_adapter::n_rows(matrix);
    if (row_index > n_rows-1)
        throw densitas::densitas_error("row index larger than rows in matrix: " + std::to_string(row_index));
    const auto n_cols = densitas::matrix_adapter::n_columns(matrix);
    auto vector = densitas::vector_adapter::construct_uninitialized<VectorType>(n_cols);
    for (size_t i=0; i<n_cols; ++i) {
        const auto value = densitas::matrix_adapter::get_element<ElementType>(matrix, row_index, i);
        densitas::vector_adapter::set_element<ElementType>(vector, i, value);
    }
    return vector;
}


template<typename ElementType, typename VectorType, typename MatrixType, typename ModelType>
ElementType predict_proba_for_row(ModelType& model, const MatrixType& X, size_t row_index)
{
    const auto n_cols = densitas::matrix_adapter::n_columns(X);
    const auto feature_row = densitas::core::extract_row<ElementType, VectorType>(X, row_index);
    auto feature_matrix = densitas::matrix_adapter::construct_uninitialized<MatrixType>(1, n_cols);
    densitas::core::assign_vector_to_row<ElementType>(feature_matrix, 0, feature_row);
    const auto prob_pred = densitas::model_adapter::predict_proba<VectorType>(model, feature_matrix);
    return densitas::vector_adapter::get_element<ElementType>(prob_pred, 0);
}


} // core
} // densitas
