#pragma once
#include "type_check.hpp"


namespace densitas {
namespace matrix_adapter {

/**
 * Constructs a new, uninitialized matrix
 */
template<typename MatrixType>
MatrixType construct_uninitialized(size_t n_rows, size_t n_cols)
{
    return MatrixType(n_rows, n_cols);
}

/**
 * Returns the number of rows
 */
template<typename MatrixType>
size_t n_rows(const MatrixType& matrix)
{
    return static_cast<size_t>(matrix.n_rows());
}

/**
 * Returns the number of columns
 */
template<typename MatrixType>
size_t n_columns(const MatrixType& matrix)
{
    return static_cast<size_t>(matrix.n_cols());
}

/**
 * Returns the element at given row and column index
 */
template<typename ElementType, typename MatrixType>
ElementType get_element(const MatrixType& matrix, size_t row_index, size_t col_index)
{
    densitas::core::check_element_type<ElementType>();
    return static_cast<ElementType>(matrix(row_index, col_index));
}

/**
 * Sets the element at given row and column index
 */
template<typename ElementType, typename MatrixType>
void set_element(MatrixType& matrix, size_t row_index, size_t col_index, ElementType value)
{
    densitas::core::check_element_type<ElementType>();
    matrix(row_index, col_index) = value;
}


} // matrix_adapter
} // densitas
