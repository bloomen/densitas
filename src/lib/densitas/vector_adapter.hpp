#pragma once
#include "type_check.hpp"


namespace densitas {
namespace vector_adapter {

/**
 * Constructs a new, uninitialized vector
 */
template<typename VectorType>
VectorType construct_uninitialized(std::size_t n_elem)
{
    return VectorType(n_elem);
}

/**
 * Returns the number of elements
 */
template<typename VectorType>
std::size_t n_elements(const VectorType& vector)
{
    return static_cast<std::size_t>(vector.size());
}

/**
 * Returns the element at given index
 */
template<typename ElementType, typename VectorType>
ElementType get_element(const VectorType& vector, std::size_t index)
{
    densitas::core::check_element_type<ElementType>();
    return static_cast<ElementType>(vector(index));
}

/**
 * Sets the element at given index
 */
template<typename ElementType, typename VectorType>
void set_element(VectorType& vector, std::size_t index, ElementType value)
{
    densitas::core::check_element_type<ElementType>();
    vector(index) = value;
}


} // vector_adapter
} // densitas
