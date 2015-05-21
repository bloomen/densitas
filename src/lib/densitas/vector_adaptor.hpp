#pragma once
#include "type_check.hpp"


namespace densitas {
namespace vector_adaptor {


template<typename VectorType>
VectorType construct_uninitialized(size_t n_elem)
{
    return VectorType(n_elem);
}


template<typename VectorType>
size_t n_elements(const VectorType& vector)
{
    return static_cast<size_t>(vector.size());
}


template<typename ElementType, typename VectorType>
ElementType get_element(const VectorType& vector, size_t index)
{
    densitas::core::check_element_type<ElementType>();
    return static_cast<ElementType>(vector(index));
}


template<typename ElementType, typename VectorType>
void set_element(VectorType& vector, size_t index, ElementType value)
{
    densitas::core::check_element_type<ElementType>();
    vector(index) = value;
}


} // vector_adaptor
} // densitas
