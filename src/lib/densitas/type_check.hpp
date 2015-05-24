#pragma once
#include <type_traits>


namespace densitas {
namespace core {


template<typename ElementType>
void check_element_type()
{
    static_assert(std::is_floating_point<ElementType>::value, "element type is not a floating point type");
}


} // core
} // densitas
