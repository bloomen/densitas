#include "densitas/densitas_error.hpp"


namespace densitas {


densitas_error::densitas_error(const std::string& message)
    : std::runtime_error(message)
{}


} // densitas
