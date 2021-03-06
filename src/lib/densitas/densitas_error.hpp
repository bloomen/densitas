#pragma once
#include <stdexcept>
#include <string>


namespace densitas {


class densitas_error : public std::runtime_error {
public:
    explicit
    densitas_error(const std::string& message);
};


} // densitas
