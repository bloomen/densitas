#include "utils.hpp"


COLLECTION(density_error) {

TEST(test_exception) {
    const std::string message = "something";
    try {
        throw densitas::densitas_error(message);
    } catch (const std::runtime_error& msg) {
        assert_equal(message, msg.what(), SPOT);
    }
}

}
