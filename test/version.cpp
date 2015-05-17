#include "utils.hpp"


COLLECTION(version) {

TEST(test_length) {
    assert_equal(5u, densitas::version().size(), SPOT);
}

}
