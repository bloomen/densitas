#include "utils.hpp"


COLLECTION(manipulation_extract_row) {

auto function = densitas::core::extract_row<double, vector_t, matrix_t>;

TEST(test_happy_path) {
    const auto row1 = mkrow({1, 2, 3});
    const auto row2 = mkrow({10, 20, 30});
    auto matrix = matrix_t(2, 3);
    matrix.row(0) = row1;
    matrix.row(1) = row2;
    const auto extracted_row = function(matrix, 1);
    assert_equal_containers(row2, extracted_row, SPOT);
}

TEST(test_row_index_too_big) {
    auto matrix = matrix_t(2, 3);
    assert_throw<densitas::densitas_error>([&]() { function(matrix, 2); });
}

}
