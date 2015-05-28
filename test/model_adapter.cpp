#include "utils.hpp"


COLLECTION(model_adapter) {

TEST(test_train) {
    auto model = mock_model();
    auto X = matrix_t(1, 2);
    X.row(0) = vector_t{-1, -2}.t();
    auto y = vector_t{3, 4.5};
    densitas::model_adapter::train(model, X, y);
    assert_equal_containers(X, model.train_X, SPOT);
    assert_equal_containers(y, model.train_y, SPOT);
}

TEST(test_predict_proba) {
    auto model = mock_model();
    model.prediction = vector_t{3, 4};
    auto X = matrix_t(1, 2);
    X.row(0) = vector_t{-1, -2}.t();
    const auto prediction = densitas::model_adapter::predict_proba<vector_t>(model, X);
    assert_equal_containers(model.prediction, prediction, SPOT);
}

TEST(test_yes) {
    assert_equal(1, densitas::model_adapter::yes<mock_model>());
}

TEST(test_no) {
    assert_equal(-1, densitas::model_adapter::no<mock_model>());
}

}
