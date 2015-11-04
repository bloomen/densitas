#include "utils.hpp"


COLLECTION(model_adapter) {

TEST(test_clone) {
    auto model = mock_model();
    auto X = matrix_t(1, 2);
    X.row(0) = mkrow({-1, -2});
    auto y = mkcol({3, 4.5});
    densitas::model_adapter::train(model, X, y);
    auto cloned = densitas::model_adapter::clone(model);
    assert_equal_containers(X, cloned->train_X, SPOT);
    assert_equal_containers(y, cloned->train_y, SPOT);
}

TEST(test_train) {
    auto model = mock_model();
    auto X = matrix_t(1, 2);
    X.row(0) = mkrow({-1, -2});
    auto y = mkcol({3, 4.5});
    densitas::model_adapter::train(model, X, y);
    assert_equal_containers(X, model.train_X, SPOT);
    assert_equal_containers(y, model.train_y, SPOT);
}

TEST(test_predict_proba) {
    auto model = mock_model();
    model.prediction = mkcol({3, 4});
    auto X = matrix_t(1, 2);
    X.row(0) = mkrow({-1, -2});
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
