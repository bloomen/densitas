#include "utils.hpp"


COLLECTION(density_estimator) {

typedef densitas::density_estimator<double, matrix_t, vector_t, mock_model> estimator_t;

struct extended_estimator : estimator_t {

    extended_estimator(const mock_model& model, size_t n_models)
        : estimator_t(model, n_models)
    {}

    std::vector<mock_model> get_models() const
    {
        return models_;
    }

    vector_t get_trained_quantiles() const
    {
        return trained_quantiles_;
    }

    vector_t get_predicted_quantiles() const
    {
        return predicted_quantiles_;
    }

    double get_accuracy_predicted_quantiles() const
    {
        return accuracy_predicted_quantiles_;
    }

};

matrix_t get_X()
{
    auto X = matrix_t(5, 3);
    X.row(0) = vector_t{1, 2, 3}.t();
    X.row(1) = vector_t{10, 20, 30}.t();
    X.row(2) = vector_t{100, 200, 300}.t();
    X.row(3) = vector_t{1000, 2000, 3000}.t();
    X.row(4) = vector_t{10000, 20000, 30000}.t();
    return X;
}

extended_estimator train_estimator()
{
    auto model = mock_model();
    model.prediction = vector_t{0.5};
    auto estimator = extended_estimator(model, 2);
    const auto X = get_X();
    const auto y = vector_t{5, 6, 7, 8, 9};
    estimator.train(X, y);
    return estimator;
}

TEST(test_train) {
    const auto estimator = train_estimator();
    const auto quantiles = estimator.get_trained_quantiles();
    const auto exp_quantiles = vector_t{5, 6.5, 9};
    assert_equal_containers(exp_quantiles, quantiles, SPOT);
    const auto models = estimator.get_models();
    assert_equal(2u, models.size());
    const auto X = get_X();
    assert_equal_containers(X, models[0].train_X, SPOT);
    assert_equal_containers(X, models[1].train_X, SPOT);
    const auto target1 = vector_t{yes, yes, no, no, no};
    const auto target2 = vector_t{no, no, yes, yes, yes};
    assert_equal_containers(target1, models[0].train_y, SPOT);
    assert_equal_containers(target2, models[1].train_y, SPOT);
}

TEST(test_predict) {
    auto estimator = train_estimator();
    estimator.predicted_quantiles(vector_t{0.5, 0.9});
    const auto X = get_X();
    const auto y_resp = estimator.predict(X);
    const auto models = estimator.get_models();
    assert_equal(X.n_rows, models[0].predict_X.size(), SPOT);
    assert_equal(X.n_rows, models[1].predict_X.size(), SPOT);
    for (size_t i=0; i<X.n_rows; ++i) {
        assert_equal_containers(vector_t(X.row(i).t()), vector_t(models[0].predict_X[i].row(0).t()), SPOT);
        assert_equal_containers(vector_t(X.row(i).t()), vector_t(models[1].predict_X[i].row(0).t()), SPOT);
    }
    assert_equal(X.n_rows, y_resp.n_rows, SPOT);
    assert_equal(2u, y_resp.n_cols, SPOT);
    auto y_exp = matrix_t(X.n_rows, 2u);
    for (size_t i=0; i<X.n_rows; ++i) {
        y_exp.row(i) = vector_t{5.75, 7.35}.t();
    }
    assert_equal_containers(y_exp, y_resp, SPOT);
}

TEST(test_number_of_models_too_small) {
    auto model = mock_model();
    assert_throw<densitas::densitas_error>([&]() { estimator_t(model, 1); }, SPOT);
}

TEST(test_predicted_quantiles_setter) {
    auto model = mock_model();
    auto estimator = extended_estimator(model, 2);
    const auto quantiles = vector_t{0.5, 0.9};
    estimator.predicted_quantiles(quantiles);
    assert_equal_containers(quantiles, estimator.get_predicted_quantiles(), SPOT);
}

TEST(test_accuracy_predicted_quantiles_setter) {
    auto model = mock_model();
    auto estimator = extended_estimator(model, 2);
    const auto accuracy = 1e-3;
    estimator.accuracy_predicted_quantiles(accuracy);
    assert_equal(accuracy, estimator.get_accuracy_predicted_quantiles(), SPOT);
}

}
