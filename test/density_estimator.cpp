#include "utils.hpp"


COLLECTION(density_estimator) {


struct estimator_t : densitas::density_estimator<estimator_t, mock_model, matrix_t, vector_t> {

    estimator_t()
    : density_estimator_type{}
    {}

    estimator_t(const mock_model& model, std::size_t n_models)
    : density_estimator_type{model, n_models}
    {}

    const std::vector<std::unique_ptr<mock_model>>& get_models() const
    {
        return models_;
    }

    vector_t get_trained_centers() const
    {
        return trained_centers_;
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
    X.row(0) = mkrow({1, 2, 3});
    X.row(1) = mkrow({10, 20, 30});
    X.row(2) = mkrow({100, 200, 300});
    X.row(3) = mkrow({1000, 2000, 3000});
    X.row(4) = mkrow({10000, 20000, 30000});
    return X;
}

std::unique_ptr<estimator_t> train_estimator(bool async=false)
{
    auto model = mock_model();
    model.prediction = mkcol({0.5});
    auto estimator = std::unique_ptr<estimator_t>(new estimator_t(model, 2));
    const auto X = get_X();
    const auto y = mkcol({5, 6, 7, 8, 9});
    const auto threads = async ? 3 : 1;
    estimator->train(X, y, threads);
    return estimator;
}

void make_test_train(bool async)
{
    auto estimator = train_estimator(async);
    const auto centers = estimator->get_trained_centers();
    const auto exp_centers = mkcol({5.5, 8});
    assert_equal_containers(exp_centers, centers, SPOT);
    const auto& models = estimator->get_models();
    assert_equal(2u, models.size());
    const auto X = get_X();
    assert_equal_containers(X, models[0]->train_X, SPOT);
    assert_equal_containers(X, models[1]->train_X, SPOT);
    const auto target1 = mkcol({dyes, dyes, dno, dno, dno});
    const auto target2 = mkcol({dno, dno, dyes, dyes, dyes});
    assert_equal_containers(target1, models[0]->train_y, SPOT);
    assert_equal_containers(target2, models[1]->train_y, SPOT);
}

TEST(test_train) {
    make_test_train(false);
}

TEST(test_train_async) {
    make_test_train(true);
}

void make_test_predict(bool async)
{
    auto estimator = train_estimator();
    estimator->predicted_quantiles(mkcol({0.5, 0.9}));
    const auto X = get_X();
    const auto threads = async ? 3 : 1;
    const auto y_resp = estimator->predict(X, threads);
    assert_equal(X.n_rows, y_resp.n_rows, SPOT);
    assert_equal(2u, y_resp.n_cols, SPOT);
    auto y_exp = matrix_t(X.n_rows, 2u);
    for (std::size_t i=0; i<X.n_rows; ++i) {
        y_exp.row(i) = mkrow({5.5, 7.5});
    }
    assert_equal_containers(y_exp, y_resp, SPOT);
}

TEST(test_predict) {
    make_test_predict(false);
}

TEST(test_predict_async) {
    make_test_predict(true);
}

TEST(test_number_of_models_too_small) {
    auto model = mock_model();
    assert_throw<densitas::densitas_error>([&]() { estimator_t(model, 1); }, SPOT);
}

TEST(test_predicted_quantiles_setter) {
    auto model = mock_model();
    estimator_t estimator(model, 2);
    const auto quantiles = mkcol({0.5, 0.9});
    estimator.predicted_quantiles(quantiles);
    assert_equal_containers(quantiles, estimator.get_predicted_quantiles(), SPOT);
}

TEST(test_accuracy_predicted_quantiles_setter) {
    auto model = mock_model();
    estimator_t estimator(model, 2);
    const auto accuracy = 1e-3;
    estimator.accuracy_predicted_quantiles(accuracy);
    assert_equal(accuracy, estimator.get_accuracy_predicted_quantiles(), SPOT);
}

TEST(test_clone) {
    auto model = mock_model();
    estimator_t estimator;
    estimator.set_models(model, 2);
    auto cloned = estimator.clone();
    assert_true(cloned, SPOT);
}

TEST(test_typedefs) {
    static_assert(std::is_same<densitas::density_estimator<estimator_t, mock_model, matrix_t, vector_t>, typename estimator_t::density_estimator_type>::value, "");
    static_assert(std::is_same<mock_model, typename estimator_t::model_type>::value, "");
    static_assert(std::is_same<matrix_t, typename estimator_t::matrix_type>::value, "");
    static_assert(std::is_same<vector_t, typename estimator_t::vector_type>::value, "");
    static_assert(std::is_same<double, typename estimator_t::element_type>::value, "");
}

}
