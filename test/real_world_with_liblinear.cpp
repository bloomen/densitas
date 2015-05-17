#include "utils.hpp"
#include <linear.h>
#include <memory>


#ifndef DATADIR
#define DATADIR "."
#endif


COLLECTION(real_world_with_liblinear) {


// a wrapper to avoid name conflicts with classifier::train()
struct model* liblinear_train(const struct problem *prob, const struct parameter *param)
{
    return train(prob, param);
}


struct model_deleter {
    void operator()(model* model)
    {
        if (model) {
            free_and_destroy_model(&model);
        }
    }
};


struct problem_deleter {
    explicit
    problem_deleter(bool is_training)
        : is_training_(is_training)
    {}
    void operator()(problem* problem)
    {
        if (problem) {
            if (problem->y) {
                delete[] problem->y;
            }
            if (problem->x) {
                if (!is_training_)
                    delete[] problem->x[0];
                delete[] problem->x;
            }
            delete problem;
        }
    }
private:
    bool is_training_;
};


struct classifier {

    classifier()
        : model_(), params_()
    {
        init_params();
    }

    classifier(const classifier&) = delete;
    classifier(classifier&&) = delete;

    // doesn't have to actually copy to work with density_estimator,
    // it just needs to re-initialize
    classifier& operator=(const classifier& other)
    {
        if (&other != this) {
            model_.reset();
            free_param_weights();
            init_params();
        }
        return *this;
    }

    classifier& operator=(classifier&&) = delete;

    ~classifier()
    {
        free_param_weights();
    }

    void train(matrix_t& X, vector_t& y)
    {
        auto problem = make_problem(X, y, true);
        model_.reset();
        model_ = std::unique_ptr<model, model_deleter>(liblinear_train(problem.get(), &params_), model_deleter());
    }

    vector_t predict_proba(matrix_t& X)
    {
         auto problem = make_problem(X, vector_t{}, false);
         auto probas = vector_t(problem->l);
         double estimates[2];
         for (size_t i=0; i<probas.n_elem; ++i) {
             const auto value = predict_probability(model_.get(), problem->x[i], estimates);
             probas(i) = value==densitas::model_adaptor::yes<classifier>() ? std::max(estimates[0], estimates[1]) : std::min(estimates[0], estimates[1]);
         }
         return probas;
    }

private:
    std::unique_ptr<model, model_deleter> model_;
    parameter params_;

    void init_params()
    {
        set_print_string_function([](const char*) {});
        params_.solver_type = L2R_LR;
        params_.eps = 1e-3;
        params_.C = 1;
        params_.nr_weight = 0;
        params_.weight_label = nullptr;
        params_.weight = nullptr;
    }

    void free_param_weights()
    {
        if (params_.weight) {
           delete[] params_.weight;
        }
        if (params_.weight_label) {
           delete[] params_.weight_label;
        }
    }

    std::unique_ptr<problem, problem_deleter> make_problem(const matrix_t& X, const vector_t& y, bool is_training)
    {
        const auto n_rows = X.n_rows;
        const auto n_cols = X.n_cols;
        auto prob = std::unique_ptr<problem, problem_deleter>(new problem, problem_deleter(is_training));
        prob->l = static_cast<int>(n_rows);
        if (is_training) {
            prob->y = new int[n_rows];
            std::copy(y.begin(), y.end(), prob->y);
        } else {
            prob->y = nullptr;
        }
        prob->x = new feature_node*[n_rows];
        auto xspace = new feature_node[(n_cols+1)*n_rows];
        for (int i=0; i<static_cast<int>(n_rows); ++i) {
            for (int j=0; j<static_cast<int>(n_cols); ++j) {
                xspace[(n_cols+1)*i+j].index = j+1;
                xspace[(n_cols+1)*i+j].value = X(i, j);
            }
            xspace[(n_cols+1)*i+n_cols].index = -1;
            prob->x[i] = &xspace[(n_cols+1)*i];
        }
        return std::move(prob);
    }

};


typedef densitas::density_estimator<double, matrix_t, vector_t, classifier> estimator_t;

std::string dataset()
{
    return std::string(DATADIR) + "/diabetes.txt";
}

matrix_t get_X()
{
    auto X = matrix_t();
    X.load(dataset(), arma::csv_ascii);
    X.shed_col(10);
    assert_equal(10, X.n_cols);
    assert_equal(442, X.n_rows);
    return std::move(X);
}

vector_t get_y()
{
    auto X = matrix_t();
    X.load(dataset(), arma::csv_ascii);
    assert_equal(442, X.n_rows);
    return vector_t{X.col(10)};
}

TEST(test_density_estimator) {
    const auto X = get_X();
    const auto y = get_y();
    const classifier model;

    const size_t n_models = 9;
    estimator_t estimator(model, n_models);
    estimator.train(X, y);

    const vector_t prediction = estimator.predict(X).col(0);
    assert_equal(y.n_elem, prediction.n_elem, SPOT);

    double error = 0;
    for (size_t i=0; i<y.n_elem; ++i) {
        error += std::abs(y(i) - prediction(i));
    }
    error /= y.n_elem;
    assert_lesser(error, 45., SPOT);
}

TEST(test_density_estimator_predict_more_quantiles) {
    const auto X = get_X();
    const auto y = get_y();
    const classifier model;

    const size_t n_models = 9;
    estimator_t estimator(model, n_models);
    estimator.train(X, y);

    const auto quantiles = vector_t{0.05, 0.5, 0.95};
    estimator.predicted_quantiles(quantiles);
    const matrix_t prediction = estimator.predict(X);
    assert_equal(y.n_elem, prediction.n_rows, SPOT);
    assert_equal(quantiles.n_elem, prediction.n_cols, SPOT);
}

}
