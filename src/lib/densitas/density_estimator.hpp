#pragma once
#include "type_check.hpp"
#include "math.hpp"
#include "model_adapter.hpp"
#include "matrix_adapter.hpp"
#include "vector_adapter.hpp"
#include "manipulation.hpp"
#include "thread_pool.hpp"
#include <vector>
#include <memory>


namespace densitas {

/**
 * The density estimator, once trained, predicts any given quantiles
 * of the unknown probability distribution around the true value.
 *
 * Use the density estimator to solve regression problems when you need to
 * predict certain quantiles instead of just a single expectation value.
 *
 * The performance of the density estimator will depend greatly on the
 * model given by the user.
 *
 * ModelType: Operations on the model type are defined in model_adapter.hpp.
 *            Specialize the functions in there if your model does things differently
 * MatrixType: Operations on the matrix type are defined in matrix_adapter.hpp.
 *             Specialize the functions in there if your matrix does things differently
 * VectorType: Operations on the vector type are defined in vector_adapter.hpp.
 *             Specialize the functions in there if your vector does things differently
 * ElementType: Must be a floating point type, e.g., float or double
 */
template<typename ModelType, typename MatrixType, typename VectorType, typename ElementType=double>
class density_estimator {
public:

    /**
     * Constructor
     */
    density_estimator()
    : models_{}, trained_centers_{}, predicted_quantiles_{}, accuracy_predicted_quantiles_{}
    {
        init();
    }

    /**
     * Constructor
     * @param model A binary classifier. Must implement copy semantics
     * @param n_models The number of models to use
     */
    density_estimator(const ModelType& model, std::size_t n_models)
    : models_{}, trained_centers_{}, predicted_quantiles_{}, accuracy_predicted_quantiles_{}
    {
        init();
        set_models(model, n_models);
    }

    /**
     * Returns a clone of this density estimator
     */
    virtual std::unique_ptr<density_estimator> clone()
    {
        auto estimator = std::unique_ptr<density_estimator>(new density_estimator);
        estimator->models_ = models_;
        estimator->trained_centers_ = trained_centers_;
        estimator->predicted_quantiles_ = predicted_quantiles_;
        estimator->accuracy_predicted_quantiles_ = accuracy_predicted_quantiles_;
        return std::move(estimator);
    }

    /**
     * Set the internal models using a reference model object
     * @param model A binary classifier. Must implement copy semantics
     * @param n_models The number of models to use
     */
    void set_models(const ModelType& model, std::size_t n_models)
    {
        check_n_models(n_models);
        models_.clear();
        models_.assign(n_models, model);
    }

    /**
     * Sets the predicted quantiles which must be values between
     *  zero and one. Default: {0.05, 0.5, 0.95}
     * @param quantiles The predicted quantiles
     */
    void predicted_quantiles(const VectorType& quantiles)
    {
        predicted_quantiles_ = quantiles;
    }

    /**
     * Sets the computation accuracy of the predicted quantiles. Must be
     *  a value between zero and one. The closer to zero the better
     *  the accuracy but the higher the computation demand. Default: 1e-2
     * @param accuracy The predicted quantile accuracy
     */
    void accuracy_predicted_quantiles(ElementType accuracy)
    {
        accuracy_predicted_quantiles_ = accuracy;
    }

    /**
     * Trains the density estimator
     * @param X A matrix of shape (n_events, n_features)
     * @param y A vector of shape (n_events)
     * @param threads Max number of threads to launch, single-threaded if <= 1
     *  checks of whether the thread pool can accept another thread
     */
    void train(const MatrixType& X, const VectorType& y, int threads=1)
    {
        check_n_models(models_.size());
        const auto quantiles = densitas::math::linspace<VectorType, ElementType>(0, 1, models_.size() + 1);
        const auto trained_quantiles = densitas::math::quantiles<ElementType>(y, quantiles);
        trained_centers_ = densitas::math::centers<ElementType>(y, trained_quantiles);
        const auto params = train_params{y, trained_quantiles};
        if (threads > 1) {
            densitas::core::thread_pool pool{threads};
            for (std::size_t i=0; i<models_.size(); ++i) {
                pool.wait_for_slot();
                on_train_status(models_[i], i, X, params);
                pool.launch_new(density_estimator::train_model, std::ref(models_[i]), i, X, std::ref(params));
            }
        } else {
            for (std::size_t i=0; i<models_.size(); ++i) {
                on_train_status(models_[i], i, X, params);
                density_estimator::train_model(models_[i], i, X, params);
            }
        }
    }

    /**
     * Predicts events using this trained density estimator
     * @param X A matrix of shape (n_events, n_features)
     * @param threads Max number of threads to launch, single-threaded if <= 1
     * @return A matrix of shape (n_events, n_predicted_quantiles)
     */
    MatrixType predict(const MatrixType& X, int threads=1)
    {
        check_n_models(models_.size());
        const auto n_rows = densitas::matrix_adapter::n_rows(X);
        const auto n_quantiles = densitas::vector_adapter::n_elements(predicted_quantiles_);
        auto prediction = densitas::matrix_adapter::construct_uninitialized<MatrixType>(n_rows, n_quantiles);
        const auto params = predict_params{X, trained_centers_, predicted_quantiles_, accuracy_predicted_quantiles_};
        if (threads > 1) {
            densitas::core::thread_pool pool{threads};
            for (std::size_t i=0; i<n_rows; ++i) {
                pool.wait_for_slot();
                on_predict_status(prediction, models_, i, params);
                pool.launch_new(density_estimator::predict_event, std::ref(prediction), std::ref(models_), i, std::ref(params));
            }
        } else {
            for (std::size_t i=0; i<n_rows; ++i) {
                on_predict_status(prediction, models_, i, params);
                density_estimator::predict_event(prediction, models_, i, params);
            }
        }
        return prediction;
    }

    density_estimator(const density_estimator&) = delete;
    density_estimator& operator=(const density_estimator&) = delete;
    density_estimator(density_estimator&&) = delete;
    density_estimator& operator=(density_estimator&&) = delete;

    virtual ~density_estimator() {}

protected:

    std::vector<ModelType> models_;
    VectorType trained_centers_;
    VectorType predicted_quantiles_;
    ElementType accuracy_predicted_quantiles_;

    struct train_params {
        const VectorType& y;
        const VectorType& trained_quantiles;
    };

    struct predict_params {
        const MatrixType& features;
        const VectorType& centers;
        const VectorType& quantiles;
        const double accuracy;
    };

    virtual void on_train_status(const ModelType&, std::size_t, const MatrixType&, const train_params&) {}

    virtual void on_predict_status(const MatrixType&, const std::vector<ModelType>&, std::size_t, const predict_params&) {}

    virtual void init()
    {
        densitas::core::check_element_type<ElementType>();
        accuracy_predicted_quantiles_ = 1e-2;
        trained_centers_ = densitas::vector_adapter::construct_uninitialized<VectorType>(0);
        predicted_quantiles_ = densitas::vector_adapter::construct_uninitialized<VectorType>(3);
        densitas::vector_adapter::set_element<ElementType>(predicted_quantiles_, 0, 0.05);
        densitas::vector_adapter::set_element<ElementType>(predicted_quantiles_, 1, 0.5);
        densitas::vector_adapter::set_element<ElementType>(predicted_quantiles_, 2, 0.95);
    }

    virtual void check_n_models(std::size_t n_models) const
    {
        if (!(n_models > 1))
            throw densitas::densitas_error("number of models must be larger than one");
    }

    static void train_model(ModelType& model, std::size_t model_index, MatrixType features, const train_params& params)
    {
        const auto lower = densitas::vector_adapter::get_element<ElementType>(params.trained_quantiles, model_index);
        const auto upper = densitas::vector_adapter::get_element<ElementType>(params.trained_quantiles, model_index + 1);
        auto target = densitas::math::make_classification_target<ModelType>(params.y, lower, upper);
        densitas::model_adapter::train(model, features, target);
    }

    static void predict_event(MatrixType& prediction, std::vector<ModelType>& models, std::size_t event_index, const predict_params& params)
    {
        auto weights = densitas::vector_adapter::construct_uninitialized<VectorType>(models.size());
        for (std::size_t j=0; j<models.size(); ++j) {
            const auto prob_value = densitas::core::predict_proba_for_row<ElementType, VectorType>(models[j], params.features, event_index);
            densitas::vector_adapter::set_element<ElementType>(weights, j, prob_value);
        }
        const auto quants = densitas::math::quantiles_weighted<ElementType>(params.centers, weights, params.quantiles, params.accuracy);
        densitas::core::assign_vector_to_row<ElementType>(prediction, event_index, quants);
    }

};


} // densitas
