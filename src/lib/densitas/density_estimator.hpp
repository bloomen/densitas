#pragma once
#include "type_check.hpp"
#include "math.hpp"
#include "model_adapter.hpp"
#include "matrix_adapter.hpp"
#include "vector_adapter.hpp"
#include "manipulation.hpp"
#include "task_manager.hpp"
#include <vector>


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
 * SubType: The sub-class that is inheriting from density_estimator (CRTP)
 * ModelType: Operations on the model type are defined in model_adapter.hpp.
 *            Specialize the functions in there if your model does things differently
 * MatrixType: Operations on the matrix type are defined in matrix_adapter.hpp.
 *             Specialize the functions in there if your matrix does things differently
 * VectorType: Operations on the vector type are defined in vector_adapter.hpp.
 *             Specialize the functions in there if your vector does things differently
 * ElementType: Must be a floating point type, e.g., float or double
 */
template<typename SubType, typename ModelType, typename MatrixType, typename VectorType, typename ElementType=double>
class density_estimator {
public:

    typedef density_estimator density_estimator_type;
    typedef ModelType model_type;
    typedef MatrixType matrix_type;
    typedef VectorType vector_type;
    typedef ElementType element_type;

    /**
     * Returns a clone of this density estimator
     */
    virtual std::unique_ptr<density_estimator> clone() const
    {
        auto estimator = std::unique_ptr<SubType>{new SubType};
        for (const auto& model : models_) {
            estimator->models_.emplace_back(densitas::model_adapter::clone(*model));
        }
        estimator->trained_centers_ = trained_centers_;
        estimator->predicted_quantiles_ = predicted_quantiles_;
        estimator->accuracy_predicted_quantiles_ = accuracy_predicted_quantiles_;
        return std::move(estimator);
    }

    /**
     * Set the internal models using a reference model object
     * @param model A binary classifier. Must be clonable
     * @param n_models The number of models to use
     */
    void set_models(const model_type& model, std::size_t n_models)
    {
        check_n_models(n_models);
        models_.clear();
        for (std::size_t i=0; i<n_models; ++i) {
            models_.emplace_back(densitas::model_adapter::clone(model));
        }
    }

    /**
     * Sets the predicted quantiles which must be values between
     *  zero and one. Default: {0.05, 0.5, 0.95}
     * @param quantiles The predicted quantiles
     */
    void predicted_quantiles(const vector_type& quantiles)
    {
        predicted_quantiles_ = quantiles;
    }

    /**
     * Sets the computation accuracy of the predicted quantiles. Must be
     *  a value between zero and one. The closer to zero the better
     *  the accuracy but the higher the computation demand. Default: 1e-2
     * @param accuracy The predicted quantile accuracy
     */
    void accuracy_predicted_quantiles(element_type accuracy)
    {
        accuracy_predicted_quantiles_ = accuracy;
    }

    /**
     * Trains the density estimator
     * @param X A matrix of shape (n_events, n_features)
     * @param y A vector of shape (n_events)
     * @param threads Max number of threads to launch, single-threaded if <= 1
     */
    void train(const matrix_type& X, const vector_type& y, int threads=1)
    {
        check_n_models(models_.size());
        const auto quantiles = densitas::math::linspace<vector_type, element_type>(0, 1, models_.size() + 1);
        const auto trained_quantiles = densitas::math::quantiles<element_type>(y, quantiles);
        trained_centers_ = densitas::math::centers<element_type>(y, trained_quantiles);
        const auto params = train_params{y, trained_quantiles};
        if (threads > 1) {
            densitas::core::task_manager manager(threads);
            for (std::size_t i=0; i<models_.size(); ++i) {
                manager.wait_for_slot();
                on_train_status(*models_[i], i, X, params);
                manager.launch_new(density_estimator::train_model, std::ref(*models_[i]), i, X, std::ref(params));
            }
        } else {
            for (std::size_t i=0; i<models_.size(); ++i) {
                on_train_status(*models_[i], i, X, params);
                density_estimator::train_model(*models_[i], i, X, params);
            }
        }
    }

    /**
     * Predicts events using this trained density estimator
     * @param X A matrix of shape (n_events, n_features)
     * @param threads Max number of threads to launch, single-threaded if <= 1
     * @return A matrix of shape (n_events, n_predicted_quantiles)
     */
    matrix_type predict(const matrix_type& X, int threads=1) const
    {
        check_n_models(models_.size());
        const auto n_rows = densitas::matrix_adapter::n_rows(X);
        const auto n_quantiles = densitas::vector_adapter::n_elements(predicted_quantiles_);
        auto prediction = densitas::matrix_adapter::construct_uninitialized<matrix_type>(n_rows, n_quantiles);
        const auto params = predict_params{X, trained_centers_, predicted_quantiles_, accuracy_predicted_quantiles_};
        if (threads > 1) {
            densitas::core::task_manager manager(threads);
            for (std::size_t i=0; i<n_rows; ++i) {
                manager.wait_for_slot();
                on_predict_status(prediction, models_, i, params);
                manager.launch_new(density_estimator::predict_event, std::ref(prediction), std::ref(models_), i, std::ref(params));
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
     * @param model A binary classifier. Must be clonable
     * @param n_models The number of models to use
     */
    density_estimator(const model_type& model, std::size_t n_models)
    : models_{}, trained_centers_{}, predicted_quantiles_{}, accuracy_predicted_quantiles_{}
    {
        init();
        set_models(model, n_models);
    }

    std::vector<std::unique_ptr<model_type>> models_;
    vector_type trained_centers_;
    vector_type predicted_quantiles_;
    element_type accuracy_predicted_quantiles_;

    struct train_params {
        const vector_type& y;
        const vector_type& trained_quantiles;
    };

    struct predict_params {
        const matrix_type& features;
        const vector_type& centers;
        const vector_type& quantiles;
        const double accuracy;
    };

    virtual void on_train_status(const model_type&, std::size_t, const matrix_type&, const train_params&) const {}

    virtual void on_predict_status(const matrix_type&, const std::vector<std::unique_ptr<model_type>>&, std::size_t, const predict_params&) const {}

    virtual void init()
    {
        static_assert(std::is_base_of<density_estimator_type, SubType>::value, "SubType is not inheriting from density_estimator");
        densitas::core::check_element_type<element_type>();
        accuracy_predicted_quantiles_ = 1e-2;
        trained_centers_ = densitas::vector_adapter::construct_uninitialized<vector_type>(0);
        predicted_quantiles_ = densitas::vector_adapter::construct_uninitialized<vector_type>(3);
        densitas::vector_adapter::set_element<element_type>(predicted_quantiles_, 0, 0.05);
        densitas::vector_adapter::set_element<element_type>(predicted_quantiles_, 1, 0.5);
        densitas::vector_adapter::set_element<element_type>(predicted_quantiles_, 2, 0.95);
    }

    virtual void check_n_models(std::size_t n_models) const
    {
        if (!(n_models > 1))
            throw densitas::densitas_error("number of models must be larger than one");
    }

    static void train_model(model_type& model, std::size_t model_index, matrix_type features, const train_params& params)
    {
        const auto lower = densitas::vector_adapter::get_element<element_type>(params.trained_quantiles, model_index);
        const auto upper = densitas::vector_adapter::get_element<element_type>(params.trained_quantiles, model_index + 1);
        auto target = densitas::math::make_classification_target<model_type>(params.y, lower, upper);
        densitas::model_adapter::train(model, features, target);
    }

    static void predict_event(matrix_type& prediction, const std::vector<std::unique_ptr<model_type>>& models, std::size_t event_index, const predict_params& params)
    {
        auto weights = densitas::vector_adapter::construct_uninitialized<vector_type>(models.size());
        for (std::size_t j=0; j<models.size(); ++j) {
            const auto prob_value = densitas::core::predict_proba_for_row<element_type, vector_type>(*models[j], params.features, event_index);
            densitas::vector_adapter::set_element<element_type>(weights, j, prob_value);
        }
        const auto quants = densitas::math::quantiles_weighted<element_type>(params.centers, weights, params.quantiles, params.accuracy);
        densitas::core::assign_vector_to_row<element_type>(prediction, event_index, quants);
    }

};


} // densitas
