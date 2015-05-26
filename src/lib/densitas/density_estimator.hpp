#pragma once
#include "type_check.hpp"
#include "math.hpp"
#include "model_adapter.hpp"
#include "matrix_adapter.hpp"
#include "vector_adapter.hpp"
#include "manipulation.hpp"
#include <vector>
#include <future>


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
 * ElementType: Must be an arithmetic type, e.g., float or double
 */
template<typename ModelType, typename MatrixType, typename VectorType, typename ElementType=double>
class density_estimator {
public:

    /**
     * Constructor
     * @param model A binary classifier
     * @param n_models The number of models to use
     */
    density_estimator(const ModelType& model, size_t n_models)
        : models_(n_models),
          trained_quantiles_(densitas::vector_adapter::construct_uninitialized<VectorType>(0)),
          predicted_quantiles_(densitas::vector_adapter::construct_uninitialized<VectorType>(1)),
          accuracy_predicted_quantiles_(1e-2)
    {
        densitas::core::check_element_type<ElementType>();
        if (!(n_models > 1))
            throw densitas::densitas_error("n_models must be larger than one");
        for (auto& m : models_)
            m = model;
        densitas::vector_adapter::set_element<ElementType>(predicted_quantiles_, 0, 0.5);
    }

    /**
     * Sets the predicted quantiles which must be values between
     *  zero and one. Default: {0.5}
     * @param quantiles The predicted probabilties
     */
    void predicted_quantiles(const VectorType& quantiles)
    {
        predicted_quantiles_ = quantiles;
    }

    /**
     * Sets the computation accuracy of the predicted quantiles. Must be
     *  a value between zero and one. The closer to zero the better
     *  the accuracy but the higher the computation demand. Default: 1e-2
     */
    void accuracy_predicted_quantiles(ElementType accuracy)
    {
        accuracy_predicted_quantiles_ = accuracy;
    }

    /**
     * Trains the density estimator
     * @param X A matrix of shape (n_events, n_features)
     * @param y A vector of shape (n_events)
     * @param async Whether to train each model asynchronously
     */
    void train(const MatrixType& X, const VectorType& y, bool async=false)
    {
        const auto quantiles = densitas::math::linspace<VectorType, ElementType>(0, 1, models_.size() + 1);
        trained_quantiles_ = densitas::math::quantiles<ElementType>(y, quantiles);
        if (async) {
			std::vector<std::future<void>> futures(models_.size());
			for (size_t i=0; i<models_.size(); ++i) {
				futures[i] = std::async(density_estimator::train_model, std::ref(models_[i]), i, X, std::ref(y), std::ref(trained_quantiles_));
			}
			for (auto& future : futures)
				future.get();
        } else {
			for (size_t i=0; i<models_.size(); ++i) {
				density_estimator::train_model(models_[i], i, X, y, trained_quantiles_);
			}
        }
    }

    /**
     * Predicts events using this trained density estimator
     * @param X A matrix of shape (n_events, n_features)
     * @return A matrix of shape (n_events, n_predicted_quantiles)
     */
    MatrixType predict(const MatrixType& X)
    {
        const auto centers = densitas::math::centers<ElementType>(trained_quantiles_);
        const auto n_rows = densitas::matrix_adapter::n_rows(X);
        const auto n_quantiles = densitas::vector_adapter::n_elements(predicted_quantiles_);
        auto prediction = densitas::matrix_adapter::construct_uninitialized<MatrixType>(n_rows, n_quantiles);
        for (size_t i=0; i<n_rows; ++i) {
            auto weights = densitas::vector_adapter::construct_uninitialized<VectorType>(models_.size());
            for (size_t j=0; j<models_.size(); ++j) {
                const auto prob_value = densitas::core::predict_proba_for_row<ElementType, VectorType>(models_[j], X, i);
                densitas::vector_adapter::set_element<ElementType>(weights, j, prob_value);
            }
            const auto quantiles = densitas::math::quantiles_weighted<ElementType>(centers, weights, predicted_quantiles_, accuracy_predicted_quantiles_);
            densitas::core::assign_vector_to_row<ElementType>(prediction, i, quantiles);
        }
        return prediction;
    }

    density_estimator(const density_estimator& other)
		: models_(other.models_),
		  trained_quantiles_(other.trained_quantiles_),
		  predicted_quantiles_(other.predicted_quantiles_),
		  accuracy_predicted_quantiles_(other.accuracy_predicted_quantiles_)
    {
        densitas::core::check_element_type<ElementType>();
    }

    density_estimator& operator=(const density_estimator& other)
    {
    	if (this != &other) {
			models_ = other.models_;
			trained_quantiles_ = other.trained_quantiles_;
			predicted_quantiles_ = other.predicted_quantiles_;
			accuracy_predicted_quantiles_ = other.accuracy_predicted_quantiles_;
    	}
    	return *this;
    }

    density_estimator(density_estimator&& other)
		: models_(std::move(other.models_)),
		  trained_quantiles_(std::move(other.trained_quantiles_)),
		  predicted_quantiles_(std::move(other.predicted_quantiles_)),
		  accuracy_predicted_quantiles_(std::move(other.accuracy_predicted_quantiles_))
    {
        densitas::core::check_element_type<ElementType>();
    }

    density_estimator& operator=(density_estimator&& other)
    {
    	if (this != &other) {
			models_ = std::move(other.models_);
			trained_quantiles_ = std::move(other.trained_quantiles_);
			predicted_quantiles_ = std::move(other.predicted_quantiles_);
			accuracy_predicted_quantiles_ = std::move(other.accuracy_predicted_quantiles_);
    	}
    	return *this;
    }

    virtual ~density_estimator() = default;

protected:

    ModelType ref_model_;
    std::vector<ModelType> models_;
    VectorType trained_quantiles_;
    VectorType predicted_quantiles_;
    ElementType accuracy_predicted_quantiles_;

    static void train_model(ModelType& model, size_t model_index, MatrixType features, const VectorType& y, const VectorType& trained_quantiles)
    {
        const auto lower = densitas::vector_adapter::get_element<ElementType>(trained_quantiles, model_index);
        const auto upper = densitas::vector_adapter::get_element<ElementType>(trained_quantiles, model_index + 1);
		auto target = densitas::math::make_classification_target<ModelType>(y, lower, upper);
		densitas::model_adapter::train(model, features, target);
    }

};


} // densitas
