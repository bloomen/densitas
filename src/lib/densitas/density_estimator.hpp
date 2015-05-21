#pragma once
#include <vector>
#include "type_check.hpp"
#include "math.hpp"
#include "model_adaptor.hpp"
#include "matrix_adaptor.hpp"
#include "vector_adaptor.hpp"
#include "manipulation.hpp"


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
 * ModelType: Operations on the model type are defined in model_adaptor.hpp.
 *            Specialize the functions in there if your model does things differently
 * MatrixType: Operations on the matrix type are defined in matrix_adaptor.hpp.
 *             Specialize the functions in there if your matrix does things differently
 * VectorType: Operations on the vector type are defined in vector_adaptor.hpp.
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
          trained_quantiles_(densitas::vector_adaptor::construct_uninitialized<VectorType>(0)),
          predicted_quantiles_(densitas::vector_adaptor::construct_uninitialized<VectorType>(1)),
          accuracy_predicted_quantiles_(1e-2)
    {
        densitas::core::check_element_type<ElementType>();
        if (!(n_models > 1))
            throw densitas::densitas_error("n_models must be larger than one");
        for (auto& m : models_)
            m = model;
        densitas::vector_adaptor::set_element<ElementType>(predicted_quantiles_, 0, 0.5);
    }

    virtual ~density_estimator() = default;

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
     */
    void train(const MatrixType& X, const VectorType& y)
    {
        const auto quantiles = densitas::math::linspace<VectorType, ElementType>(0, 1, models_.size() + 1);
        trained_quantiles_ = densitas::math::quantiles<ElementType>(y, quantiles);
        for (size_t i=0; i<models_.size(); ++i) {
            const auto lower = densitas::vector_adaptor::get_element<ElementType>(trained_quantiles_, i);
            const auto upper = densitas::vector_adaptor::get_element<ElementType>(trained_quantiles_, i+1);
            auto features = MatrixType(X);
            auto target = densitas::math::make_classification_target<ModelType>(y, lower, upper);
            densitas::model_adaptor::train(models_[i], features, target);
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
        const auto n_rows = densitas::matrix_adaptor::n_rows(X);
        const auto n_quantiles = densitas::vector_adaptor::n_elements(predicted_quantiles_);
        auto prediction = densitas::matrix_adaptor::construct_uninitialized<MatrixType>(n_rows, n_quantiles);
        for (size_t i=0; i<n_rows; ++i) {
            auto weights = densitas::vector_adaptor::construct_uninitialized<VectorType>(models_.size());
            for (size_t j=0; j<models_.size(); ++j) {
                const auto prob_value = densitas::core::predict_proba_for_row<ElementType, VectorType>(models_[j], X, i);
                densitas::vector_adaptor::set_element<ElementType>(weights, j, prob_value);
            }
            const auto quantiles = densitas::math::quantiles_weighted<ElementType>(centers, weights, predicted_quantiles_, accuracy_predicted_quantiles_);
            densitas::core::assign_vector_to_row<ElementType>(prediction, i, quantiles);
        }
        return prediction;
    }

protected:

    std::vector<ModelType> models_;
    VectorType trained_quantiles_;
    VectorType predicted_quantiles_;
    ElementType accuracy_predicted_quantiles_;
};


} // densitas
