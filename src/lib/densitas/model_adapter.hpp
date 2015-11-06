#pragma once
#include <memory>


namespace densitas {
namespace model_adapter {

/**
 * Clones the given model
 */
template<typename ModelType>
std::unique_ptr<ModelType> clone(const ModelType& model)
{
    return std::unique_ptr<ModelType>{dynamic_cast<ModelType*>(model.clone().release())};
}

/**
 * Trains the model with given features X and target y. y should be
 * a binary target containing 'yes' and 'no'
 */
template<typename ModelType, typename MatrixType, typename VectorType>
void train(ModelType& model, MatrixType& X, VectorType& y)
{
    model.train(X, y);
}

/**
 * Predicts events using a trained model for given features X. Should
 * return probability values between 0 and 1
 */
template<typename VectorType, typename ModelType, typename MatrixType>
VectorType predict_proba(ModelType& model, MatrixType& X)
{
    return model.predict_proba(X);
}

/**
 * Returns the numerical representation of 'yes' as valid for the model type
 */
template<typename ModelType>
int yes()
{
    return 1;
}

/**
 * Returns the numerical representation of 'no' as valid for the model type
 */
template<typename ModelType>
int no()
{
    return -1;
}


} // model_adapter
} // densitas
