#pragma once


namespace densitas {
namespace model_adaptor {


template<typename ModelType, typename MatrixType, typename VectorType>
void train(ModelType& model, MatrixType& X, VectorType& y)
{
    model.train(X, y);
}


template<typename VectorType, typename ModelType, typename MatrixType>
VectorType predict_proba(ModelType& model, MatrixType& X)
{
    return model.predict_proba(X);
}


template<typename ModelType>
constexpr int yes()
{
    return 1;
}


template<typename ModelType>
constexpr int no()
{
    return -1;
}


} // model_adaptor
} // densitas
