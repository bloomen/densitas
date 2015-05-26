#pragma once
#include "type_check.hpp"
#include "model_adapter.hpp"
#include "matrix_adapter.hpp"
#include "vector_adapter.hpp"
#include "densitas_error.hpp"
#include <vector>
#include <algorithm>
#include <string>


namespace densitas {
namespace math {


template<typename ModelType, typename ElementType, typename VectorType>
VectorType make_classification_target(const VectorType& y, ElementType lower, ElementType upper)
{
    densitas::core::check_element_type<ElementType>();
    const auto n_elem = densitas::vector_adapter::n_elements(y);
    auto target = densitas::vector_adapter::construct_uninitialized<VectorType>(n_elem);
    for (size_t i=0; i<n_elem; ++i) {
        const auto value = densitas::vector_adapter::get_element<ElementType>(y, i);
        const auto cls_result = value<lower || value>upper ? densitas::model_adapter::no<ModelType>() : densitas::model_adapter::yes<ModelType>();
        densitas::vector_adapter::set_element<ElementType>(target, i, cls_result);
    }
    return target;
}


template<typename ElementType, typename VectorType>
ElementType minimum(const VectorType& vector)
{
    densitas::core::check_element_type<ElementType>();
    const auto n_elem = densitas::vector_adapter::n_elements(vector);
    if (!(n_elem > 0))
        throw densitas::densitas_error("vector is of size zero");
    auto minimum = std::numeric_limits<ElementType>::max();
    for (size_t i=0; i<n_elem; ++i) {
        const auto value = densitas::vector_adapter::get_element<ElementType>(vector, i);
        if (value < minimum)
            minimum = value;
    }
    return minimum;
}


template<typename ElementType>
ElementType quantile(std::vector<ElementType>& data, ElementType proba)
{
    densitas::core::check_element_type<ElementType>();
    if (!data.size())
        throw densitas::densitas_error("vector contains no values");
    if (proba < 0 || proba > 1)
        throw densitas::densitas_error("proba must be between zero and one, not: " + std::to_string(proba));
    if (proba < 1.0 / data.size())
        return *std::min_element(data.begin(), data.end());
    if (proba == 1)
        return *std::max_element(data.begin(), data.end());
    const ElementType pos = data.size() * proba;
    const size_t ind = static_cast<size_t>(pos);
    const ElementType delta = pos - ind;
    std::nth_element(data.begin(), data.begin() + ind - 1, data.end());
    const ElementType i1 = *(data.begin() + ind - 1);
    const ElementType i2 = *std::min_element(data.begin() + ind, data.end());
    return i1 * (1. - delta) + i2 * delta;
}


template<typename ElementType, typename VectorType>
VectorType quantiles(const VectorType& vector, const VectorType& probas)
{
    densitas::core::check_element_type<ElementType>();
    const auto n_elem = densitas::vector_adapter::n_elements(vector);
    std::vector<ElementType> data(n_elem);
    for (size_t i=0; i<n_elem; ++i) {
        data[i] = densitas::vector_adapter::get_element<ElementType>(vector, i);
    }
    const auto n_probas = densitas::vector_adapter::n_elements(probas);
    auto quantiles = densitas::vector_adapter::construct_uninitialized<VectorType>(n_probas);
    for (size_t i=0; i<n_probas; ++i) {
        const auto proba = densitas::vector_adapter::get_element<ElementType>(probas, i);
        const auto quantile = densitas::math::quantile(data, proba);
        densitas::vector_adapter::set_element<ElementType>(quantiles, i, quantile);
    }
    return quantiles;
}


template<typename ElementType, typename VectorType>
VectorType quantiles_weighted(const VectorType& vector, const VectorType& weights, const VectorType& probas, ElementType accuracy)
{
    densitas::core::check_element_type<ElementType>();
    const auto n_elem = densitas::vector_adapter::n_elements(vector);
    if (n_elem != densitas::vector_adapter::n_elements(weights))
        throw densitas::densitas_error("vector and weights must be of equal size");
    if (!(accuracy > 0 && accuracy < 1))
        throw densitas::densitas_error("quantile accuracy must be between zero and one, not: " + std::to_string(accuracy));
    auto min_weight = densitas::math::minimum<ElementType>(weights);
    if (min_weight < accuracy) min_weight = accuracy;
    std::vector<size_t> counts(n_elem);
    for (size_t i=0; i<n_elem; ++i) {
        const auto weight = densitas::vector_adapter::get_element<ElementType>(weights, i);
        counts[i] = static_cast<size_t>(weight / min_weight);
    }
    const auto n_vals = std::accumulate(counts.begin(), counts.end(), 0);
    auto extended = n_vals > 0 ? densitas::vector_adapter::construct_uninitialized<VectorType>(n_vals) : vector;
    if (n_vals > 0) {
        size_t index = 0;
        for (size_t i=0; i<counts.size(); ++i) {
            const auto value = densitas::vector_adapter::get_element<ElementType>(vector, i);
            for (size_t j=0; j<counts[i]; ++j) {
                densitas::vector_adapter::set_element<ElementType>(extended, index, value);
                ++index;
            }
        }
    }
    return densitas::math::quantiles<ElementType>(extended, probas);
}


template<typename VectorType, typename ElementType>
VectorType linspace(ElementType start, ElementType end, size_t n)
{
    densitas::core::check_element_type<ElementType>();
    if (!(end > start))
        throw densitas::densitas_error("end is not larger than start");
    if (!(n > 1))
        throw densitas::densitas_error("n must be larger than one, not: " + std::to_string(n));
    const auto delta = (end - start) / (n - 1);
    auto linspace = densitas::vector_adapter::construct_uninitialized<VectorType>(n);
    for (size_t i=0; i<n; ++i) {
        densitas::vector_adapter::set_element<ElementType>(linspace, i, start + i*delta);
    }
    return linspace;
}


template<typename ElementType, typename VectorType>
VectorType centers(const VectorType& vector)
{
    densitas::core::check_element_type<ElementType>();
    const auto n_elem = densitas::vector_adapter::n_elements(vector);
    if (!(n_elem > 1))
        throw densitas::densitas_error("size of vector must be larger than one, not: " + std::to_string(n_elem));
    auto centers = densitas::vector_adapter::construct_uninitialized<VectorType>(n_elem - 1);
    for (size_t i=0; i<n_elem-1; ++i) {
        const auto first = densitas::vector_adapter::get_element<ElementType>(vector, i);
        const auto second = densitas::vector_adapter::get_element<ElementType>(vector, i+1);
        densitas::vector_adapter::set_element<ElementType>(centers, i, (first + second) / 2);
    }
    return centers;
}


} // math
} // densitas
