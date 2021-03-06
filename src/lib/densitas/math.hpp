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
    for (std::size_t i=0; i<n_elem; ++i) {
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
    for (std::size_t i=0; i<n_elem; ++i) {
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
    const std::size_t ind = static_cast<std::size_t>(pos);
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
    for (std::size_t i=0; i<n_elem; ++i) {
        data[i] = densitas::vector_adapter::get_element<ElementType>(vector, i);
    }
    const auto n_probas = densitas::vector_adapter::n_elements(probas);
    auto quantiles = densitas::vector_adapter::construct_uninitialized<VectorType>(n_probas);
    for (std::size_t i=0; i<n_probas; ++i) {
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
    std::vector<std::size_t> counts(n_elem);
    for (std::size_t i=0; i<n_elem; ++i) {
        const auto weight = densitas::vector_adapter::get_element<ElementType>(weights, i);
        counts[i] = static_cast<std::size_t>(weight / min_weight);
    }
    const auto n_vals = std::accumulate(counts.begin(), counts.end(), 0);
    auto extended = n_vals > 0 ? densitas::vector_adapter::construct_uninitialized<VectorType>(n_vals) : vector;
    if (n_vals > 0) {
        std::size_t index = 0;
        for (std::size_t i=0; i<counts.size(); ++i) {
            const auto value = densitas::vector_adapter::get_element<ElementType>(vector, i);
            for (std::size_t j=0; j<counts[i]; ++j) {
                densitas::vector_adapter::set_element<ElementType>(extended, index, value);
                ++index;
            }
        }
    }
    return densitas::math::quantiles<ElementType>(extended, probas);
}


template<typename VectorType, typename ElementType>
VectorType linspace(ElementType start, ElementType end, std::size_t n)
{
    densitas::core::check_element_type<ElementType>();
    if (!(end > start))
        throw densitas::densitas_error("end is not larger than start");
    if (!(n > 1))
        throw densitas::densitas_error("n must be larger than one, not: " + std::to_string(n));
    const auto delta = (end - start) / (n - 1);
    auto linspace = densitas::vector_adapter::construct_uninitialized<VectorType>(n);
    for (std::size_t i=0; i<n; ++i) {
        densitas::vector_adapter::set_element<ElementType>(linspace, i, start + i*delta);
    }
    return linspace;
}


template<typename ElementType, typename VectorType>
VectorType centers(const VectorType& data, const VectorType& quantiles)
{
    densitas::core::check_element_type<ElementType>();
    const auto n_data = densitas::vector_adapter::n_elements(data);
    if (!(n_data > 0))
        throw densitas::densitas_error("size of data is zero");
    const auto n_quant = densitas::vector_adapter::n_elements(quantiles);
    if (!(n_quant > 1))
        throw densitas::densitas_error("size of quantiles must be larger than one, not: " + std::to_string(n_quant));
    const auto n_elem = n_quant - 1;
    std::vector<std::size_t> counter(n_elem, 0);
    std::vector<ElementType> accumulator(n_elem, 0.);
    for (std::size_t i=0; i<n_data; ++i) {
        int current_j = -1;
        const auto value = densitas::vector_adapter::get_element<ElementType>(data, i);
        for (std::size_t j=0; j<n_elem; ++j) {
            const auto first = densitas::vector_adapter::get_element<ElementType>(quantiles, j);
            const auto second = densitas::vector_adapter::get_element<ElementType>(quantiles, j + 1);
            if (value>=first && value<=second) {
                accumulator[j] += value;
                ++counter[j];
                current_j = j;
            }
            if (current_j>0 && static_cast<size_t>(current_j)!=j) break;
        }
    }
    auto centers = densitas::vector_adapter::construct_uninitialized<VectorType>(n_elem);
    for (std::size_t j=0; j<n_elem; ++j) {
        if (counter[j]==0) counter[j] = 1;
        densitas::vector_adapter::set_element<ElementType>(centers, j, accumulator[j] / counter[j]);
    }
    return centers;
}


} // math
} // densitas
