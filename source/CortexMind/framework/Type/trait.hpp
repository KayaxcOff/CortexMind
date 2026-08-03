//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TYPE_TRAIT_HPP
#define CORTEXMIND_FRAMEWORK_TYPE_TRAIT_HPP

#include <CortexMind/framework/Tools/types.hpp>
#include <CortexMind/framework/Type/dtype.hpp>
#include <tlx/concepts.hpp>
#include <cstdint>

namespace cortex::_fw {
    template<tlx::arithmetic_like>
    struct TypeTraits;

    template<>
    struct TypeTraits<std::int32_t> {
        static constexpr auto type = DType::Int32;
    };

    template<>
    struct TypeTraits<std::int64_t> {
        static constexpr auto type = DType::Int64;
    };

    template<>
    struct TypeTraits<half> {
        static constexpr auto type = DType::Float16;
    };

    template<>
    struct TypeTraits<float> {
        static constexpr auto type = DType::Float32;
    };

    template<>
    struct TypeTraits<double> {
        static constexpr auto type = DType::Float64;
    };

    template<>
    struct TypeTraits<bfloat16> {
        static constexpr auto type = DType::BFloat16;
    };

    template<>
    struct TypeTraits<qint16> {
        static constexpr auto type = DType::QInt16;
    };

    template<>
    struct TypeTraits<qint8> {
        static constexpr auto type = DType::QInt8;
    };

    template<>
    struct TypeTraits<quint16> {
        static constexpr auto type = DType::QUInt16;
    };

    template<>
    struct TypeTraits<quint8> {
        static constexpr auto type = DType::QUInt8;
    };

    template<tlx::arithmetic_like T>
    constexpr auto ttype_of = TypeTraits<T>::type;

    template<DType>
    struct RTypeTraits;

    template<>
    struct RTypeTraits<DType::Int32> {
        using type = std::int32_t;
    };

    template<>
    struct RTypeTraits<DType::Int64> {
        using type = std::int64_t;
    };

    template<>
    struct RTypeTraits<DType::Float16> {
        using type = half;
    };

    template<>
    struct RTypeTraits<DType::Float32> {
        using type = float;
    };

    template<>
    struct RTypeTraits<DType::Float64> {
        using type = double;
    };

    template<>
    struct RTypeTraits<DType::BFloat16> {
        using type = bfloat16;
    };

    template<>
    struct RTypeTraits<DType::QInt16> {
        using type = qint16;
    };

    template<>
    struct RTypeTraits<DType::QInt8> {
        using type = qint8;
    };

    template<>
    struct RTypeTraits<DType::QUInt16> {
        using type = quint16;
    };

    template<>
    struct RTypeTraits<DType::QUInt8> {
        using type = quint8;
    };

    template<>
    struct RTypeTraits<DType::Unknown> {
        using type = void;
    };

    template<DType T>
    using rtype_of = RTypeTraits<T>::type;
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TYPE_TRAIT_HPP