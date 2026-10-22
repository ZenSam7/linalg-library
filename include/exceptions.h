#pragma once

#include <stdexcept>
#include <string>

namespace {
    inline std::string put_description(const std::string& message, const std::string& description = "") {
        return description.empty() ? message : (message + ": " + description);
    }
}

struct MatrixMustBeSquared : std::runtime_error {
    explicit MatrixMustBeSquared(const std::string& description = "")
            : std::runtime_error(put_description("матрица должна быть квадратной", description)) {};
};

struct ShapesMismatch : std::runtime_error {
    explicit ShapesMismatch(const std::string& description = "")
            : std::runtime_error(put_description("размеры не подходят друг другу", description)) {};
};

struct MatrixIsSingular : std::runtime_error {
    explicit MatrixIsSingular(const std::string& description = "")
            : std::runtime_error(put_description("матрица вырождена", description)) {};
};

struct InvalidShape : std::runtime_error {
    explicit InvalidShape(const std::string& description = "")
            : std::runtime_error(put_description("размер неправильный", description)) {};
};

struct OutOfMatrix : std::runtime_error {
    explicit OutOfMatrix(const std::string& description = "")
            : std::runtime_error(put_description("выход за пределы матрицы", description)) {};
};

struct EmptyMatrix : std::runtime_error {
    explicit EmptyMatrix(const std::string& description = "")
            : std::runtime_error(put_description("матрица без строк и столбцов", description)) {};
};