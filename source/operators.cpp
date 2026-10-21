#include "matrix.h"
#include <exceptions.h>
#include <vector>
#include <sstream>
#include <cmath>

using linalg::Matrix;
using linalg::DOUBLE_EPSILON;

namespace {
    void require_same_shape(const Matrix& a, const Matrix& b) {
        if (a.rows() != b.rows() || a.columns() != b.columns())
            throw ShapesMismatch();
    }

    // сравнение с точностью: абсолютная для малых чисел, относительная для больших
    bool nearly_equal(double a, double b) noexcept {
        double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b))); // max(1, |a|, |b|)
        return std::fabs(a - b) <= DOUBLE_EPSILON * scale;
    }
}


/*      доступ к элементам      */

const double& Matrix::operator()(size_t row, size_t col) const {
    if (row >= _rows || col >= _columns)
        throw OutOfMatrix();
    return _data_ptr[row * _columns + col];
}

double& Matrix::operator()(size_t row, size_t col) {
    /* if (row >= _rows || col >= _columns)
        throw OutOfMatrix();
    return _data_ptr[row * _columns + col]; */

    // чтобы не копировать те же 3 строчки что в константном операторе написал эту хрень
    // (да, я люблю DRY)

    // 0) *this внутри неконстантного метода имеет тип Matrix&
    // 1) Matrix& → const Matrix&
    // 2) (row, col) от константного оператора, вернёт const double&
    // 3) const double& → просто double&
    // (изначально this неконстантный так то логика не нарушена от снятия const)
    return const_cast<double&>(static_cast<const Matrix&>(*this)(row, col));
}

/*      арифметика (методы)      */

namespace linalg {
    Matrix Matrix::operator-() const {
        Matrix result;
        result.reshape(_rows, _columns); // без зануления, всё равно перезапишем
        double* dst = result._data_ptr;
        const double* src = _data_ptr;
        for (size_t i = 0, n = size(); i < n; ++i)
            dst[i] = -src[i];
        return result;
    }

    Matrix& Matrix::operator+=(const Matrix& other) {
        require_same_shape(*this, other); // исключение до любых изменений
        double* dst = _data_ptr;
        const double* src = other._data_ptr;
        for (size_t i = 0, n = size(); i < n; ++i)
            dst[i] += src[i];
        return *this;
    }

    Matrix& Matrix::operator-=(const Matrix& other) {
        require_same_shape(*this, other); // исключение до любых изменений
        double* dst = _data_ptr;
        const double* src = other._data_ptr;
        for (size_t i = 0, n = size(); i < n; ++i)
            dst[i] -= src[i];
        return *this;
    }

    Matrix& Matrix::operator*=(double scalar) {
        double* dst = _data_ptr;
        for (size_t i = 0, n = size(); i < n; ++i)
            dst[i] *= scalar;
        return *this;
    }

    std::ostream& operator<<(std::ostream& to_print, const Matrix& matrix) {
        if (matrix.empty()) return to_print;
        const size_t rows = matrix.rows(), columns = matrix.columns();

        // 1 проход: превращаем элементы в строки и ищем ширину каждого столбца
        std::vector<std::string> cells;
        cells.reserve(matrix.size());  // сюда сложим все элементы-строки
        std::vector<size_t> widths(columns, 0);
        std::ostringstream buffer;   // формат берём у выходного потока

        buffer.flags(to_print.flags());          // копируем флаги
        buffer.precision(to_print.precision());  // копируем точность вывода
        buffer.imbue(to_print.getloc());         // копируем локаль

        // записываем и сами строки и их ширину
        const double* p = matrix.begin();
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < columns; ++j, ++p) {
                buffer.str("");
                buffer << *p;
                cells.push_back(buffer.str());

                widths[j] = std::max(widths[j], cells.back().size());
            }
        }

        // 2 проход: печать с выравниванием по правому краю; после каждой строки '\n'
        size_t idx = 0;
        for (size_t i = 0; i < rows; ++i) {
            to_print << '|';
            for (size_t j = 0; j < columns; ++j, ++idx) {
                if (j) to_print << ' ';  // пробел после палочки
                for (size_t pad = widths[j] - cells[idx].size(); pad > 0; --pad)
                    to_print << ' ';
                to_print << cells[idx];
            }
            to_print << '|' << '\n';
        }
        return to_print;
    }

    Matrix operator+(const Matrix& matrix1, const Matrix& matrix2) {
        require_same_shape(matrix1, matrix2);
        Matrix result;
        result.reshape(matrix1.rows(), matrix1.columns());
        double* dst = result.begin();
        const double* a = matrix1.begin();
        const double* b = matrix2.begin();
        for (size_t i = 0, n = result.size(); i < n; ++i)
            dst[i] = a[i] + b[i];
        return result;
    }

    Matrix operator*(const Matrix& matrix, double scalar) {
        Matrix result;
        result.reshape(matrix.rows(), matrix.columns());
        double* dst = result.begin();
        const double* src = matrix.begin();
        for (size_t i = 0, n = result.size(); i < n; ++i)
            dst[i] = src[i] * scalar;
        return result;
    }

    // порядок циклов i-k-j: внутренний цикл идёт подряд по строкам matrix2 и result,
    // это кэш-дружелюбно и компилятор может векторизовать )см мой конспектик SIMD)
    // (в отличие от классического i-j-k, где matrix2 читается по столбцу)
    // т.е. тут мы
    Matrix operator*(const Matrix& matrix1, const Matrix& matrix2) {
        if (matrix1.columns() != matrix2.rows())
            throw ShapesMismatch();
        const size_t row1 = matrix1.rows(), col1 = matrix1.columns(), col2 = matrix2.columns();
        Matrix result;
        result.reshape(row1, col2);
        // заполняем нулями (потом будем прибавлять, а не перезаписывать)
        std::fill(result.begin(), result.begin() + result.size(), 0.0);

        const double* m1 = matrix1.begin();
        const double* m2 = matrix2.begin();
        double* dst = result.begin();

        for (size_t i = 0; i < row1; ++i) {
            // записываем в dst по строкам
            double* dst_row = dst + i * col2;

            for (size_t k = 0; k < col1; ++k) {
                // внешний цикл по КОЛОНКАМ первой матрицы

                const double m1_ik = m1[i * col1 + k]; // matrix(i, k) элемент
                const double* m2_row = m2 + k * col2;  // k'я строка
                for (size_t j = 0; j < col2; ++j)
                    // внутренний по СТРОКАМ (в кеш загружаются строки)
                    dst_row[j] += m1_ik * m2_row[j];
            }
        }
        return result;
    }

    bool operator==(const Matrix& matrix1, const Matrix& matrix2) {
        if (matrix1.rows() != matrix2.rows() || matrix1.columns() != matrix2.columns())
            return false;
        const double* a = matrix1.begin();
        const double* b = matrix2.begin();
        for (size_t i = 0, n = matrix1.size(); i < n; ++i)
            if (!nearly_equal(a[i], b[i])) return false;
        return true;
    }

}
