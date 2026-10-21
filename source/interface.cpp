#include "matrix.h"
#include "exceptions.h"

#include <algorithm>
#include <cmath>
#include <sstream>

using linalg::Matrix;
using linalg::DOUBLE_EPSILON;


/*      внутренние вспомогательные функции (видны только в этом файле)      */
namespace {

    // максимальный элемент массива по модулю
    double max_abs(const double* ptr, size_t n) noexcept {
        double result = 0.0;
        for (size_t i = 0; i < n; ++i)
            result = std::max(result, std::fabs(ptr[i]));
        return result;
    }

    // порог, ниже которого элемент считаем нулём (относительно самого большого элемента)
    double zero_tolerance(const double* ptr, size_t n) noexcept {
        return DOUBLE_EPSILON * max_abs(ptr, n);
    }

    Matrix make_identity(size_t n) {
        Matrix result(n, n);        // нули
        double* ptr = result.begin();
        for (size_t i = 0; i < n; ++i)
            ptr[i * (n + 1)] = 1.0; // диагональ
        return result;
    }

    // прямой ход гаусса с частичным выбором ведущего элемента (по модулю максимум в столбце)
    // выбор максимума нужен для устойчивости: делить на маленькое число = взрыв погрешности
    // приводит a (rows x cols) к ступенчатому виду, возвращает ранг
    // odd_swaps переключается на каждую перестановку строк (нужно для знака определителя)
    size_t forward_pass(double* arr, size_t rows, size_t cols, double zero_bound, bool& odd_swaps) noexcept {
        size_t rank = 0;  // он же номер текущей ведущей строки
        for (size_t col = 0; col < cols && rank < rows; ++col) {
            size_t best = rank;  // № строки с максимальным элементом
            double best_abs = std::fabs(arr[rank * cols + col]);
            // выбираем максимум из столбца (из строк ниже ведущего элемента)
            for (size_t r = rank + 1; r < rows; ++r) {
                double v = std::fabs(arr[r * cols + col]);
                if (v > best_abs) { best_abs = v; best = r; }
            }
            // в столбце нет максимального элемента (он 0), остатки зануляем, идём к следующему столбцу
            if (best_abs <= zero_bound) {   // zero_bound = граница нуля (какой элемент считаем нулём)
                for (size_t r = rank; r < rows; ++r) arr[r * cols + col] = 0.0;
                continue;
            }
            // строка с максимальным элементом (мо умолчанию текущая)
            double* pivot_row = arr + rank * cols;
            if (best != rank) {
                // меняем местами строку [pivot_row; pivot_row+cols) и
                // [arr + best*cols;  arr + best*cols + pivot_row + cols)
                std::swap_ranges(pivot_row, pivot_row + cols, arr + best * cols);
                odd_swaps = !odd_swaps;
            }

            // зануляем все элементы ниже главного значения (вычитаем главную строку)
            const double pivot = pivot_row[col];
            for (size_t r = rank + 1; r < rows; ++r) {
                double* down = arr + r * cols;
                double factor = down[col] / pivot;

                // если в этой строке и так 0 под главным, то цикл далее можем вообще не делать
                // (тут не zero_bound т.к. factor может быть очень мелким, но элемент матрицы большим
                // и в теории они могут друг друга компенсировать)
                if (factor == 0.0)
                    // down[col] уже = 0.0
                    continue;

                // вычитаем главную строку
                down[col] = 0.0;
                for (size_t c = col + 1; c < cols; ++c)
                    down[c] -= factor * pivot_row[c];
            }
            ++rank;
        }
        return rank;
    }

    // обратный ход: идём снизу вверх, делаем ведущий элемент строки единицей
    // и зануляем столбец над ним. ожидает ступенчатый вид (результат forward_pass)
    void backward_pass(double* arr, size_t rows, size_t cols, double zero_bound) noexcept {
        for (size_t r = std::min(rows, cols); r-- > 0;) {  // строки >= cols в ступенчатом виде нулевые
            // выбираем главное значение главной строки
            double* row = arr + r * cols;
            size_t p = r;  // ведущий в строке r не может быть левее столбца r
            while (p < cols && std::fabs(row[p]) <= zero_bound) ++p;
            if (p == cols) continue;   // нулевая строка
            const double pivot = row[p];

            // делим строку на главное значение
            for (size_t c = p + 1; c < cols; ++c)
                if (std::fabs(row[c]) > zero_bound) row[c] /= pivot;
            row[p] = 1.0;

            // зануляем столбец выше главного значения
            for (size_t k = 0; k < r; ++k) {
                double* up = arr + k * cols;
                double factor = up[p];

                // если в этой строке и так 0 под главным, то цикл далее можем вообще не делать
                // (тут не zero_bound т.к. factor может быть очень мелким, но элемент матрицы большим
                // и в теории они могут друг друга компенсировать)
                if (factor == 0.0)
                    // up[p] уже = 0.0
                    continue;

                // вычитаем главную строку
                up[p] = 0.0;
                for (size_t c = p + 1; c < cols; ++c)
                    up[c] -= factor * row[c];
            }
        }
    }
}

/*      вычислительные методы      */

// sqrt(сумма квадратов)
double Matrix::norm() const noexcept {
    double sum = 0.0;
    for (double p : *this)
        sum += p * p;
    /* то же, что:
    for (const double* p = begin(); p != end(); ++p)
        sum += *p * *p;
    */
    return std::sqrt(sum);
}

// для неквадратной матрицы считаем сумму a_ii по min(rows, columns) элементам
// (так не нужно бросать исключение, а метод остаётся noexcept)
double Matrix::trace() const noexcept {
    size_t n = std::min(_rows, _columns);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
        sum += _data_ptr[i * (_columns + 1)];
    return sum;
}

// приводим к ступенчатому виду (гауссом) потом умножаем диагональные элементы
double Matrix::det() const {
    if (_rows != _columns)
        throw MatrixMustBeSquared();
    Matrix temp(*this);  // работаем с копией, this не меняем
    bool odd_swaps = false;
    size_t rank = forward_pass(temp._data_ptr, _rows, _columns,
                               zero_tolerance(_data_ptr, size()), odd_swaps);
    if (rank < _rows) return 0.0; // вырожденная

    double result = odd_swaps ? -1.0 : 1.0; // каждая перестановка строк меняет знак
    // произведение диагональных элем
    for (size_t i = 0; i < _rows; ++i)
        result *= temp(i, i);
    return result;
}

size_t Matrix::rank() const {
    Matrix tmp(*this);  // главное this случайно не поменять
    bool odd_swaps = false;
    return forward_pass(tmp._data_ptr, _rows, _columns,
                        zero_tolerance(_data_ptr, size()), odd_swaps);
}

Matrix& Matrix::gauss_forward() {
    bool odd_swaps = false;
    forward_pass(_data_ptr, _rows, _columns, zero_tolerance(_data_ptr, size()), odd_swaps);
    return *this;
}

Matrix& Matrix::gauss_backward() {
    backward_pass(_data_ptr, _rows, _columns, zero_tolerance(_data_ptr, size()));
    return *this;
}


/*      интерфейс модификации      */

// после reshape содержимое не определено (по заданию старые элементы можно не сохранять),
// поэтому новую память не зануляем, (это экономит проход по всему массиву)
void Matrix::reshape(size_t rows, size_t columns) {
    if (rows == 0 && columns == 0) {
        clear();
        return;
    }
    else if (rows == 0 && columns != 0 || rows != 0 && columns == 0) {
        throw InvalidShape("число строк или столбцов = 0 при ненулевом числе другого");
    }

    size_t n = _checked_size(rows, columns);
    if (n > _capacity)
        _reallocate(n);
    _rows = rows;
    _columns = columns;
}

// как у std::vector: только увеличивает вместимость, элементы сохраняет
void Matrix::reserve(size_t n) {
    if (n > _capacity) _reallocate(n, true);
}

// обрезаем массив до size() ("перевыделяет память в соответствии с текущим размером матрицы")
void Matrix::shrink_to_fit() {
    if (size() != _capacity)  // size() > _capacity
        _reallocate(size(), true);
}

// capacity и память не трогаем, мы их в процессе дальнейшей
// работы с матрицей сами перезатрём новыми данными (или перевыделим память)
void Matrix::clear() {
    _rows = 0;
    _columns = 0;
}

void Matrix::swap(Matrix& other) noexcept {
    std::swap(_data_ptr, other._data_ptr);
    std::swap(_rows, other._rows);
    std::swap(_columns, other._columns);
    std::swap(_capacity, other._capacity);
}

/*      свободные функции и операторы (которые friend)      */

namespace linalg {

    void swap(Matrix& matrix1, Matrix& matrix2) noexcept { matrix1.swap(matrix2); }

    Matrix concatenate(const Matrix& matrix1, const Matrix& matrix2) {
        if (matrix1.rows() != matrix2.rows())
            throw ShapesMismatch("количество строк разное");
        const size_t rows = matrix1.rows();
        const size_t c1 = matrix1.columns(), c2 = matrix2.columns();
        Matrix result;
        result.reshape(rows, c1 + c2);

        const double* a = matrix1.begin();
        const double* b = matrix2.begin();
        double* dst = result.begin();
        for (size_t i = 0; i < rows; ++i) {
            dst = std::copy(a + i * c1, a + (i + 1) * c1, dst);
            dst = std::copy(b + i * c2, b + (i + 1) * c2, dst);
        }
        return result;
    }

    // транспонирование блоками 32x32: так и чтение, и запись остаются в кэше
    // (наивный вариант на больших матрицах пишет в память с шагом rows и постоянно мажет мимо кэша)
    Matrix transpose(const Matrix& matrix) {
        constexpr size_t block = 32;
        const size_t rows = matrix.rows(), cols = matrix.columns();
        Matrix result;
        result.reshape(cols, rows);

        const double* src = matrix.begin();
        double* dst = result.begin();
        for (size_t ib = 0; ib < rows; ib += block) {
            const size_t i_end = std::min(ib + block, rows);
            for (size_t jb = 0; jb < cols; jb += block) {
                const size_t j_end = std::min(jb + block, cols);
                for (size_t i = ib; i < i_end; ++i)
                    for (size_t j = jb; j < j_end; ++j)
                        dst[j * rows + i] = src[i * cols + j];
            }
        }
        return result;
    }

    // решаем A*X = F для расширенной матрицы [A | F]: прямой ход, проверка на вырожденность, обратный ход
    // после обратного хода слева получается единичная, а справа решение
    // (f может иметь несколько столбцов, тогда решаются сразу несколько систем)
    Matrix solve(const Matrix& matrix, const Matrix& vector) {
        const size_t n = matrix.rows();
        if (matrix.columns() != n)
            throw MatrixMustBeSquared();
        if (vector.rows() != n)
            throw ShapesMismatch("вектор не подходит по размеру для матрицы");
        const size_t k = vector.columns();
        const size_t width = n + k;

        Matrix augmented = concatenate(matrix, vector);
        double* a = augmented.begin();
        const double tol = zero_tolerance(matrix.begin(), matrix.size());  // порог по A, а не по [A|F]

        bool odd_swaps = false;
        forward_pass(a, n, width, tol, odd_swaps);
        for (size_t i = 0; i < n; ++i)              // нет ведущего на диагонали = A вырождена
            if (std::fabs(a[i * width + i]) <= tol)
                throw MatrixIsSingular();
        backward_pass(a, n, width, tol);

        Matrix result;
        result.reshape(n, k);
        double* dst = result.begin();
        for (size_t i = 0; i < n; ++i)
            dst = std::copy(a + i * width + n, a + (i + 1) * width, dst);
        return result;
    }

    Matrix invert(const Matrix& matrix) {           // A^-1 = решение A*X = E
        if (matrix.rows() != matrix.columns())
            throw MatrixMustBeSquared();
        return solve(matrix, make_identity(matrix.rows()));
    }

    // быстрое возведение в степень: O(log p) умножений вместо p
    Matrix power(const Matrix& matrix, int exponent) {
        if (matrix.rows() != matrix.columns())
            throw MatrixMustBeSquared();
        if (exponent == 0) return make_identity(matrix.rows());

        long long e = exponent;                     // long long, чтобы -INT_MIN не переполнился
        if (e < 0) e = -e;
        Matrix base = (exponent < 0) ? invert(matrix) : Matrix(matrix);

        while ((e & 1) == 0) {                      // пропускаем нулевые младшие биты без умножения на единичную
            base *= base;
            e >>= 1;
        }
        Matrix result = base;
        for (e >>= 1; e != 0; e >>= 1) {
            base *= base;
            if (e & 1) result *= base;
        }
        return result;
    }
}
