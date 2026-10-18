#include "../include/matrix.h"
#include "../include/exceptions.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using linalg::Matrix;
using linalg::DOUBLE_EPSILON;


/*      внутренние вспомогательные функции (видны только в этом файле)      */

namespace {

    // вычисляем rows * columns с защитой от переполнения size_t
    size_t checked_size(size_t rows, size_t columns) {
        if (columns != 0 && rows > std::numeric_limits<size_t>::max() / columns)
            throw InvalidShape("размерчик слишком большой");
        return rows * columns;
    }

    // выделяет новую память на capacity элементов и копирует туда первые count (count <= capacity)
    // если new бросит исключение, то ничего не утекло и ничего не изменилось
    double* clone_block(const double* src, size_t count, size_t capacity) {
        if (count > capacity)
            throw OutOfMatrix();
        auto* ptr = new double[capacity];
        std::copy(src, src + count, ptr);
        return ptr;
    }

    // сравнение с точностью: абсолютная для малых чисел, относительная для больших
    bool nearly_equal(double a, double b) noexcept {
        double scale = std::max({1.0, std::fabs(a), std::fabs(b)});
        return std::fabs(a - b) <= DOUBLE_EPSILON * scale;
    }

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

    void require_same_shape(const Matrix& a, const Matrix& b) {
        if (a.rows() != b.rows() || a.columns() != b.columns())
            throw ShapesMismatch();
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
    size_t forward_pass(double* a, size_t rows, size_t cols, double tol, bool& odd_swaps) noexcept {
        size_t rank = 0;                            // он же номер текущей строки-ведущей
        for (size_t col = 0; col < cols && rank < rows; ++col) {
            size_t best = rank;
            double best_abs = std::fabs(a[rank * cols + col]);
            for (size_t r = rank + 1; r < rows; ++r) {
                double v = std::fabs(a[r * cols + col]);
                if (v > best_abs) { best_abs = v; best = r; }
            }
            if (best_abs <= tol) {                  // в столбце нет ведущего, остатки зануляем
                for (size_t r = rank; r < rows; ++r) a[r * cols + col] = 0.0;
                continue;
            }
            double* pivot_row = a + rank * cols;
            if (best != rank) {
                std::swap_ranges(pivot_row, pivot_row + cols, a + best * cols);
                odd_swaps = !odd_swaps;
            }
            const double pivot = pivot_row[col];
            for (size_t r = rank + 1; r < rows; ++r) {
                double* row = a + r * cols;
                double factor = row[col] / pivot;
                if (factor == 0.0) continue;
                row[col] = 0.0;                     // точный ноль, а не 1e-17
                for (size_t c = col + 1; c < cols; ++c)
                    row[c] -= factor * pivot_row[c];
            }
            ++rank;
        }
        return rank;
    }

    // обратный ход: идём снизу вверх, делаем ведущий элемент строки единицей
    // и зануляем столбец над ним. ожидает ступенчатый вид (результат forward_pass)
    void backward_pass(double* a, size_t rows, size_t cols, double tol) noexcept {
        for (size_t r = std::min(rows, cols); r-- > 0; ) {  // строки >= cols в ступенчатом виде нулевые
            double* row = a + r * cols;
            size_t p = r;                                   // ведущий в строке r не может быть левее столбца r
            while (p < cols && std::fabs(row[p]) <= tol) ++p;
            if (p == cols) continue;                        // нулевая строка

            const double pivot = row[p];
            for (size_t c = p + 1; c < cols; ++c)
                if (row[c] != 0.0) row[c] /= pivot;         // иначе 0 / (отрицательное) даёт -0 в выводе
            row[p] = 1.0;

            for (size_t k = 0; k < r; ++k) {
                double* up = a + k * cols;
                double factor = up[p];
                if (factor == 0.0) continue;
                up[p] = 0.0;
                for (size_t c = p + 1; c < cols; ++c)
                    up[c] -= factor * row[c];
            }
        }
    }
}


/*      конструкторы, деструктор, присваивания      */

// у нас в полях будет не мусор т.к. в заголовочном инициализируемся
// нулями, значения которых и использует default
Matrix::Matrix() = default;

Matrix::~Matrix() { delete[] _data_ptr; }

// служебная функция чтобы не писать постоянно 4 строчки
// порядок важен: сначала new (может бросить bad_alloc, тогда *this цел), потом delete[]
// keep_data = true копирует size() старых элементов (нужно size() <= new_capacity)
void Matrix::_reallocate(size_t new_capacity, bool keep_data) {
    double* new_ptr = nullptr;
    if (new_capacity)
        new_ptr = keep_data ? clone_block(_data_ptr, size(), new_capacity)
                            : new double[new_capacity];
    delete[] _data_ptr;
    _data_ptr = new_ptr;
    _capacity = new_capacity;
}

Matrix::Matrix(size_t rows, size_t columns) {
    size_t n = checked_size(rows, columns);
    if (n) _data_ptr = new double[n]();  // () = value-init, т.е. нули
    // поля меняются только после успешного new
    _rows = rows;
    _columns = columns;
    _capacity = n;
}

Matrix::Matrix(const Matrix& other) {
    size_t n = other.size();
    if (n) _data_ptr = clone_block(other._data_ptr, n, n);  // вместимость копии = ровно сколько нужно
    _rows = other._rows;
    _columns = other._columns;
    _capacity = n;
}

Matrix::Matrix(Matrix&& other) noexcept
        : _data_ptr(std::exchange(other._data_ptr, nullptr)),
          _rows(std::exchange(other._rows, 0)),
          _columns(std::exchange(other._columns, 0)),
          _capacity(std::exchange(other._capacity, 0)) {/*спиздили*/}

Matrix::Matrix(std::initializer_list<double> list) {
    size_t n = list.size();
    if (n) {
        _data_ptr = new double[n];
        std::copy(list.begin(), list.end(), _data_ptr);
    }
    _rows = n; // столбец
    _columns = 1;
    _capacity = n;
}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> list) {
    size_t rows = list.size();
    size_t columns = rows ? list.begin()->size() : 0;
    for (const auto& row : list)    // сначала проверка, потом выделение памяти
        if (row.size() != columns)
            throw InvalidShape("строки в списке инициализации разной длины");

    // checked_size не нужен: переполнения быть не может (столько элементов уже лежит в памяти)
    size_t n = rows * columns;
    if (n) {
        _data_ptr = new double[n];
        double* dst = _data_ptr;
        for (const auto& row : list)
            // копируем т.к. initializer_list это const double*, а у нас _data_ptr не const
            dst = std::copy(row.begin(), row.end(), dst);
    }
    _rows = rows;
    _columns = columns;
    _capacity = n;
}

// копирующий
Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this;
    size_t n = other.size();
    if (n > _capacity)
        _reallocate(n); // если бросит ошибку, то *this останется нетронутым
    std::copy(other.begin(), other.end(), _data_ptr);
    _rows = other._rows;
    _columns = other._columns;
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    // инициализируем tmp по rvalye (крадём ресурсы у other), см метод выше (обнуление other там же)
    Matrix tmp(std::move(other));
    swap(tmp);  // свапаем поля *this и other нашим методом swap
    return *this;
}   // вот тут tmp с нашими старыми данными умирает (неявное delete[] _date_ptr старых данных)
/* хотя намного понятнее написать просто:
   if (this != &other) {
        delete[] _data_ptr;
        _data_ptr = std::exchange(other._data_ptr, nullptr);
        _rows     = std::exchange(other._rows, 0);
        _columns  = std::exchange(other._columns, 0);
        _capacity = std::exchange(other._capacity, 0);
    }
    return *this;
 (любитель поусложнять, да пописать многострочные комментарии, да, уж простите) */



/*      интерфейс модификации      */

// после reshape содержимое не определено (по заданию старые элементы можно не сохранять),
// поэтому новую память не зануляем, (это экономит проход по всему массиву)
void Matrix::reshape(size_t rows, size_t columns) {
    size_t n = checked_size(rows, columns);
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
    // (да, я люблю DRY) (так же из-за as_const требуем c++17)

    // 0) *this внутри неконстантного метода имеет тип Matrix&
    // 1) Matrix& → const Matrix&
    // 2) (row, col) от константного оператора, вернёт const double&
    // 3) const double& → просто double&
    // (изначально this неконстантный так то логика не нарушена от снятия const)
    return const_cast<double&>(static_cast<const Matrix&>(*this)(row, col));
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

double Matrix::det() const {
    if (_rows != _columns)
        throw MatrixMustBeSquared();
    Matrix tmp(*this);                              // работаем с копией, this не меняем
    bool odd_swaps = false;
    size_t rank = forward_pass(tmp._data_ptr, _rows, _columns,
                               zero_tolerance(_data_ptr, size()), odd_swaps);
    if (rank < _rows) return 0.0;                   // вырожденная

    double result = odd_swaps ? -1.0 : 1.0;         // каждая перестановка строк меняет знак
    for (size_t i = 0; i < _rows; ++i)
        result *= tmp._data_ptr[i * (_columns + 1)];
    return result;
}

size_t Matrix::rank() const {
    Matrix tmp(*this);
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


/*      арифметика (методы)      */

Matrix Matrix::operator-() const {
    Matrix result;
    result.reshape(_rows, _columns);                // без зануления, всё равно перезапишем
    double* dst = result._data_ptr;
    const double* src = _data_ptr;
    for (size_t i = 0, n = size(); i < n; ++i)
        dst[i] = -src[i];
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    require_same_shape(*this, other);               // исключение до любых изменений
    double* dst = _data_ptr;
    const double* src = other._data_ptr;
    for (size_t i = 0, n = size(); i < n; ++i)
        dst[i] += src[i];
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    require_same_shape(*this, other);
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


/*      свободные функции и операторы (которые friend)      */

namespace linalg {

    void swap(Matrix& matrix1, Matrix& matrix2) noexcept { matrix1.swap(matrix2); }

    std::ostream& operator<<(std::ostream& to_print, const Matrix& matrix) {
        if (matrix.empty()) return to_print;
        const size_t rows = matrix.rows(), columns = matrix.columns();

        // 1 проход: превращаем элементы в строки и ищем ширину каждого столбца
        std::vector<std::string> cells;
        cells.reserve(matrix.size());
        std::vector<size_t> widths(columns, 0);
        std::ostringstream buffer;                  // формат (точность и т.д.) берём у выходного потока
        buffer.flags(to_print.flags());
        buffer.precision(to_print.precision());
        buffer.imbue(to_print.getloc());

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
            to_print.put('|');
            for (size_t j = 0; j < columns; ++j, ++idx) {
                if (j) to_print.put(' ');
                for (size_t pad = widths[j] - cells[idx].size(); pad > 0; --pad)
                    to_print.put(' ');
                to_print << cells[idx];
            }
            to_print.put('|');
            to_print.put('\n');
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

    // порядок циклов i-k-j: внутренний цикл идёт подряд по строкам b и result,
    // это кэш-дружелюбно и компилятор может векторизовать (в отличие от классического i-j-k, где b читается по столбцу)
    Matrix operator*(const Matrix& matrix1, const Matrix& matrix2) {
        if (matrix1.columns() != matrix2.rows())
            throw ShapesMismatch();
        const size_t n = matrix1.rows(), m = matrix1.columns(), p = matrix2.columns();
        Matrix result;
        result.reshape(n, p);

        const double* a = matrix1.begin();
        const double* b = matrix2.begin();
        double* c = result.begin();
        for (size_t i = 0; i < n; ++i) {
            double* c_row = c + i * p;
            std::fill(c_row, c_row + p, 0.0);
            for (size_t k = 0; k < m; ++k) {
                const double a_ik = a[i * m + k];
                const double* b_row = b + k * p;
                for (size_t j = 0; j < p; ++j)
                    c_row[j] += a_ik * b_row[j];
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