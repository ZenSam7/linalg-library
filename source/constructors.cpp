#include "matrix.h"
#include "exceptions.h"
#include <limits>
#include <algorithm>
#include <utility>

using linalg::Matrix;

/*      внутренние вспомогательные функции (видны только в этом файле)      */
namespace {

    // выделяет новую память на capacity элементов и копирует туда первые count (count <= capacity)
    // если new бросит исключение, то ничего не утекло и ничего не изменилось
    double* clone_block(const double* src, size_t count, size_t capacity) {
        if (count > capacity)
            throw OutOfMatrix();
        auto* ptr = new double[capacity];
        std::copy(src, src + count, ptr);
        return ptr;
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

// вычисляем rows * columns с защитой от переполнения size_t
size_t Matrix::_checked_size(size_t rows, size_t columns) {
    if (columns != 0 && rows > std::numeric_limits<size_t>::max() / columns)
        throw InvalidShape("размерчик слишком большой");
    return rows * columns;
}

Matrix::Matrix(size_t rows, size_t columns) {
    if (rows == 0 && columns != 0 || rows != 0 && columns == 0) {
        throw InvalidShape("число строк или столбцов = 0 при ненулевом числе другого");
    }
    size_t n = _checked_size(rows, columns);

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
        : _data_ptr(other._data_ptr),
          _rows(other._rows),
          _columns(other._columns),
          _capacity(other._capacity) {
    other._data_ptr = nullptr;
    other._rows = 0;
    other._columns = 0;
    other._capacity = 0;

}

Matrix::Matrix(std::initializer_list<double> list) {
    size_t n = list.size();
    if (n) {
        _data_ptr = new double[n];
        std::copy(list.begin(), list.end(), _data_ptr);
    }
    else {
        clear();
        return;
    }
    _rows = n; // столбец
    _columns = 1;
    _capacity = n;
}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> list) {
    size_t rows = list.size();
    size_t columns = rows ? list.begin()->size() : 0;
    if (columns == 0 || rows == 0) {
        clear();
        return;
    }

    for (const auto& row : list)    // сначала проверка, потом выделение памяти
        if (row.size() != columns)
            throw InvalidShape("строки в списке инициализации разной длины");

    size_t n = _checked_size(rows, columns);
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


