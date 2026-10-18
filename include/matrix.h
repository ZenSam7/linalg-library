#pragma once
#include <initializer_list>
// cstddef вместо stdlib и iosfwd вместо iostream т.к. они тянут много лишнего
#include <cstddef>
using std::size_t;
#include <iosfwd>
using std::ostream;


namespace linalg {
    static constexpr double DOUBLE_EPSILON = 1e-9;

    class Matrix {
    private:
        double* _data_ptr = nullptr;
        size_t _rows = 0, _columns = 0;
        size_t _capacity = 0;

        // служебная функция чтобы не писать постоянно 4 строчки
        void _reallocate(size_t new_capacity, bool keep_data=false);

    public:

        /*      разного рода конструкторы и деструктор      */

        Matrix();
        ~Matrix(); // ← один по жизни
        // inline explicit для перегрузки значением по умолчанию columns=1
        inline explicit Matrix(size_t rows) : Matrix(rows, 1) {};
        Matrix(size_t rows, size_t columns);
        Matrix(const Matrix& matrix); // копирования
        Matrix(     Matrix&& matrix) noexcept; // перемещения
        // для инициализации сразу с данными
        Matrix(std::initializer_list<double> list);                        // одномерный
        Matrix(std::initializer_list<std::initializer_list<double>> list); // двумерный


        /*      api      */

        // доступ к полям, для согласованности (хоть и в inline необходимости и нету)
        inline bool    empty()    const noexcept { return size() == 0;};
        inline size_t  rows()     const noexcept { return _rows; };
        inline size_t  columns()  const noexcept { return _columns; }
        inline size_t  capacity() const noexcept { return _capacity; }
        inline size_t  size()     const noexcept { return _rows * _columns; }
        inline double* begin()             noexcept { return _data_ptr; }
        inline double* end()               noexcept { return _data_ptr + _rows*_columns; }
        inline const double* begin() const noexcept { return _data_ptr; }
        inline const double* end()   const noexcept { return _data_ptr + _rows*_columns; }

        // интерфейс модификации
        void reshape(size_t rows, size_t columns);
        void reserve(size_t n);
        void shrink_to_fit();
        void clear();
        friend void swap(Matrix& matrix1, Matrix& matrix2) noexcept; // глобальная функция
        void swap(Matrix& matrix2)                         noexcept; // метод

        // всякого рода функции
        double norm()  const noexcept;
        double trace() const noexcept;
        double det()   const; // не noexcept т.к. будет создана копия матрицы → потенциальный bad_alloc
        size_t rank()  const; // не noexcept т.к. будет создана копия матрицы → потенциальный bad_alloc
        Matrix& gauss_forward ();
        Matrix& gauss_backward();
        // этим ребятам ↓ приватные поля не нужны, но для сохранения стиля пусть будут friend
        friend Matrix concatenate(const Matrix& matrix1, const Matrix& matrix2);
        friend Matrix transpose  (const Matrix& matrix);
        friend Matrix invert     (const Matrix& matrix);
        friend Matrix power      (const Matrix& matrix, int power);  // степень в т.ч. отрицательная
        friend Matrix solve      (const Matrix& matrix, const Matrix& vector);
        Matrix& T() { return *this=transpose(*this); }; // так удобнее


        /*      операторы       */

        Matrix& operator=(const Matrix& other); // присваивание копирующее → потенциальный bad_alloc
        Matrix& operator=(     Matrix&& other) noexcept; // перемещающее присваивание
              double& operator()(size_t row, size_t col)      ;
        const double& operator()(size_t row, size_t col) const;  // в return также может быть double
        friend std::ostream& operator<<(std::ostream& to_print, const Matrix& matrix);

        // арифметика
        Matrix operator+() const { return *this; };
        friend Matrix operator+(const Matrix& matrix1, const Matrix& matrix2);
        Matrix& operator+=(const Matrix& other);
        // "{return *this = *this+other;};" создаст временную матрицу и потом сделает
        // перемещение через "=". аналогично для -= и *=scalar
        // , но вот для *=matrix временная матрица неизбежна, так что её можно оставить

        Matrix operator-() const;
        friend Matrix operator-(const Matrix& matrix1, const Matrix& matrix2) { return matrix1 + (-matrix2);};
        Matrix& operator-=(const Matrix& other);

        friend Matrix operator*(const Matrix& matrix1, const Matrix& matrix2);
        friend Matrix operator*(const Matrix& matrix, double scalar);
        friend Matrix operator*(double scalar, const Matrix& matrix) { return matrix * scalar; };
        Matrix& operator*=(const Matrix& other) { return *this = (*this) * other; };
        Matrix& operator*=(double scalar);

        friend bool operator==(const Matrix& matrix1, const Matrix& matrix2);
        friend bool operator!=(const Matrix& matrix1, const Matrix& matrix2) { return !(matrix1==matrix2); };
    };
}