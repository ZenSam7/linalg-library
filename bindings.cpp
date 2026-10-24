#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <sstream>
#include <string>
#include <vector>

#include "matrix.h"
#include "exceptions.h"

namespace py = pybind11;
using linalg::Matrix;

namespace {

// поддержка отрицательных индексов как в питоне
    size_t norm_index(py::ssize_t i, size_t n, const char* axis) {
        const auto sn = static_cast<py::ssize_t>(n);
        if (i < 0) i += sn;
        if (i < 0 || i >= sn)
            throw py::index_error(std::string("индекс вне матрицы по оси: ") + axis);
        return static_cast<size_t>(i);
    }

    Matrix from_rows(const std::vector<std::vector<double>>& rows) {
        if (rows.empty()) return Matrix();
        const size_t cols = rows[0].size();
        for (const auto& r : rows)
            if (r.size() != cols) throw py::value_error("строки разной длины");
        Matrix m(rows.size(), cols);
        for (size_t i = 0; i < rows.size(); ++i)
            for (size_t j = 0; j < cols; ++j)
                m(i, j) = rows[i][j];
        return m;
    }

    Matrix from_flat(const std::vector<double>& v) {   // одномерный список → столбец
        if (v.empty()) return Matrix();
        Matrix m(v.size(), 1);
        for (size_t i = 0; i < v.size(); ++i) m(i, 0) = v[i];
        return m;
    }

    std::string to_string(const Matrix& m) {
        std::ostringstream os;
        os << m;
        return os.str();
    }

} // namespace

PYBIND11_MODULE(linalg, m) {
    m.doc() = "Линейная алгебра: класс Matrix и функции над ним";

    /* ---------- исключения ---------- */
    py::register_exception<MatrixMustBeSquared>(m, "MatrixMustBeSquared", PyExc_RuntimeError);
    py::register_exception<ShapesMismatch>     (m, "ShapesMismatch",      PyExc_RuntimeError);
    py::register_exception<MatrixIsSingular>   (m, "MatrixIsSingular",    PyExc_RuntimeError);
    py::register_exception<InvalidShape>       (m, "InvalidShape",        PyExc_RuntimeError);
    py::register_exception<OutOfMatrix>        (m, "OutOfMatrix",         PyExc_RuntimeError);
    py::register_exception<EmptyMatrix>        (m, "EmptyMatrix",         PyExc_RuntimeError);

    m.attr("DOUBLE_EPSILON") = linalg::DOUBLE_EPSILON;

    /* ---------- Matrix ---------- */
    py::class_<Matrix>(m, "Matrix")
            // конструкторы
            .def(py::init<>())
            .def(py::init<size_t>(), py::arg("rows"),
                 "Матрица rows x 1 (столбец), заполненная по умолчанию")
            .def(py::init<size_t, size_t>(), py::arg("rows"), py::arg("columns"))
            .def(py::init([](const std::vector<std::vector<double>>& rows) { return from_rows(rows); }),
                 py::arg("rows"), "Из списка списков: Matrix([[1, 2], [3, 4]])")
            .def(py::init([](const std::vector<double>& v) { return from_flat(v); }),
                 py::arg("values"), "Из плоского списка (получится столбец)")
            .def("__copy__",     [](const Matrix& a) { return Matrix(a); })
            .def("__deepcopy__", [](const Matrix& a, py::dict) { return Matrix(a); }, py::arg("memo"))

                    // свойства
            .def_property_readonly("rows",     &Matrix::rows)
            .def_property_readonly("columns",  &Matrix::columns)
            .def_property_readonly("size",     &Matrix::size)
            .def_property_readonly("capacity", &Matrix::capacity)
            .def_property_readonly("shape", [](const Matrix& a) {
                return py::make_tuple(a.rows(), a.columns());
            })
            .def("empty", &Matrix::empty)

                    // модификация
            .def("reshape",       &Matrix::reshape, py::arg("rows"), py::arg("columns"))
            .def("reserve",       &Matrix::reserve, py::arg("n"))
            .def("shrink_to_fit", &Matrix::shrink_to_fit)
            .def("clear",         &Matrix::clear)
            .def("swap", [](Matrix& a, Matrix& b) { a.swap(b); }, py::arg("other"))

                    // вычисления
            .def("norm",  &Matrix::norm)
            .def("trace", &Matrix::trace)
            .def("det",   &Matrix::det)
            .def("rank",  &Matrix::rank)
            .def("gauss_forward",  &Matrix::gauss_forward,  py::return_value_policy::reference,
                 "Прямой ход Гаусса (на месте), возвращает self")
            .def("gauss_backward", &Matrix::gauss_backward, py::return_value_policy::reference,
                 "Обратный ход Гаусса (на месте), возвращает self")
            .def("transpose_inplace", [](Matrix& a) -> Matrix& { return a.T(); },
                 py::return_value_policy::reference, "Транспонировать на месте, возвращает self")
            .def_property_readonly("T", [](const Matrix& a) { return transpose(a); },
                                   "Транспонированная копия")

                    // индексация m[i, j]
            .def("__getitem__", [](const Matrix& a, std::pair<py::ssize_t, py::ssize_t> ij) {
                return a(norm_index(ij.first, a.rows(), "строк"),
                         norm_index(ij.second, a.columns(), "столбцов"));
            })
            .def("__setitem__", [](Matrix& a, std::pair<py::ssize_t, py::ssize_t> ij, double v) {
                a(norm_index(ij.first, a.rows(), "строк"),
                  norm_index(ij.second, a.columns(), "столбцов")) = v;
            })
            .def("tolist", [](const Matrix& a) {
                std::vector<std::vector<double>> out(a.rows(), std::vector<double>(a.columns()));
                for (size_t i = 0; i < a.rows(); ++i)
                    for (size_t j = 0; j < a.columns(); ++j)
                        out[i][j] = a(i, j);
                return out;
            })

                    // вывод
            .def("__str__",  [](const Matrix& a) { return to_string(a); })
            .def("__repr__", [](const Matrix& a) { return "Matrix(\n" + to_string(a) + ")"; })

                    // арифметика
            .def(-py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self * py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self == py::self)
            .def(py::self != py::self)
                    // in-place: возвращаем тот же самый python-объект, а не копию
            .def("__iadd__", [](Matrix& a, const Matrix& b) -> Matrix& { return a += b; },
                 py::is_operator(), py::return_value_policy::reference)
            .def("__isub__", [](Matrix& a, const Matrix& b) -> Matrix& { return a -= b; },
                 py::is_operator(), py::return_value_policy::reference)
            .def("__imul__", [](Matrix& a, const Matrix& b) -> Matrix& { return a *= b; },
                 py::is_operator(), py::return_value_policy::reference)
            .def("__imul__", [](Matrix& a, double s) -> Matrix& { return a *= s; },
                 py::is_operator(), py::return_value_policy::reference);

    /* ---------- свободные функции (friend → через лямбды) ---------- */
    m.def("transpose",   [](const Matrix& a) { return transpose(a); }, py::arg("matrix"));
    m.def("invert",      [](const Matrix& a) { return invert(a); },    py::arg("matrix"));
    m.def("power",       [](const Matrix& a, int p) { return power(a, p); },
          py::arg("matrix"), py::arg("power"));
    m.def("solve",       [](const Matrix& a, const Matrix& b) { return solve(a, b); },
          py::arg("matrix"), py::arg("vector"), "Решить A x = b");
    m.def("concatenate", [](const Matrix& a, const Matrix& b) { return concatenate(a, b); },
          py::arg("matrix1"), py::arg("matrix2"));
}