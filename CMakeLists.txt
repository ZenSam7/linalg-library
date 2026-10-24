cmake_minimum_required(VERSION 3.20.0)
project(linalg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # статическая либа линкуется в .so (на Linux обязательно)

# русские сообщения в исключениях
if(MSVC)
    add_compile_options(/utf-8)
endif()

# рекурсивно добавляем все файлы
file(GLOB sources CONFIGURE_DEPENDS "source/*.cpp")
file(GLOB includes CONFIGURE_DEPENDS "include/*.h")

add_library(linalg
        ${sources}
        ${includes}
)

# для возможности подключать хедеры либы
target_include_directories(linalg PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)



#[[
                для биндинга в питоновскую либу:
]]

# заставляем питон сказать где pybind11 (сначала сделайте pip install pybind11)
# ещё надо с clion в параметры cmake: -DPython_EXECUTABLE=C:/ProgramData/miniconda3/envs/main/python.exe
find_package(Python 3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
)
file(TO_CMAKE_PATH "${pybind11_DIR}" pybind11_DIR)
find_package(pybind11 CONFIG REQUIRED)

# биндим линалг в питоновски-подключаемую либу (?)
pybind11_add_module(linalg_py bindings.cpp)
target_link_libraries(linalg_py PRIVATE linalg)

if(MINGW)  # компилирую через mingw
    # зашиваем libstdc++, libgcc и winpthread прямо в .pyd, чтобы не искать DLL
    target_link_options(linalg_py PRIVATE -static -static-libgcc -static-libstdc++)
endif()

set(PY_OUT_DIR "${CMAKE_SOURCE_DIR}/build/python")
set_target_properties(linalg_py PROPERTIES
        OUTPUT_NAME linalg
        LIBRARY_OUTPUT_DIRECTORY         "${PY_OUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY         "${PY_OUT_DIR}"   # для .pyd на Windows
        LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${PY_OUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PY_OUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${PY_OUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PY_OUT_DIR}"
)

# после каждой сборки: .pyi для PyCharm (нужен pip install pybind11-stubgen в окружении main)
add_custom_command(TARGET linalg_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=$<TARGET_FILE_DIR:linalg_py>"
        "${Python_EXECUTABLE}" -m pybind11_stubgen linalg -o "$<TARGET_FILE_DIR:linalg_py>"
        COMMENT "Генерация linalg.pyi"
        VERBATIM
)

# опционально: копировать в site-packages окружения main
# (Python_SITELIB уже известен после find_package(Python), conda run не нужен)
add_custom_command(TARGET linalg_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "$<TARGET_FILE_DIR:linalg_py>" "${Python_SITELIB}/"
        VERBATIM
)
