cmake_minimum_required(VERSION 3.4.0)
project(linalg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

add_library(linalg
        source/matrix.cpp
        include/matrix.h
        include/exceptions.h
)

target_include_directories(linalg PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)