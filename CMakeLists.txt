cmake_minimum_required(VERSION 3.20.0)
project(linalg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # статическая либа линкуется в .so (на Linux обязательно)

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
