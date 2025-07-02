cmake_minimum_required(VERSION 3.31)
project(parquet_golden_files)

set(CMAKE_CXX_STANDARD 20)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

add_executable(parquet_golden_files main.cpp)
target_link_libraries(parquet_golden_files PRIVATE Arrow::arrow_shared Parquet::parquet_shared)
