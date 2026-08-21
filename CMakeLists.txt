cmake_minimum_required(VERSION 3.14)
project(Matrix CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mfma")

#O3 for code optimization
# mfma = fused multiply add = does addition and multiplication in sigle clock cicle
include_directories(include)
add_executable(matrix src/main.cpp src/Matrix.cpp)
