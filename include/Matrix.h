//
// Created by DeniT on 12-Aug-26.
//
#ifndef MATRIX_H
#define MATRIX_H

#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <immintrin.h>

class Matrix{
public:
    int rows;
    int cols;
    std::vector<float> data;
    Matrix(int r, int c);
    float& operator()(int r, int c);
    const float& operator()(int r, int c) const;

    void print() const;

    static Matrix multiply_naive(const Matrix& A, const Matrix& B);
    static Matrix multiply_avx2(const Matrix &A, const Matrix &B);
    static Matrix load_from_bin(const std::string& filename);
    static Matrix multiply_parallel(const Matrix& A, const Matrix& B);
    void apply_relu_avx2();

};



#endif //MATRIX_H
