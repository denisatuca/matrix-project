//
// Created by DeniT on 12-Aug-26.
//
#include <iostream>
#include <chrono>
#include "../include/Matrix.h"

int main() {

    Matrix A(2, 3);
    Matrix B(3, 2);

    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
    A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;

    B(0, 0) = 7.0f; B(0, 1) = 8.0f;
    B(1, 0) = 9.0f; B(1, 1) = 1.0f;
    B(2, 0) = 2.0f; B(2, 1) = 3.0f;

    std::cout << "Matrix A (2x3):\n";
    A.print();
    std::cout << "\nMatrix B (3x2):\n";
    B.print();

    Matrix C = Matrix::multiply_naive(A,B);
    std::cout << "\nResult Matrix C (2x2):\n";
    C.print();

    std::cout << "\n--- Running Baseline Benchmark (512x512) ---\n";
    int N = 512;
    Matrix BigA(N, N);
    Matrix BigB(N, N);

    for (int i = 0; i < N * N; ++i) {
        BigA.data[i] = 1.0f;
        BigB.data[i] = 2.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();
    Matrix BigC = Matrix::multiply_naive(BigA, BigB);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "512x512 Naive Execution Time: " << duration.count() << " ms\n";
//551.458 ms


    try{
        Matrix PyTorchWeights = Matrix::load_from_bin("weights.bin");
        std::cout << "Loaded weights.bin successfully with dimensions: " << PyTorchWeights.rows << "x" << PyTorchWeights.cols << "\n\n";

    }
    catch(const std::exception& e ){
        std::cerr<<"Warning"<<e.what();

    }
    auto start_avx = std::chrono::high_resolution_clock::now();
    Matrix BigC_avx = Matrix::multiply_avx2(BigA, BigB);
    auto end_avx = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_avx = end_avx-start_avx;

    std::cout << "512x512 AVX2 Time: "<<elapsed_avx.count() << " ms\n";
    double speedup = duration.count() / elapsed_avx.count();
    std::cout<<"\n Speedup: "<< speedup <<"x faster";

    return 0;
}