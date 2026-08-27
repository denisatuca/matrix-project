//
// Created by DeniT on 12-Aug-26.
//
#include <iostream>
#include <chrono>
#include <omp.h>
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



   
    int N = 1024;
    Matrix BigA(N, N);
    Matrix BigB(N, N);

    for (int i = 0; i < N * N; ++i) {
        BigA.data[i] = 1.0f;
        BigB.data[i] = 2.0f;
    }


 ;
    try{
        Matrix PyTorchWeights = Matrix::load_from_bin("weights/layer2_weights.bin");
        std::cout << "Loaded weights.bin successfully with dimensions: " << PyTorchWeights.rows << "x" << PyTorchWeights.cols << "\n\n";

    }
    catch(const std::exception& e ){
        std::cerr<<"Warning: "<< e.what() <<"\n\n";

    }
   




  

    std::cout << "\nI. Naive Implementation: \nn";
    auto start = std::chrono::high_resolution_clock::now();
    Matrix BigC = Matrix::multiply_naive(BigA, BigB);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_naive = end - start;
    std::cout << "Naive Execution Time: " << duration_naive.count() << " ms\n";

    std::cout << "\n\nII. AVX2 Implementation: \n";
    auto start_avx = std::chrono::high_resolution_clock::now();
    Matrix BigC_avx = Matrix::multiply_avx2(BigA, BigB);
    auto end_avx = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_avx = end_avx-start_avx;
    std::cout << " AVX2 Time: "<< duration_avx.count() << " ms\n\n";


     std::cout << "\n\nIII. Parallelization implementation: \n";
    std::cout << "Hardware threads available: " << omp_get_max_threads() << "\nMatrix Dimensions: " << N << "x" << N << "\n";
    auto start3 = std::chrono::high_resolution_clock::now();
    C = Matrix::multiply_parallel(BigA, BigB);
    auto end3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_parallel = end3-start3;
    std::cout << "Parallelizatin time: " << duration_parallel.count() << "ms\n";

    std::cout << "Speedup naive/avx: " << duration_naive.count()/duration_avx.count() 
              << "x\nSpeedup avx/parallel: " << duration_avx.count()/duration_parallel.count()
              << "x\nSpeedup naive/parallel: " << duration_naive.count()/duration_parallel.count()<<"x\n";
    
    return 0;
}