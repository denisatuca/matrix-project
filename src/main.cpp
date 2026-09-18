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
    

    try{
        std::cout << "Loading binary weights\n";
        Matrix fc1_w = Matrix::load_from_bin("fc1_w.bin");
        Matrix fc1_b = Matrix::load_from_bin("fc1_b.bin");
        Matrix fc2_w = Matrix::load_from_bin("fc2_w.bin");
        Matrix fc2_b = Matrix::load_from_bin("fc2_b.bin");
        Matrix fc3_w = Matrix::load_from_bin("fc3_w.bin");
        Matrix fc3_b = Matrix::load_from_bin("fc3_b.bin");

        Matrix input_image = Matrix::load_from_bin("sample_image.bin");
        std::cout << "Binary files loaded\n";

        auto start_time = std::chrono::high_resolution_clock::now();

        Matrix fc1_w_T(fc1_w.cols, fc1_w.rows);
        for(int r=0; r<fc1_w.rows; r++)
            for(int c=0; c<fc1_w.cols; c++)
            fc1_w_T(c,r) = fc1_w(r,c);

        Matrix layer1_out = Matrix::multiply_parallel(input_image, fc1_w_T);
        for(int i=0; i<layer1_out.cols; i++) layer1_out.data[i] +=fc1_b.data[i];
        layer1_out.apply_relu_avx2();

        Matrix fc2_w_T(fc2_w.cols, fc2_w.rows);
        for(int r=0; r<fc2_w.rows; ++r)
            for(int c=0; c<fc2_w.cols; ++c)
                fc2_w_T(c, r) = fc2_w(r, c);

        Matrix layer2_out = Matrix::multiply_parallel(layer1_out, fc2_w_T);
        for(int i=0; i<layer2_out.cols; ++i) layer2_out.data[i] += fc2_b.data[i];
        layer2_out.apply_relu_avx2();

        Matrix fc3_w_T(fc3_w.cols, fc3_w.rows);
            for(int r=0; r<fc3_w.rows; ++r)
                for(int c=0; c<fc3_w.cols; ++c)
                    fc3_w_T(c, r) = fc3_w(r, c);

        Matrix logits = Matrix::multiply_parallel(layer2_out, fc3_w_T);
        for(int i=0; i<logits.cols; i++) logits.data[i] +=fc3_b.data[i];

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration <double, std::milli> latency = end_time - start_time;
        int predicted_digit = 0;
        float max_logit = logits.data[0];
        for(int i=1; i<10; i++){
            if(logits.data[i] >max_logit)
            {
                max_logit = logits.data[i];
                predicted_digit = i;
            }
        }

        std::cout << "Predicted digit: " << predicted_digit << "\nInference latency: " << latency.count() << "ms \n\n";
    }

    catch(const std::exception& e){
        std::cerr << "Error: " << e.what() << "\n";
    }
    return 0;
}