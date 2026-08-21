//
// Created by DeniT on 12-Aug-26.
//

#include "../include/Matrix.h"

Matrix::Matrix(int r, int c)
    : rows(r), cols(c), data (r*c, 0.0f) {}

float& Matrix::operator()(int r, int c){
    return data[r*cols+c];
}

const float& Matrix::operator()(int r, int c) const{
    return data[r*cols+c];
}

void Matrix::print() const{
    for(int r=0; r<rows; r++){
    for (int c=0; c<cols; c++){
        std::cout << (*this)(r,c)<< "\t";
        }
        std::cout<<"\n";
    }
}

Matrix Matrix::multiply_naive(const Matrix& A, const Matrix& B){
    if(A.cols!=B.rows){
       throw std::invalid_argument ( "Invalid Matrix Dimensions");
    }
    Matrix C(A.rows, B.cols);

    for(int i=0; i<A.rows; i++){
        for(int j=0; j<B.cols; j++)
        {
            float sum = 0.0f;
            for(int k=0; k<A.cols;k++)
                sum+= A(i,k) * B(k,j);
            C(i,j) = sum;
        }
    }
    return C;  

}

Matrix Matrix::load_from_bin(const std::string& filename){
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open()){
        throw std::runtime_error("Fail to open binary file");

    }
    uint32_t r,c;
    //reads the header(rows,cols)
    file.read(reinterpret_cast<char*>(&r), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&c), sizeof(uint32_t));

    //reads the payload
    Matrix M(r,c);
    file.read(reinterpret_cast<char*>(M.data.data()), r*c*(sizeof(float)));
    return M;
}

Matrix Matrix::multiply_avx2(const Matrix&A,const Matrix& B){
    if(A.cols != B.rows) throw std::invalid_argument("Incompatible Matrix Dimeansions");
    Matrix C(A.rows, B.cols);
    for(int i=0; i<A.rows; i++)
        for(int k=0; k<A.cols; k++)
        {
            __m256 a_vec = _mm256_set1_ps(A(i,k));
            int j=0; 
            for(;j<=B.cols-8; j+=8)
            {
                __m256 b_vec = _mm256_loadu_ps(&B(k,j));
                __m256 c_vec = _mm256_loadu_ps(&C(i,j));
                c_vec = _mm256_fmadd_ps(a_vec,b_vec,c_vec);
                _mm256_storeu_ps(&C(i,j), c_vec);
            }

            for(;j<B.cols; j++)
                C(i,j) +=A(i,k) * B(k,j); 
        }
        return C;
}
