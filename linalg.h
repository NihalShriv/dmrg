#pragma once
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "tensor.h"

class LinAlg {
public:
    cublasHandle_t blas;
    cusolverDnHandle_t solver;

    LinAlg();
    ~LinAlg();

    void syevd(Tensor& A, Tensor& eigvals, int dim);
    void svd(Tensor& A, Tensor& U, Tensor& S, Tensor& V,
             int m, int n);
    void gemm(const Tensor& A,
              const Tensor& B,
              Tensor& C,
              int m, int k, int n);
    void qr(Tensor& A, Tensor& Q, Tensor& R,
            int m, int n);
    double nrm2(const Tensor& A);
    double dot(const Tensor& A, const Tensor& B);
    void scal(Tensor& A, double alpha);
    void axpy(double alpha, const Tensor& X, Tensor& Y);
    void copy(const Tensor& X, Tensor& Y);
};