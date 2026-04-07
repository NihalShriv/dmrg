#include "linalg.h"
#include <algorithm>
#include <iostream>

namespace {
__global__ void extract_upper_triangle(
    double* R,
    const double* A,
    int m,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n * n)
        return;

    int row = idx % n;
    int col = idx / n;

    if(row <= col && row < m)
        R[idx] = A[row + m * col];
    else
        R[idx] = 0.0;
}

void check_dev_info(int* d_info, const char* op)
{
    int info = 0;
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if(info != 0)
        std::cerr << op << " failed with devInfo=" << info << std::endl;
}
} // namespace

LinAlg::LinAlg()
{
    cublasCreate(&blas);
    cusolverDnCreate(&solver);
}

LinAlg::~LinAlg()
{
    cublasDestroy(blas);
    cusolverDnDestroy(solver);
}

void LinAlg::gemm(const Tensor& A,
                  const Tensor& B,
                  Tensor& C,
                  int m, int k, int n)
{
    double alpha = 1.0;
    double beta  = 0.0;

    cublasDgemm(
        blas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,n,k,
        &alpha,
        A.d_data,m,
        B.d_data,k,
        &beta,
        C.d_data,m);
}

void LinAlg::qr(Tensor& A,
                Tensor& Q,
                Tensor& R,
                int m, int n)
{
    int lwork=0;

    cusolverDnDgeqrf_bufferSize(
        solver,m,n,
        A.d_data,m,
        &lwork);

    Tensor work({lwork});
    Tensor tau({std::min(m,n)});
    int* devInfo;
    cudaMalloc(&devInfo,sizeof(int));

    cusolverDnDgeqrf(
        solver,m,n,
        A.d_data,m,
        tau.d_data,
        work.d_data,lwork,
        devInfo);

    int threads = 256;
    int blocks = (n * n + threads - 1) / threads;
    extract_upper_triangle<<<blocks,threads>>>(R.d_data, A.d_data, m, n);

    cusolverDnDorgqr(
        solver,m,n,
        std::min(m,n),
        A.d_data,m,
        tau.d_data,
        work.d_data,lwork,
        devInfo);

    check_dev_info(devInfo, "qr");

    Q = std::move(A);

    cudaFree(devInfo);
}

double LinAlg::nrm2(const Tensor& A)
{
    double result = 0.0;
    cublasDnrm2(blas, A.size, A.d_data, 1, &result);
    return result;
}

double LinAlg::dot(const Tensor& A, const Tensor& B)
{
    double result = 0.0;
    cublasDdot(blas, A.size, A.d_data, 1, B.d_data, 1, &result);
    return result;
}

void LinAlg::scal(Tensor& A, double alpha)
{
    cublasDscal(blas, A.size, &alpha, A.d_data, 1);
}

void LinAlg::axpy(double alpha, const Tensor& X, Tensor& Y)
{
    cublasDaxpy(blas, X.size, &alpha, X.d_data, 1, Y.d_data, 1);
}

void LinAlg::copy(const Tensor& X, Tensor& Y)
{
    cublasDcopy(blas, X.size, X.d_data, 1, Y.d_data, 1);
}

void LinAlg::syevd(Tensor& A,
                   Tensor& eigvals,
                   int dim)
{
    int lwork=0;

    cusolverDnDsyevd_bufferSize(
        solver,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        dim,
        A.d_data,
        dim,
        eigvals.d_data,
        &lwork);

    Tensor work({lwork});
    int* devInfo;
    cudaMalloc(&devInfo,sizeof(int));

    cusolverDnDsyevd(
        solver,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        dim,
        A.d_data,
        dim,
        eigvals.d_data,
        work.d_data,
        lwork,
        devInfo);

    check_dev_info(devInfo, "syevd");
    cudaFree(devInfo);
}

void LinAlg::svd(Tensor& A,
                 Tensor& U,
                 Tensor& S,
                 Tensor& V,
                 int m, int n)
{
    gesvdjInfo_t params;
    cusolverDnCreateGesvdjInfo(&params);

    int lwork=0;

    cusolverDnDgesvdj_bufferSize(
        solver,
        CUSOLVER_EIG_MODE_VECTOR,
        0,
        m,n,
        A.d_data,m,
        S.d_data,
        U.d_data,m,
        V.d_data,n,
        &lwork,
        params);

    Tensor work({lwork});
    int* devInfo;
    cudaMalloc(&devInfo,sizeof(int));

    cusolverDnDgesvdj(
        solver,
        CUSOLVER_EIG_MODE_VECTOR,
        0,
        m,n,
        A.d_data,m,
        S.d_data,
        U.d_data,m,
        V.d_data,n,
        work.d_data,
        lwork,
        devInfo,
        params);

    check_dev_info(devInfo, "svd");
    cusolverDnDestroyGesvdjInfo(params);
    cudaFree(devInfo);
}