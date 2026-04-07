#include "norm_eval.h"

namespace {
__global__ void norm_step_kernel(
    double* next,
    const double* current,
    const double* A,
    int Dl,
    int d,
    int Dr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Dr * Dr)
        return;

    int c = idx % Dr;
    int d2 = idx / Dr;

    double sum = 0.0;
    for(int a=0;a<Dl;a++)
    for(int b=0;b<Dl;b++)
    for(int p=0;p<d;p++)
    {
        sum += current[a + Dl*b]
             * A[a + Dl*(p + d*c)]
             * A[b + Dl*(p + d*d2)];
    }

    next[idx] = sum;
}
} // namespace

double compute_norm(MPS& psi)
{
    Tensor current({1,1});
    current.from_host({1.0});

    for(int site = 0; site < psi.N; site++)
    {
        const Tensor& A = psi.A[site];
        int Dl = A.shape[0];
        int d  = A.shape[1];
        int Dr = A.shape[2];

        Tensor next({Dr,Dr});
        int total = Dr * Dr;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        norm_step_kernel<<<blocks,threads>>>(
            next.d_data,
            current.d_data,
            A.d_data,
            Dl, d, Dr);
        current = std::move(next);
    }

    return current.item();
}