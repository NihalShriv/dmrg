#pragma once
#include <vector>
#include "tensor.h"

namespace layout {
// Global tensor layout helpers used by MPO, environments, and Heff apply.
inline __host__ __device__ int mpo_idx(int wl, int wr, int p, int q, int Wl, int Wr, int d)
{
    return wl + Wl * (wr + Wr * (p + d * q));
}

inline __host__ __device__ int env_idx(int left, int right, int mpo, int left_dim, int right_dim)
{
    return left + left_dim * (right + right_dim * mpo);
}

inline __host__ __device__ int mps_idx(int left, int phys, int right, int Dl, int d)
{
    return left + Dl * (phys + d * right);
}
}

class MPO {
public:
    int N;
    int d;
    int Wdim;

    std::vector<Tensor> W; // tensors (Wl, Wr, d, d)

    MPO(int N_, int d_, int Wdim_);

    void initialize_ising(double Jzz, double h);
};