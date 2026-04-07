#include "two_site_split.h"
#include "tensor.h"
#include "heff_builder.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
__global__ void merge_two_site_theta(double* theta, const double* A1, const double* A2, int Dl, int d, int Dm, int Dr)
{
    int dim = Dl * d * d * Dr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= dim) return;

    int a = idx / (d * d * Dr);
    int tmp = idx % (d * d * Dr);
    int p = tmp / (d * Dr);
    tmp %= (d * Dr);
    int q = tmp / Dr;
    int c = tmp % Dr;

    double sum = 0.0;
    for(int m = 0; m < Dm; ++m)
        sum += A1[a + Dl * (p + d * m)] * A2[m + Dm * (q + d * c)];
    theta[idx] = sum;
}

__global__ void ground_to_theta(double* theta, const double* ground, int Dl, int d, int Dr)
{
    int dim = Dl * d * d * Dr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= dim) return;

    int a = idx / (d * d * Dr);
    int tmp = idx % (d * d * Dr);
    int p = tmp / (d * Dr);
    tmp %= (d * Dr);
    int q = tmp / Dr;
    int c = tmp % Dr;

    int left_dim = Dl * d;
    int row = a * d + p;
    int col = q * Dr + c;
    theta[row + left_dim * col] = ground[idx];
}

__global__ void build_left_tensor(double* A1, const double* U, int Dl, int d, int chi, int left_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Dl * d * chi) return;

    int a = idx % Dl;
    int tmp = idx / Dl;
    int p = tmp % d;
    int k = tmp / d;
    int row = a * d + p;
    A1[idx] = U[row + left_dim * k];
}

__global__ void build_left_tensor_with_s(double* A1, const double* U, const double* S, int Dl, int d, int chi, int left_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Dl * d * chi) return;

    int a = idx % Dl;
    int tmp = idx / Dl;
    int p = tmp % d;
    int k = tmp / d;
    int row = a * d + p;
    A1[idx] = U[row + left_dim * k] * S[k];
}

__global__ void build_right_tensor(double* A2, const double* S, const double* V, int chi, int d, int Dr, int right_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= chi * d * Dr) return;

    int k = idx % chi;
    int tmp = idx / chi;
    int q = tmp % d;
    int c = tmp / d;
    int col = q * Dr + c;
    A2[idx] = S[k] * V[col + right_dim * k];
}

__global__ void build_right_tensor_plain(double* A2, const double* V, int chi, int d, int Dr, int right_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= chi * d * Dr) return;

    int k = idx % chi;
    int tmp = idx / chi;
    int q = tmp % d;
    int c = tmp / d;
    int col = q * Dr + c;
    A2[idx] = V[col + right_dim * k];
}

__global__ void copy_vector_to_column(double* matrix, const double* vec, int ld, int col, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    matrix[idx + ld * col] = vec[idx];
}

__global__ void copy_column_to_vector(double* vec, const double* matrix, int ld, int col, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    vec[idx] = matrix[idx + ld * col];
}

__global__ void write_symmetric_column(double* projected, const double* overlap, int stride, int col, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count) return;
    double value = overlap[idx];
    projected[idx + stride * col] = value;
    projected[col + stride * idx] = value;
}

__global__ void copy_leading_block(double* packed, const double* full, int packed_dim, int full_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= packed_dim * packed_dim) return;
    int row = idx % packed_dim;
    int col = idx / packed_dim;
    packed[row + packed_dim * col] = full[row + full_dim * col];
}

__global__ void build_effective_diag(
    double* diag,
    const double* L,
    const double* R,
    const double* W1,
    const double* W2,
    int Dl,
    int d,
    int Dr,
    int W1l,
    int W1r,
    int W2r,
    int Ld0,
    int Ld1,
    int Rd0,
    int Rd1,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= dim) return;

    int a = idx / (d * d * Dr);
    int tmp = idx % (d * d * Dr);
    int p = tmp / (d * Dr);
    tmp %= (d * Dr);
    int q = tmp / Dr;
    int c = tmp % Dr;

    double value = 0.0;
    for(int x = 0; x < W1l; ++x)
    for(int y = 0; y < W1r; ++y)
    for(int z = 0; z < W2r; ++z)
    {
        double Lval = L[layout::env_idx(a, a, x, Ld0, Ld1)];
        double Rval = R[layout::env_idx(c, c, z, Rd0, Rd1)];
        double W1val = W1[layout::mpo_idx(x, y, p, p, W1l, W1r, d)];
        double W2val = W2[layout::mpo_idx(y, z, q, q, W1r, W2r, d)];
        value += Lval * W1val * W2val * Rval;
    }
    diag[idx] = value;
}

__global__ void apply_diagonal_preconditioner(double* out, const double* residual, const double* diag, double eigenvalue, double eps, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;

    double denom = diag[idx] - eigenvalue;
    if(fabs(denom) < eps)
        denom = (denom >= 0.0) ? eps : -eps;
    out[idx] = residual[idx] / denom;
}

__global__ void select_bond_dim_kernel(int* chi_out, const double* S, int max_keep, int rank, double discarded_weight_tol)
{
    if(blockIdx.x != 0 || threadIdx.x != 0) return;
    if(rank == 0)
    {
        *chi_out = 0;
        return;
    }

    int keep = max_keep;
    double tail_weight = 0.0;
    for(int k = rank - 1; k >= 0; --k)
    {
        double sk = S[k];
        tail_weight += sk * sk;
        if(k < max_keep && tail_weight > discarded_weight_tol)
        {
            keep = k + 1;
            break;
        }
    }

    if(keep <= 0)
        keep = 1;
    if(keep > max_keep)
        keep = max_keep;
    *chi_out = keep;
}

struct RitzPair {
    double eigenvalue = 0.0;
    Tensor coeffs;
};

void gemm_tn_device(cublasHandle_t blas, const double* A, const double* B, double* C, int m, int k, int n, int lda, int ldb, int ldc)
{
    double alpha = 1.0;
    double beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

void gemm_nn_device(cublasHandle_t blas, const double* A, const double* B, double* C, int m, int k, int n, int lda, int ldb, int ldc)
{
    double alpha = 1.0;
    double beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

RitzPair solve_projected_problem_device(const Tensor& projected_full, int actual_dim, int max_subspace, LinAlg& la)
{
    RitzPair result;
    result.coeffs = Tensor({actual_dim, 1});

    Tensor packed({actual_dim, actual_dim});
    Tensor eigvals({actual_dim});
    int threads = 256;
    int blocks = (actual_dim * actual_dim + threads - 1) / threads;
    CUDA_KERNEL_CHECK((copy_leading_block<<<blocks, threads>>>(packed.d_data, projected_full.d_data, actual_dim, max_subspace)));

    la.syevd(packed, eigvals, actual_dim);

    double eigenvalue = 0.0;
    CUDA_CHECK(cudaMemcpy(&eigenvalue, eigvals.d_data, sizeof(double), cudaMemcpyDeviceToHost));
    result.eigenvalue = eigenvalue;

    int vec_blocks = (actual_dim + threads - 1) / threads;
    CUDA_KERNEL_CHECK((copy_column_to_vector<<<vec_blocks, threads>>>(result.coeffs.d_data, packed.d_data, actual_dim, 0, actual_dim)));
    return result;
}

double orthonormalize_against_basis_device(Tensor& vec, Tensor& basis_matrix, int dim, int actual_dim, LinAlg& la)
{
    if(actual_dim <= 0)
    {
        double norm = la.nrm2(vec);
        if(norm > 1e-12 && !std::isnan(norm))
            la.scal(vec, 1.0 / norm);
        return norm;
    }

    Tensor overlap({actual_dim, 1});
    Tensor projection({dim, 1});
    for(int pass = 0; pass < 2; ++pass)
    {
        gemm_tn_device(la.blas, basis_matrix.d_data, vec.d_data, overlap.d_data, actual_dim, dim, 1, dim, dim, actual_dim);
        gemm_nn_device(la.blas, basis_matrix.d_data, overlap.d_data, projection.d_data, dim, actual_dim, 1, dim, actual_dim, dim);
        la.axpy(-1.0, projection, vec);
    }

    double norm = la.nrm2(vec);
    if(norm > 1e-12 && !std::isnan(norm))
        la.scal(vec, 1.0 / norm);
    return norm;
}

bool solve_ground_state_davidson(
    Tensor& ground,
    const MPS& psi,
    const MPO& H,
    const Environment& env,
    LinAlg& la,
    int site,
    double truncation_target,
    double* residual_out)
{
    const Tensor& A1 = psi.A[site];
    const Tensor& A2 = psi.A[site + 1];
    const Tensor& W1 = H.W[site];
    const Tensor& W2 = H.W[site + 1];
    const Tensor& L = env.L[site];
    const Tensor& R = env.R[site + 2];

    int Dl = A1.shape[0];
    int d = psi.d;
    int Dm = A1.shape[2];
    int Dr = A2.shape[2];
    int dim = Dl * d * d * Dr;
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;

    Tensor seed({dim});
    CUDA_KERNEL_CHECK((merge_two_site_theta<<<blocks, threads>>>(seed.d_data, A1.d_data, A2.d_data, Dl, d, Dm, Dr)));
    double seed_norm = la.nrm2(seed);
    if(seed_norm < 1e-12 || std::isnan(seed_norm)) return false;
    la.scal(seed, 1.0 / seed_norm);

    int max_subspace_cap = std::min(dim, 160);
    int current_subspace = std::min(max_subspace_cap, std::max(48, std::min(96, dim / 128 + 32)));
    int max_restarts = 8;
    double residual_tol = std::max(1e-13, truncation_target);

    Tensor projected({max_subspace_cap, max_subspace_cap});
    Tensor basis_matrix({dim, max_subspace_cap});
    Tensor image_matrix({dim, max_subspace_cap});
    Tensor diagonal({dim, 1});
    CUDA_KERNEL_CHECK((build_effective_diag<<<blocks, threads>>>(
        diagonal.d_data,
        L.d_data,
        R.d_data,
        W1.d_data,
        W2.d_data,
        Dl,
        d,
        Dr,
        W1.shape[0],
        W1.shape[1],
        W2.shape[1],
        L.shape[0],
        L.shape[1],
        R.shape[0],
        R.shape[1],
        dim)));

    Tensor current({dim});
    current.copy_from(seed);
    double last_residual = std::numeric_limits<double>::infinity();

    for(int restart = 0; restart < max_restarts; ++restart)
    {
        CUDA_CHECK(cudaMemset(projected.d_data, 0, static_cast<size_t>(max_subspace_cap) * max_subspace_cap * sizeof(double)));

        double current_norm = la.nrm2(current);
        if(current_norm < 1e-12 || std::isnan(current_norm))
            return false;
        la.scal(current, 1.0 / current_norm);

        int column_blocks = (dim + threads - 1) / threads;
        CUDA_KERNEL_CHECK((copy_vector_to_column<<<column_blocks, threads>>>(basis_matrix.d_data, current.d_data, dim, 0, dim)));

        bool converged = false;
        int basis_count = 1;
        int max_inner_iters = current_subspace;
        for(int iter = 0; iter < max_inner_iters; ++iter)
        {
            int new_idx = basis_count - 1;
            Tensor basis_vec({dim});
            Tensor image_vec({dim});
            CUDA_KERNEL_CHECK((copy_column_to_vector<<<column_blocks, threads>>>(basis_vec.d_data, basis_matrix.d_data, dim, new_idx, dim)));
            apply_two_site_heff(image_vec, basis_vec, psi, H, env, site);
            CUDA_KERNEL_CHECK((copy_vector_to_column<<<column_blocks, threads>>>(image_matrix.d_data, image_vec.d_data, dim, new_idx, dim)));

            Tensor overlap({basis_count, 1});
            gemm_tn_device(la.blas, basis_matrix.d_data, image_vec.d_data, overlap.d_data, basis_count, dim, 1, dim, dim, basis_count);
            int overlap_blocks = (basis_count + threads - 1) / threads;
            CUDA_KERNEL_CHECK((write_symmetric_column<<<overlap_blocks, threads>>>(projected.d_data, overlap.d_data, max_subspace_cap, new_idx, basis_count)));

            RitzPair pair = solve_projected_problem_device(projected, basis_count, max_subspace_cap, la);

            Tensor ground_vec({dim, 1});
            Tensor hground_vec({dim, 1});
            gemm_nn_device(la.blas, basis_matrix.d_data, pair.coeffs.d_data, ground_vec.d_data, dim, basis_count, 1, dim, basis_count, dim);
            gemm_nn_device(la.blas, image_matrix.d_data, pair.coeffs.d_data, hground_vec.d_data, dim, basis_count, 1, dim, basis_count, dim);

            double norm = la.nrm2(ground_vec);
            if(norm < 1e-12 || std::isnan(norm))
                return false;
            la.scal(ground_vec, 1.0 / norm);
            la.scal(hground_vec, 1.0 / norm);

            ground.copy_from(ground_vec);
            double ground_energy = la.dot(ground_vec, hground_vec);

            Tensor residual({dim, 1});
            residual.copy_from(hground_vec);
            la.axpy(-ground_energy, ground_vec, residual);
            last_residual = la.nrm2(residual);
            if(last_residual <= residual_tol)
            {
                converged = true;
                break;
            }

            if(basis_count >= current_subspace || iter + 1 >= max_inner_iters)
            {
                current.copy_from(ground_vec);
                break;
            }

            Tensor correction({dim, 1});
            CUDA_KERNEL_CHECK((apply_diagonal_preconditioner<<<blocks, threads>>>(
                correction.d_data,
                residual.d_data,
                diagonal.d_data,
                ground_energy,
                1e-8,
                dim)));
            double corr_norm = orthonormalize_against_basis_device(correction, basis_matrix, dim, basis_count, la);
            if(corr_norm < 1e-12 || std::isnan(corr_norm))
            {
                current.copy_from(ground_vec);
                break;
            }

            CUDA_KERNEL_CHECK((copy_vector_to_column<<<column_blocks, threads>>>(basis_matrix.d_data, correction.d_data, dim, basis_count, dim)));
            ++basis_count;
            current.copy_from(ground_vec);
        }

        if(converged || last_residual <= residual_tol)
        {
            if(residual_out) *residual_out = last_residual;
            return true;
        }

        if(last_residual > 10.0 * residual_tol && current_subspace < max_subspace_cap)
            current_subspace = std::min(max_subspace_cap, current_subspace + 16);
    }

    if(residual_out) *residual_out = last_residual;
    return std::isfinite(last_residual);
}
} // namespace

TwoSiteUpdateStats two_site_update(MPS& psi, const MPO& H, Environment& env, LinAlg& la, int site, SweepDirection direction)
{
    TwoSiteUpdateStats stats;

    const Tensor& A1 = psi.A[site];
    const Tensor& A2 = psi.A[site + 1];
    int Dl = A1.shape[0];
    int d = psi.d;
    int Dr = A2.shape[2];
    int dim = Dl * d * d * Dr;
    int left_dim = Dl * d;
    int right_dim = d * Dr;
    int rank = std::min(left_dim, right_dim);
    if(rank <= 0) return stats;

    const double truncation_target = 1e-12;
    Tensor ground({dim});
    if(!solve_ground_state_davidson(ground, psi, H, env, la, site, truncation_target, &stats.residual_norm))
        return stats;

    Tensor Theta({left_dim, right_dim});
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    CUDA_KERNEL_CHECK((ground_to_theta<<<blocks, threads>>>(Theta.d_data, ground.d_data, Dl, d, Dr)));

    double theta_norm = la.nrm2(Theta);
    if(theta_norm < 1e-12 || std::isnan(theta_norm)) return stats;
    la.scal(Theta, 1.0 / theta_norm);

    Tensor U({left_dim, left_dim});
    Tensor S({rank});
    Tensor V({right_dim, right_dim});
    la.svd(Theta, U, S, V, left_dim, right_dim);

    int D = psi.D;
    double cutoff = truncation_target;
    int* chi_managed = nullptr;
    CUDA_CHECK(cudaMallocManaged(&chi_managed, sizeof(int)));
    CUDA_KERNEL_CHECK((select_bond_dim_kernel<<<1, 1>>>(chi_managed, S.d_data, std::min(D, rank), rank, cutoff)));
    CUDA_CHECK(cudaDeviceSynchronize());
    int chi = *chi_managed;
    CUDA_CHECK(cudaFree(chi_managed));
    if(chi <= 0) return stats;
    stats.kept_chi = chi;

    auto singular_values = S.to_host();
    double discarded = 0.0;
    for(int k = chi; k < rank; ++k)
        discarded += singular_values[k] * singular_values[k];
    stats.discarded_weight = discarded;

    Tensor newA1({Dl, d, chi});
    Tensor newA2({chi, d, Dr});
    int total_left = Dl * d * chi;
    int total_right = chi * d * Dr;
    int blocks_left = (total_left + threads - 1) / threads;
    int blocks_right = (total_right + threads - 1) / threads;

    if(direction == SweepDirection::LeftToRight)
    {
        CUDA_KERNEL_CHECK((build_left_tensor<<<blocks_left, threads>>>(newA1.d_data, U.d_data, Dl, d, chi, left_dim)));
        CUDA_KERNEL_CHECK((build_right_tensor<<<blocks_right, threads>>>(newA2.d_data, S.d_data, V.d_data, chi, d, Dr, right_dim)));
    }
    else
    {
        CUDA_KERNEL_CHECK((build_left_tensor_with_s<<<blocks_left, threads>>>(newA1.d_data, U.d_data, S.d_data, Dl, d, chi, left_dim)));
        CUDA_KERNEL_CHECK((build_right_tensor_plain<<<blocks_right, threads>>>(newA2.d_data, V.d_data, chi, d, Dr, right_dim)));
    }

    psi.A[site] = std::move(newA1);
    psi.A[site + 1] = std::move(newA2);
    return stats;
}

