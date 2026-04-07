#include "mpo.h"
#include <vector>

MPO::MPO(int N_, int d_, int Wdim_)
    : N(N_), d(d_), Wdim(Wdim_)
{
    W.reserve(N);

    for(int i = 0; i < N; ++i)
    {
        int Wl = (i == 0) ? 1 : Wdim;
        int Wr = (i == N - 1) ? 1 : Wdim;
        W.emplace_back(std::vector<int>{Wl, Wr, d, d});
    }
}

void MPO::initialize_ising(double Jzz, double h)
{
    // OBC TFIM MPO with bond dimension 3 in the notebook's row-to-column convention.
    // The top row propagates the identity channel, so the bulk top-row -hX entries
    // already generate the local field terms along the chain. The last tensor should
    // only close the propagated channels and must not add an extra -hX term.

    for(int site = 0; site < N; ++site)
    {
        int Wl = W[site].shape[0];
        int Wr = W[site].shape[1];
        std::vector<double> hW(Wl * Wr * d * d, 0.0);

        auto z_value = [&](int p) {
            return (p == 0) ? 1.0 : -1.0;
        };

        auto x_value = [&](int p, int q) {
            return (p != q) ? 1.0 : 0.0;
        };

        if(site == 0)
        {
            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(0, 0, p, p, Wl, Wr, d)] = 1.0;

            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(0, 1, p, p, Wl, Wr, d)] = z_value(p);

            for(int p = 0; p < d; ++p)
                for(int q = 0; q < d; ++q)
                    hW[layout::mpo_idx(0, 2, p, q, Wl, Wr, d)] = -h * x_value(p, q);
        }
        else if(site == N - 1)
        {
            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(1, 0, p, p, Wl, Wr, d)] = Jzz * z_value(p);

            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(2, 0, p, p, Wl, Wr, d)] = 1.0;
        }
        else
        {
            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(0, 0, p, p, Wl, Wr, d)] = 1.0;

            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(0, 1, p, p, Wl, Wr, d)] = z_value(p);

            for(int p = 0; p < d; ++p)
                for(int q = 0; q < d; ++q)
                    hW[layout::mpo_idx(0, 2, p, q, Wl, Wr, d)] = -h * x_value(p, q);

            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(1, 2, p, p, Wl, Wr, d)] = Jzz * z_value(p);

            for(int p = 0; p < d; ++p)
                hW[layout::mpo_idx(2, 2, p, p, Wl, Wr, d)] = 1.0;
        }

        W[site].from_host(hW);
    }
}
