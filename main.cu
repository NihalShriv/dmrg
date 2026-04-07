#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

extern double run_dmrg_prof(int N, int D, int max_sweeps, double h, bool verbose);

namespace {
double tfim_exact_small(int N, double Jzz, double h)
{
    int dim = 1 << N;
    std::vector<double> H(dim * dim, 0.0);

    for(int state = 0; state < dim; ++state)
    {
        double diag = 0.0;
        for(int i = 0; i < N - 1; ++i)
        {
            int si = ((state >> i) & 1) ? -1 : 1;
            int sj = ((state >> (i + 1)) & 1) ? -1 : 1;
            diag += Jzz * static_cast<double>(si * sj);
        }
        H[state + dim * state] += diag;

        for(int i = 0; i < N; ++i)
        {
            int flipped = state ^ (1 << i);
            H[flipped + dim * state] += -h;
        }
    }

    for(int sweep = 0; sweep < 200; ++sweep)
    {
        double max_off = 0.0;
        int p = 0;
        int q = 1;
        for(int j = 1; j < dim; ++j)
            for(int i = 0; i < j; ++i)
            {
                double val = std::fabs(H[i + dim * j]);
                if(val > max_off)
                {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        if(max_off < 1e-14)
            break;

        double app = H[p + dim * p];
        double aqq = H[q + dim * q];
        double apq = H[p + dim * q];
        double tau = (aqq - app) / (2.0 * apq);
        double t = (tau >= 0.0) ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
                                : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
        double c = 1.0 / std::sqrt(1.0 + t * t);
        double s = t * c;

        for(int k = 0; k < dim; ++k)
        {
            if(k == p || k == q) continue;
            double aik = H[std::min(k, p) + dim * std::max(k, p)];
            double akq = H[std::min(k, q) + dim * std::max(k, q)];
            double new_kp = c * aik - s * akq;
            double new_kq = s * aik + c * akq;
            H[std::min(k, p) + dim * std::max(k, p)] = new_kp;
            H[std::min(k, q) + dim * std::max(k, q)] = new_kq;
        }

        H[p + dim * p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        H[q + dim * q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        H[p + dim * q] = 0.0;
        H[q + dim * p] = 0.0;
    }

    double min_eval = H[0];
    for(int i = 1; i < dim; ++i)
        min_eval = std::min(min_eval, H[i + dim * i]);
    return min_eval;
}
}

int main()
{
    std::cout << std::setprecision(12);

    const int Ncheck = 6;
    const double hcheck = 1.0;
    double exact = tfim_exact_small(Ncheck, -1.0, hcheck);
    double dmrg_check = run_dmrg_prof(Ncheck, 32, 10, hcheck, false);
    std::cout << "Check: N=" << Ncheck << " h=" << hcheck << " Exact=" << exact << " DMRG=" << dmrg_check << " Delta=" << std::fabs(dmrg_check - exact) << std::endl;

    const int N = 80;
    const double h = 1.0;
    const int sweeps = 12;
    const std::vector<int> bond_dims = {32, 48, 64, 80, 96};

    std::cout << "Model: TFIM  N=" << N << "  h=" << h << std::endl;

    for(int D : bond_dims)
    {
        double E = run_dmrg_prof(N, D, sweeps, h, true);
        std::cout << "Final: N=" << N << " D=" << D << " Energy=" << E << std::endl;
    }

    return 0;
}
