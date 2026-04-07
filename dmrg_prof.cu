#include "mps.h"
#include "mpo.h"
#include "environment.h"
#include "heff_builder.h"
#include "canonical.h"
#include "two_site_split.h"
#include "energy_eval.h"
#include "linalg.h"
#include "norm_eval.h"
#include <iostream>
#include <cmath>
#include <limits>

double run_dmrg_prof(
    int N,
    int D,
    int max_sweeps,
    double h,
    bool verbose)
{
    int d = 2;

    std::srand(1234 + D + 17 * N);

    LinAlg la;
    MPS psi(N, d, D);
    psi.random_initialize();

    MPO H(N, d, 3);
    H.initialize_ising(-1.0, h);

    left_canonicalize(psi, la);

    double prevE = std::numeric_limits<double>::infinity();

    for(int sweep = 0; sweep < max_sweeps; ++sweep)
    {
        double max_residual = 0.0;
        double discarded_sum = 0.0;
        int update_count = 0;

        Environment env_lr(N);
        env_lr.build_right(psi, H);
        env_lr.L[0] = Tensor({1,1,1});
        env_lr.L[0].from_host({1.0});

        for(int site = 0; site < N - 1; ++site)
        {
            if(N <= 6 && sweep == 0)
            {
                if(!env_lr.is_well_conditioned(1e6))
                {
                    std::cout << "Environment instability detected before site " << site << std::endl;
                    return std::numeric_limits<double>::quiet_NaN();
                }
                if(!validate_two_site_heff_apply(psi, H, env_lr, site, 1e-9))
                {
                    std::cout << "Two-site operator validation failed on site " << site << std::endl;
                    return std::numeric_limits<double>::quiet_NaN();
                }
            }

            TwoSiteUpdateStats stats = two_site_update(psi, H, env_lr, la, site, SweepDirection::LeftToRight);
            max_residual = std::max(max_residual, stats.residual_norm);
            discarded_sum += stats.discarded_weight;
            ++update_count;
            env_lr.update_left_site(psi, H, site);

            if(!env_lr.is_well_conditioned(1e12))
            {
                std::cout << "Environment blew up after left-to-right site " << site << std::endl;
                return std::numeric_limits<double>::quiet_NaN();
            }
        }

        Environment env_rl(N);
        env_rl.build_left(psi, H);
        env_rl.R[N] = Tensor({1,1,1});
        env_rl.R[N].from_host({1.0});

        for(int site = N - 2; site >= 0; --site)
        {
            TwoSiteUpdateStats stats = two_site_update(psi, H, env_rl, la, site, SweepDirection::RightToLeft);
            max_residual = std::max(max_residual, stats.residual_norm);
            discarded_sum += stats.discarded_weight;
            ++update_count;
            env_rl.update_right_site(psi, H, site + 1);

            if(!env_rl.is_well_conditioned(1e12))
            {
                std::cout << "Environment blew up after right-to-left site " << site << std::endl;
                return std::numeric_limits<double>::quiet_NaN();
            }
        }

        left_canonicalize(psi, la);
        double E = compute_total_energy(psi, H);
        if(std::isnan(E) || !std::isfinite(E))
        {
            std::cout << "Encountered invalid energy; stopping sweep." << std::endl;
            return E;
        }

        double avg_discarded = (update_count > 0) ? discarded_sum / static_cast<double>(update_count) : 0.0;
        if(verbose)
            std::cout << "N=" << N << " D=" << D << " Sweep=" << (sweep + 1) << " Energy=" << E << std::endl;

        double energy_tol = 1e-10;
        double residual_tol = 1e-10;
        double discarded_tol = 1e-10;
        if(sweep >= 1 && std::fabs(E - prevE) < energy_tol && max_residual < residual_tol && avg_discarded < discarded_tol)
            return E;

        prevE = E;
    }

    left_canonicalize(psi, la);
    return compute_total_energy(psi, H);
}

