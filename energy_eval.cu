#include "energy_eval.h"

double compute_total_energy(
    MPS& psi,
    MPO& H)
{
    int N = psi.N;

    Environment env(N);
    env.build_left(psi, H);

    return env.L[N].item();
}
