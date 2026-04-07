#pragma once
#include "mps.h"
#include "mpo.h"
#include "environment.h"

void apply_two_site_heff(
    Tensor& theta_out,
    const Tensor& theta_in,
    const MPS& psi,
    const MPO& H,
    const Environment& env,
    int site
);

bool validate_two_site_heff_apply(
    const MPS& psi,
    const MPO& H,
    const Environment& env,
    int site,
    double tolerance
);