#pragma once
#include "mps.h"
#include "mpo.h"
#include "environment.h"
#include "linalg.h"

enum class SweepDirection {
    LeftToRight,
    RightToLeft
};

struct TwoSiteUpdateStats {
    double residual_norm = 0.0;
    double discarded_weight = 0.0;
    int kept_chi = 0;
};

TwoSiteUpdateStats two_site_update(
    MPS& psi,
    const MPO& H,
    Environment& env,
    LinAlg& la,
    int site,
    SweepDirection direction
);
