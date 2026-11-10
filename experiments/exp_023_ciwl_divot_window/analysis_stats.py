"""Paired query-class cluster inference for the frozen exp023/024 protocol.

The estimand remains a mean over evaluation rows; cluster sizes are not
silently reweighted. Checkpoints must be averaged within each row before
these functions are called. This module intentionally does not import the
standard-library ``statistics`` module, and needs only NumPy and math.
"""

import math

import numpy as np


def normal_quantile(p):
    """Inverse normal CDF by deterministic bisection (no SciPy dependency)."""
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError("A normal quantile requires 0 < p < 1")
    lo, hi = -12.0, 12.0
    for _ in range(80):
        middle = (lo + hi) / 2.0
        cdf = 0.5 * math.erfc(-middle / math.sqrt(2.0))
        if cdf < p:
            lo = middle
        else:
            hi = middle
    return (lo + hi) / 2.0


def cluster_stats(values, classes, structural_identity=False):
    """Row mean and finite-G corrected query-class cluster standard error."""
    values = np.asarray(values, dtype=np.float64)
    classes = np.asarray(classes)
    if values.ndim != 1 or classes.shape != values.shape or len(values) == 0:
        raise ValueError("Expected nonempty paired row values and query classes")
    if not np.isfinite(values).all() or not np.isfinite(classes).all():
        raise ValueError("Nonfinite values or classes")
    unique, inverse = np.unique(classes, return_inverse=True)
    n, groups = len(values), len(unique)
    mean = float(values.mean())
    counts = np.bincount(inverse, minlength=groups)
    sums = np.bincount(inverse, weights=values, minlength=groups)
    class_means = sums / counts
    scores = np.bincount(inverse, weights=values - mean, minlength=groups)
    variance = (float(groups / (groups - 1) * np.dot(scores, scores) / n**2)
                if groups > 1 else None)
    se = math.sqrt(max(0.0, variance)) if variance is not None else None
    if structural_identity and not np.all(values == 0.0):
        raise ValueError("A claimed structural identity has nonzero row differences")
    valid = bool(not structural_identity and groups > 1 and se > 1e-15)
    z = normal_quantile(0.975)
    result = {
        "mean": mean, "se": se, "n": n, "clusters": groups,
        "cluster_unit": "true query class", "estimand": "evaluation-row mean",
        "valid_normal_approximation": valid,
        "structural_identity": bool(structural_identity),
        "ci95_lower": mean - z * se if valid else (0.0 if structural_identity else None),
        "ci95_upper": mean + z * se if valid else (0.0 if structural_identity else None),
        "row_sd": float(values.std(ddof=1)) if n > 1 else 0.0,
        "row_positive_fraction": float((values > 0).mean()),
        "row_negative_fraction": float((values < 0).mean()),
        "class_mean_distribution": {
            "minimum": float(class_means.min()),
            "median": float(np.median(class_means)),
            "maximum": float(class_means.max()),
            "positive_fraction": float((class_means > 0).mean()),
            "negative_fraction": float((class_means < 0).mean()),
            "note": "Unweighted class descriptions, not the inferential estimand",
        },
        "status": ("structural_zero_no_population_test" if structural_identity else
                   "normal_cluster_approximation" if valid else
                   "degenerate_cluster_se_conservative_test"),
    }
    return result


def one_sided(values, classes, threshold=0.0, name=None, alpha=0.05):
    """Test H0: E[values] <= threshold, using paired clustered differences."""
    result = cluster_stats(values, classes)
    threshold, alpha = float(threshold), float(alpha)
    valid = result["valid_normal_approximation"]
    z = ((result["mean"] - threshold) / result["se"] if valid else None)
    p = 0.5 * math.erfc(z / math.sqrt(2.0)) if valid else 1.0
    result.update({
        "name": name, "threshold": threshold, "alternative": "mean > threshold",
        "null": "mean <= threshold", "z_vs_threshold": z,
        "p_one_sided": float(p), "p_underflow": bool(valid and p == 0.0),
        "unadjusted_one_sided_lower": (
            result["mean"] - normal_quantile(1.0 - alpha) * result["se"]
            if valid else None),
        "pass_unadjusted": bool(valid and p <= alpha and result["mean"] > threshold),
    })
    return result


def adjusted_family(tests, alpha=0.05, method="holm"):
    """Multiplicity-correct p values and Bonferroni simultaneous lower bounds."""
    if method not in ("holm", "bonferroni") or not tests:
        raise ValueError("Expected a nonempty Holm or Bonferroni family")
    tests = [dict(test) for test in tests]
    m, alpha = len(tests), float(alpha)
    adjusted = [None] * m
    if method == "holm":
        running = 0.0
        for rank, index in enumerate(sorted(range(m), key=lambda i: (tests[i]["p_one_sided"], i))):
            running = max(running, min(1.0, (m - rank) * tests[index]["p_one_sided"]))
            adjusted[index] = running
    else:
        adjusted = [min(1.0, m * test["p_one_sided"]) for test in tests]
    q = normal_quantile(1.0 - alpha / m)
    for test, p in zip(tests, adjusted):
        valid = test["valid_normal_approximation"]
        lower = test["mean"] - q * test["se"] if valid else None
        test.update({
            "family_size": m, "correction": method,
            "p_adjusted": float(p), "p_holm": float(p) if method == "holm" else None,
            "bonferroni_one_sided_lower": lower,
            "lower_bonferroni": lower,
            "bonferroni_lower_confidence": 1.0 - alpha / m,
            "pass_adjusted": bool(valid and p <= alpha and test["mean"] > test["threshold"]),
            "pass_bonferroni_lower": bool(lower is not None and lower > test["threshold"]),
        })
        if test["pass_adjusted"]:
            interpretation = "supports_prespecified_threshold"
        elif not valid:
            interpretation = "degenerate_uncertainty_no_confirmatory_support"
        elif test["ci95_upper"] < 0:
            interpretation = "opposite_direction_in_ordinary_95ci"
        elif test["ci95_upper"] < test["threshold"]:
            interpretation = "effect_smaller_than_threshold_in_ordinary_95ci"
        else:
            interpretation = "threshold_not_confirmed_precision_or_small_effect"
        test["interpretation"] = interpretation
    return {
        "alpha": alpha, "method": method, "family_size": m, "tests": tests,
        "all_pass": bool(all(test["pass_adjusted"] for test in tests)),
        "all_bonferroni_bounds_pass": bool(all(test["pass_bonferroni_lower"] for test in tests)),
        "interval_note": "ci95 is ordinary; simultaneous one-sided Bonferroni lower bounds are separate",
    }


def equivalence(values, classes, margin, name=None, alpha=0.05):
    """Prespecified diagnostic TOST; 90% CI corresponds to alpha=.05."""
    result = cluster_stats(values, classes)
    margin, alpha = float(margin), float(alpha)
    if margin <= 0:
        raise ValueError("Equivalence margin must be positive")
    valid = result["valid_normal_approximation"]
    if valid:
        z_lower = (result["mean"] + margin) / result["se"]
        z_upper = (result["mean"] - margin) / result["se"]
        p_lower = 0.5 * math.erfc(z_lower / math.sqrt(2.0))
        p_upper = 0.5 * math.erfc(-z_upper / math.sqrt(2.0))
        q = normal_quantile(1.0 - alpha)
        lower, upper = result["mean"] - q * result["se"], result["mean"] + q * result["se"]
    else:
        p_lower = p_upper = 1.0
        lower = upper = None
    result.update({
        "name": name, "equivalence_margin": margin,
        "ci90_lower": lower, "ci90_upper": upper,
        "p_tost_lower": float(p_lower), "p_tost_upper": float(p_upper),
        "p_tost": float(max(p_lower, p_upper)),
        "equivalent": bool(valid and p_lower < alpha and p_upper < alpha
                           and lower > -margin and upper < margin),
        "multiplicity": "prespecified secondary diagnostic; no family correction",
        "interpretation": "Failure to establish equivalence does not demonstrate a nonzero effect",
    })
    return result
