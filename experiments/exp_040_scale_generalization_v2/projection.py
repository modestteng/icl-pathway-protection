"""Project an optimizer's actual parameter update onto at most three halfspaces.

For update ``u`` and loss gradients ``g[j]``, solve

    min_v 0.5 * ||v-u||**2
    subject to g[j] @ v <= rho[j] * max(0, g[j] @ u).

``rho`` is a scalar or one value per constraint in [0, 1]. The experiment's
prespecified grid is 0, .25, .75. Positive rescaling of a gradient does not alter
the solution. Bounds are recomputed from the supplied raw update on every call;
the public operation is idempotent only at rho=0 (or an already feasible input).

The NumPy function is the float64 reference, without any JAX import. The JAX
function works on real floating array leaves with identical pytree structures,
including None nodes, and avoids concatenating parameter-sized vectors. It is
JIT compatible. Neither function takes or changes optimizer state: the caller
must pass the Adam/Optax returned UPDATE, then retain the unprojected optimizer
state as prespecified in the protocol.

All <=8 active sets are enumerated. Normalized Gram matrices are solved with a
spectral pseudoinverse. Candidates must meet primal, dual, and active-equality
conditions; final cast updates are checked again. A failed NumPy solve raises
ProjectionError. A failed JAX solve returns success=False and NaN array leaves;
the caller MUST host-check diagnostics before committing model/optimizer state.
There is no fallback to freezing, an unprojected update, or a weaker constraint.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np


class ProjectionError(RuntimeError):
    """No numerically valid projection; diagnostics preserve the failure."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _relaxation(xp: Any, value: Any, count: int, dtype: Any) -> Any:
    rho = xp.asarray(value, dtype=dtype)
    if rho.ndim == 0:
        return xp.broadcast_to(rho, (count,))
    if rho.shape != (count,):
        raise ValueError("relaxation must be a scalar or one value per gradient")
    return rho


def _active_sets(xp: Any, gram: Any, raw: Any, bound: Any, eps: float) -> dict:
    """Small dual QP; every returned value is NumPy/JAX array compatible."""
    count = gram.shape[0]
    masks = xp.asarray(list(product((0.0, 1.0), repeat=count)), dtype=gram.dtype)
    # Identity on excluded rows keeps every system at the same static shape.
    systems = (
        masks[:, :, None] * gram[None, :, :] * masks[:, None, :]
        + xp.eye(count, dtype=gram.dtype)[None, :, :] * (1.0 - masks[:, :, None])
    )
    systems = (systems + xp.swapaxes(systems, -1, -2)) * 0.5
    rhs = masks * (raw - bound)[None, :]
    eigenvalues, eigenvectors = xp.linalg.eigh(systems)
    rtol = 8.0 * float(xp.finfo(gram.dtype).eps) * max(1, count)
    cutoff = rtol * xp.maximum(xp.max(xp.abs(eigenvalues), axis=-1), 1.0)
    keep = eigenvalues > cutoff[:, None]
    inv = xp.where(keep, 1.0 / xp.where(keep, eigenvalues, 1.0), 0.0)
    # Only the added projector's small linear algebra gets explicit precision.
    # Never change JAX's global or the author's forward/backward matmul setting.
    precision_arguments = {}
    if xp is not np:
        import jax
        precision_arguments = {"precision": jax.lax.Precision.HIGHEST}
    rotated = xp.einsum("kji,kj->ki", eigenvectors, rhs, **precision_arguments)
    dual = xp.einsum("kij,kj->ki", eigenvectors, inv * rotated, **precision_arguments) * masks
    dual_tolerance = 64.0 * eps * (1.0 + xp.max(xp.abs(dual), axis=1))
    dual_valid = xp.all(dual >= -dual_tolerance[:, None], axis=1)
    # Clamp only roundoff-negative multipliers; check all equations after that.
    dual = xp.maximum(dual, 0.0)
    correction_dot = dual @ gram if xp is np else xp.matmul(dual, gram, **precision_arguments)
    residual = raw[None, :] - correction_dot - bound[None, :]
    scale = (
        1.0 + xp.max(xp.abs(raw)) + xp.max(xp.abs(bound))
        + xp.max(xp.abs(correction_dot), axis=1)
    )
    tolerance = 64.0 * eps * scale
    primal_valid = xp.all(residual <= tolerance[:, None], axis=1)
    equality_valid = xp.all(
        xp.abs(residual * masks) <= tolerance[:, None], axis=1
    )
    finite = xp.all(xp.isfinite(dual), axis=1)
    objective = xp.sum(dual * correction_dot, axis=1)
    finite = finite & xp.isfinite(objective)
    # Do not admit the empty set on a truly conflicting input merely because
    # the conflict is smaller than a numerical tolerance.
    empty = xp.sum(masks, axis=1) == 0
    initial_feasible = xp.all(raw <= bound)
    valid = (
        dual_valid & primal_valid & equality_valid & finite
        & (objective >= -tolerance) & (~empty | initial_feasible)
    )
    costs = xp.where(valid, xp.maximum(objective, 0.0), xp.inf)
    index = xp.argmin(costs)
    return {
        "dual": dual[index],
        "active_mask": masks[index] > 0,
        "valid_candidates": xp.sum(valid),
        "initial_feasible": initial_feasible,
        "tolerance": tolerance[index],
        "pinv_rtol": xp.asarray(rtol, dtype=gram.dtype),
    }


def _diagnostics(
    xp: Any, solve: dict, raw: Any, bound: Any, after: Any,
    gradient_scale: Any, gradient_scaled_norm: Any, update_scale: Any,
    raw_norm: Any, new_norm: Any, correction_norm: Any,
    stationarity: Any, input_valid: Any,
) -> dict:
    """Norms/dots are normalized until final presentation, avoiding overflow."""
    dual = solve["dual"]
    residual = after - bound
    violation = xp.max(xp.maximum(residual, 0.0), initial=0.0)
    comp = xp.max(xp.abs(dual * residual), initial=0.0)
    comp_scaled = comp / (1.0 + xp.max(xp.abs(dual), initial=0.0))
    tolerance = solve["tolerance"]
    finite = xp.all(xp.isfinite(after)) & xp.isfinite(stationarity)
    success = (
        input_valid & (solve["valid_candidates"] > 0) & finite
        & (violation <= tolerance) & (comp_scaled <= tolerance)
        & (stationarity <= tolerance * (1.0 + raw_norm + correction_norm))
    )

    def unnormalize(value: Any) -> Any:
        # Multiply by finite factors in this order so an exact zero stays zero
        # even when a true unnormalized diagnostic exceeds dtype range.
        return value * gradient_scale * gradient_scaled_norm * update_scale

    return {
        "success": success,
        "input_valid": input_valid,
        "initial_feasible": solve["initial_feasible"],
        "projected": success & ~solve["initial_feasible"],
        "raw_dot": unnormalize(raw),
        "bound": unnormalize(bound),
        "new_dot": unnormalize(after),
        "normalized_raw_dot": raw,
        "normalized_bound": bound,
        "normalized_new_dot": after,
        "constraint_violation": xp.max(
            unnormalize(xp.maximum(residual, 0.0)), initial=0.0
        ),
        "normalized_constraint_violation": violation,
        "relative_constraint_violation": xp.where(
            raw_norm > 0,
            violation / xp.maximum(raw_norm, xp.finfo(raw.dtype).tiny), 0.0,
        ),
        "feasibility_tolerance": tolerance,
        "active_count": xp.sum(solve["active_mask"]),
        "active_mask": solve["active_mask"],
        "normalized_dual": dual,
        "valid_candidates": solve["valid_candidates"],
        "pinv_rtol": solve["pinv_rtol"],
        "kkt_complementarity": comp,
        "kkt_complementarity_scaled": comp_scaled,
        "kkt_stationarity": stationarity,
        "raw_update_norm": raw_norm * update_scale,
        "new_update_norm": new_norm * update_scale,
        "correction_norm": correction_norm * update_scale,
        "update_norm_ratio": xp.where(
            raw_norm > 0, new_norm / xp.maximum(raw_norm, xp.finfo(raw.dtype).tiny), 1.0
        ),
        "correction_norm_ratio": xp.where(
            raw_norm > 0,
            correction_norm / xp.maximum(raw_norm, xp.finfo(raw.dtype).tiny), 0.0
        ),
        "gradient_norm": gradient_scale * gradient_scaled_norm,
    }


def _empty_diagnostics(xp: Any, norm: Any, dtype: Any, update_scale: Any = 1.0) -> dict:
    zero = xp.asarray(0.0, dtype=dtype)
    empty = xp.zeros((0,), dtype=dtype)
    solve = {
        "dual": empty, "active_mask": xp.zeros((0,), dtype=bool),
        "valid_candidates": xp.asarray(1), "initial_feasible": xp.asarray(True),
        "tolerance": zero, "pinv_rtol": zero,
    }
    return _diagnostics(
        xp, solve, empty, empty, empty, empty, empty,
        xp.asarray(update_scale, dtype=dtype), norm, norm, zero, zero, xp.isfinite(norm),
    )


def project_numpy(
    update_1d: Any, gradient_matrix: Any, relaxation: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    """Float64 reference; a finite already feasible update is preserved exactly.

    Floating output keeps the input update dtype, with post-cast feasibility
    checks. Integer inputs are promoted to float64. Diagnostics include original
    loss-change dots and normalized dots; original-unit diagnostics can overflow
    for extreme finite inputs, whereas the normalized solve remains meaningful.
    """
    original = np.asarray(update_1d)
    gradients = np.asarray(gradient_matrix)
    if original.ndim != 1 or original.size == 0:
        raise ValueError("update_1d must be a nonempty one-dimensional array")
    if gradients.ndim != 2 or gradients.shape[1] != original.size:
        raise ValueError("gradient_matrix must have shape (m, update_size)")
    if gradients.shape[0] > 3:
        raise ValueError("at most three constraints are supported")
    if np.iscomplexobj(original) or np.iscomplexobj(gradients):
        raise ValueError("updates and gradients must be real")
    output_dtype = original.dtype if np.issubdtype(original.dtype, np.floating) else np.float64
    raw = original.astype(np.float64)
    gradients = gradients.astype(np.float64)
    rho = _relaxation(np, relaxation, gradients.shape[0], np.float64)
    if not (np.all(np.isfinite(raw)) and np.all(np.isfinite(gradients))):
        raise ValueError("updates and gradients must be finite")
    rho_input = np.asarray(relaxation, dtype=np.float64)
    if not (np.all(np.isfinite(rho_input)) and np.all((rho_input >= 0) & (rho_input <= 1))):
        raise ValueError("relaxation must be finite and in [0, 1]")
    update_scale = max(float(np.max(np.abs(raw))), np.finfo(np.float64).tiny)
    u = raw / update_scale
    raw_norm = np.linalg.norm(u)
    if not gradients.shape[0]:
        with np.errstate(over="ignore", invalid="ignore"):
            diag = _empty_diagnostics(np, raw_norm, np.dtype("float64"), update_scale)
        return original.astype(output_dtype, copy=True), diag
    gradient_scale = np.max(np.abs(gradients), axis=1)
    scaled = gradients / np.where(gradient_scale > 0, gradient_scale, 1.0)[:, None]
    gradient_scaled_norm = np.linalg.norm(scaled, axis=1)
    normals = scaled / np.where(gradient_scaled_norm > 0, gradient_scaled_norm, 1.0)[:, None]
    gram = normals @ normals.T
    z = normals @ u
    bound = rho * np.maximum(z, 0.0)
    # Solver has double precision; tolerance accounts for output rounding too.
    eps = max(np.finfo(np.float64).eps, np.finfo(output_dtype).eps)
    solve = _active_sets(np, gram, z, bound, eps)
    correction = normals.T @ solve["dual"]
    v = u - correction
    projected = (v * update_scale).astype(output_dtype)
    if bool(solve["initial_feasible"]):
        projected = original.astype(output_dtype, copy=True)
    after_u = projected.astype(np.float64) / update_scale
    after = normals @ after_u
    stationarity = np.linalg.norm(after_u - u + correction)
    with np.errstate(over="ignore", invalid="ignore"):
        diag = _diagnostics(
            np, solve, z, bound, after, gradient_scale, gradient_scaled_norm,
            update_scale, raw_norm, np.linalg.norm(after_u),
            np.linalg.norm(after_u - u), stationarity,
            np.all(np.isfinite(projected)),
        )
    if not bool(diag["success"]):
        raise ProjectionError("no numerically valid closest-update projection", diag)
    return projected, diag


def project_tree(
    raw_update: Any, gradients_tuple: tuple[Any, ...], relaxation: Any
) -> tuple[Any, dict[str, Any]]:
    """JIT-compatible tree projection; import JAX only when this API is called.

    Structures, leaf shapes, and floating dtypes are validated during tracing.
    Relaxation can be a traced scalar. Runtime nonfinite/invalid inputs, or an
    unresolved numerical solve, return ``success=False`` and poison array leaves
    with NaN. The caller must reject that step before committing either state.
    """
    import jax
    import jax.numpy as jnp

    if not isinstance(gradients_tuple, tuple):
        raise TypeError("gradients_tuple must be a tuple of same-structure trees")
    if len(gradients_tuple) > 3:
        raise ValueError("at most three constraints are supported")
    raw_leaves, definition = jax.tree_util.tree_flatten(raw_update)
    gradient_leaves = []
    for gradient in gradients_tuple:
        leaves, gradient_definition = jax.tree_util.tree_flatten(gradient)
        if gradient_definition != definition:
            raise ValueError("gradient and update pytree structures must match (including None)")
        gradient_leaves.append(leaves)
    for index, raw_leaf in enumerate(raw_leaves):
        if not hasattr(raw_leaf, "dtype") or not hasattr(raw_leaf, "shape"):
            raise TypeError("update leaves must be real floating arrays or None nodes")
        if not jnp.issubdtype(raw_leaf.dtype, jnp.floating):
            raise TypeError("update leaves must have real floating dtype")
        for leaves in gradient_leaves:
            leaf = leaves[index]
            if not hasattr(leaf, "dtype") or leaf.shape != raw_leaf.shape:
                raise ValueError("gradient leaf shapes must match update leaf shapes")
            if not jnp.issubdtype(leaf.dtype, jnp.floating):
                raise TypeError("gradient leaves must have real floating dtype")
    dtype = jnp.result_type(
        jnp.float32, *(leaf.dtype for leaf in raw_leaves),
        *(leaf.dtype for leaves in gradient_leaves for leaf in leaves),
    )
    if not raw_leaves:
        # An all-None/empty tree represents a zero-dimensional no-op update.
        diag = _empty_diagnostics(jnp, jnp.asarray(0.0, dtype=dtype), dtype)
        rho = _relaxation(jnp, relaxation, len(gradients_tuple), dtype)
        rho_input = jnp.asarray(relaxation, dtype=dtype)
        valid = jnp.all(jnp.isfinite(rho_input) & (rho_input >= 0) & (rho_input <= 1))
        diag["success"] = valid
        diag["input_valid"] = valid
        return raw_update, diag
    eps = max(float(jnp.finfo(dtype).eps), *(float(jnp.finfo(x.dtype).eps) for x in raw_leaves))

    def dot(left: list, right: list) -> Any:
        terms = [
            jnp.vdot(a.reshape(-1), b.reshape(-1), precision=jax.lax.Precision.HIGHEST)
            for a, b in zip(left, right)
        ]
        return sum(terms, jnp.asarray(0.0, dtype=dtype))

    def max_abs(leaves: list) -> Any:
        return jnp.max(jnp.stack([
            jnp.max(jnp.abs(x), initial=0.0) for x in leaves
        ]))

    raw = [jnp.asarray(leaf, dtype=dtype) for leaf in raw_leaves]
    finite_input = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in raw]))
    update_scale = jnp.maximum(max_abs(raw), jnp.finfo(dtype).tiny)
    u = [leaf / update_scale for leaf in raw]
    raw_norm = jnp.sqrt(jnp.maximum(dot(u, u), 0.0))
    rho = _relaxation(jnp, relaxation, len(gradients_tuple), dtype)
    rho_input = jnp.asarray(relaxation, dtype=dtype)
    finite_input = finite_input & jnp.all(jnp.isfinite(rho_input) & (rho_input >= 0) & (rho_input <= 1))
    if not gradients_tuple:
        diag = _empty_diagnostics(jnp, raw_norm, dtype, update_scale)
        diag["success"] = finite_input
        diag["input_valid"] = finite_input
        output = [jnp.where(finite_input, x, jnp.full_like(x, jnp.nan)) for x in raw_leaves]
        return jax.tree_util.tree_unflatten(definition, output), diag
    normals, scales, norms = [], [], []
    for leaves in gradient_leaves:
        gradient = [jnp.asarray(x, dtype=dtype) for x in leaves]
        finite_input = finite_input & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in gradient]))
        scale = max_abs(gradient)
        scaled = [x / jnp.where(scale > 0, scale, 1.0) for x in gradient]
        norm = jnp.sqrt(jnp.maximum(dot(scaled, scaled), 0.0))
        normal = [x / jnp.where(norm > 0, norm, 1.0) for x in scaled]
        normals.append(normal)
        scales.append(scale)
        norms.append(norm)
    gram = jnp.stack([jnp.stack([dot(a, b) for b in normals]) for a in normals])
    z = jnp.stack([dot(normal, u) for normal in normals])
    bound = rho * jnp.maximum(z, 0.0)
    solve = _active_sets(jnp, gram, z, bound, eps)
    correction = [
        sum((solve["dual"][j] * normals[j][i] for j in range(len(normals))), jnp.zeros_like(u[i]))
        for i in range(len(u))
    ]
    projected = [
        jnp.where(
            solve["initial_feasible"], original,
            ((x - c) * update_scale).astype(original.dtype),
        )
        for original, x, c in zip(raw_leaves, u, correction)
    ]
    after_u = [jnp.asarray(x, dtype=dtype) / update_scale for x in projected]
    after = jnp.stack([dot(normal, after_u) for normal in normals])
    stationarity_tree = [v - x + c for v, x, c in zip(after_u, u, correction)]
    difference = [v - x for v, x in zip(after_u, u)]
    final_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in projected]))
    diag = _diagnostics(
        jnp, solve, z, bound, after, jnp.stack(scales), jnp.stack(norms),
        update_scale, raw_norm, jnp.sqrt(jnp.maximum(dot(after_u, after_u), 0.0)),
        jnp.sqrt(jnp.maximum(dot(difference, difference), 0.0)),
        jnp.sqrt(jnp.maximum(dot(stationarity_tree, stationarity_tree), 0.0)),
        finite_input & final_finite,
    )
    output = [jnp.where(diag["success"], x, jnp.full_like(x, jnp.nan)) for x in projected]
    return jax.tree_util.tree_unflatten(definition, output), diag
