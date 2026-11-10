"""Small CPU engineering checks, never a model/scientific experiment.

Default: Python test_projection.py (NumPy only, no JAX import).
Optional remote/cloud fidelity: EXP030_TEST_JAX=1 JAX_PLATFORMS=cpu python
test_projection.py. GPU fidelity must be run separately on the scientific host.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np

from projection import ProjectionError, project_numpy, project_tree


def independent_reference(raw, gradients, rho):
    """Independent SVD least-squares enumeration, for small well-scaled tests."""
    raw = np.asarray(raw, dtype=np.float64)
    gradients = np.asarray(gradients, dtype=np.float64)
    bound = rho * np.maximum(gradients @ raw, 0.0)
    candidates = []
    for bits in range(1 << len(gradients)):
        active = [i for i in range(len(gradients)) if bits & (1 << i)]
        if active:
            rows = gradients[active]
            dual = np.linalg.lstsq(rows @ rows.T, rows @ raw - bound[active], rcond=1e-13)[0]
            if np.min(dual) < -1e-9:
                continue
            candidate = raw - rows.T @ dual
            if np.max(np.abs(rows @ candidate - bound[active])) > 1e-8:
                continue
        else:
            candidate = raw.copy()
        if np.max(gradients @ candidate - bound, initial=0.0) <= 1e-8:
            candidates.append(candidate)
    if not candidates:
        raise AssertionError("independent oracle has no candidate")
    return min(candidates, key=lambda x: np.linalg.norm(x - raw))


class NumpyProjectionTests(unittest.TestCase):
    def assert_kkt(self, raw, gradients, rho, projected, diag):
        self.assertTrue(diag["success"])
        self.assertGreaterEqual(diag["valid_candidates"], 1)
        self.assertLessEqual(diag["normalized_constraint_violation"], diag["feasibility_tolerance"])
        self.assertLessEqual(diag["kkt_complementarity_scaled"], diag["feasibility_tolerance"])
        self.assertGreaterEqual(np.min(diag["normalized_dual"], initial=0.0), 0.0)
        np.testing.assert_allclose(diag["raw_dot"], gradients @ raw, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(diag["new_dot"], gradients @ projected, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(diag["bound"], rho * np.maximum(gradients @ raw, 0), rtol=1e-11, atol=1e-11)
        norms = np.linalg.norm(gradients, axis=1)
        normals = gradients / np.where(norms > 0, norms, 1)[:, None]
        scale = max(np.max(np.abs(raw)), np.finfo(np.float64).tiny)
        np.testing.assert_allclose(
            projected - raw + scale * normals.T @ diag["normalized_dual"],
            0.0, atol=5e-10,
        )

    def test_single_conflict_retains_tangential_update(self):
        raw = np.array([2.0, -3.0])
        gradients = np.array([[1.0, 0.0]])
        for rho in (0.0, 0.25, 0.75):
            new, diag = project_numpy(raw, gradients, rho)
            np.testing.assert_allclose(new, [rho * 2.0, -3.0], atol=1e-13)
            self.assert_kkt(raw, gradients, rho, new, diag)

    def test_nonconflict_preserves_input_exactly_and_dtype(self):
        raw = np.array([-1.125, 2.0625, -0.0], dtype=np.float32)
        new, diag = project_numpy(raw, np.array([[1, 0, 0], [0, -1, 0]]), 0.0)
        self.assertEqual(new.dtype, raw.dtype)
        self.assertEqual(new.tobytes(), raw.tobytes())
        self.assertFalse(diag["projected"])

    def test_conflicting_float32_output_is_checked_after_cast(self):
        raw = np.array([0.0073, -0.0018, 0.0027], dtype=np.float32)
        gradients = np.array([[1.0, 0.4, 0.2], [-0.3, 0.8, 0.1]], dtype=np.float32)
        new, diag = project_numpy(raw, gradients, 0.25)
        reference, _ = project_numpy(raw.astype(np.float64), gradients.astype(np.float64), 0.25)
        self.assertEqual(new.dtype, np.float32)
        np.testing.assert_allclose(new, reference, rtol=3e-6, atol=1e-9)
        self.assertTrue(diag["success"])
        self.assertLessEqual(diag["normalized_constraint_violation"], diag["feasibility_tolerance"])

    def test_orthogonal_three_constraints(self):
        raw = np.array([1.0, -2.0, 3.0, 4.0])
        gradients = np.eye(4)[:3]
        new, diag = project_numpy(raw, gradients, 0.25)
        np.testing.assert_allclose(new, [0.25, -2, 0.75, 4], atol=1e-13)
        self.assertEqual(diag["active_count"], 2)
        self.assert_kkt(raw, gradients, 0.25, new, diag)

    def test_nonconflicting_constraint_can_become_active(self):
        raw = np.array([2.0, 1.0])
        gradients = np.array([[1.0, 0.0], [-1.0, 1.0]])
        self.assertLess((gradients @ raw)[1], 0)
        new, diag = project_numpy(raw, gradients, 0.0)
        np.testing.assert_allclose(new, [0, 0], atol=2e-13)
        self.assertEqual(diag["active_count"], 2)
        self.assert_kkt(raw, gradients, 0.0, new, diag)

    def test_zero_dependent_opposite_gradients(self):
        raw = np.array([2.0, -1.0])
        cases = (
            np.array([[0.0, 0.0]]),
            np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]]),
            np.array([[1.0, 0.0], [-1.0, 0.0]]),
            np.array([[1.0, 0.0], [1.0, 0.0], [-2.0, 0.0]]),
        )
        for gradients in cases:
            for rho in (0.0, 0.25, 0.75):
                new, diag = project_numpy(raw, gradients, rho)
                expected = raw if not np.any(gradients) else np.array([2 * rho, -1])
                np.testing.assert_allclose(new, expected, atol=2e-13)
                self.assert_kkt(raw, gradients, rho, new, diag)

    def test_positive_gradient_and_update_rescaling(self):
        raw = np.array([2.0, 1.0, -3.0])
        gradients = np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
        baseline, _ = project_numpy(raw, gradients, 0.25)
        for scales in ([1e-250, 1e250], [1e250, 1e-250], [1e-20, 1e20]):
            new, diag = project_numpy(raw, gradients * np.array(scales)[:, None], 0.25)
            np.testing.assert_allclose(new, baseline, atol=3e-13)
            self.assertTrue(diag["success"])
        for scale in (1e-250, 1e250):
            new, diag = project_numpy(raw * scale, gradients, 0.25)
            np.testing.assert_allclose(new / scale, baseline, atol=3e-13)
            self.assertTrue(diag["success"])

    def test_randomized_feasibility_kkt_oracle_and_strict_idempotence(self):
        rng = np.random.default_rng(30013)  # Engineering vectors, no model data.
        for index in range(120):
            dimension = 1 + index % 7
            count = 1 + index % 3
            raw = rng.normal(size=dimension)
            gradients = rng.normal(size=(count, dimension))
            if index % 10 == 0:
                gradients[-1] = 0
            elif index % 11 == 0 and count > 1:
                gradients[-1] = -2 * gradients[0]
            rho = (0.0, 0.25, 0.75)[index % 3]
            new, diag = project_numpy(raw, gradients, rho)
            self.assert_kkt(raw, gradients, rho, new, diag)
            oracle = independent_reference(raw, gradients, rho)
            np.testing.assert_allclose(new, oracle, atol=2e-10, rtol=2e-10)
            if rho == 0.0:
                again, _ = project_numpy(new, gradients, rho)
                np.testing.assert_allclose(again, new, atol=2e-10, rtol=2e-10)

    def test_relaxed_recomputed_bound_is_not_idempotent(self):
        first, _ = project_numpy([1.0], [[1.0]], 0.25)
        second, _ = project_numpy(first, [[1.0]], 0.25)
        np.testing.assert_allclose(first, [0.25])
        np.testing.assert_allclose(second, [0.0625])

    def test_zero_update_and_no_constraints(self):
        new, diag = project_numpy(np.zeros(4), np.eye(4)[:3], 0.0)
        np.testing.assert_array_equal(new, np.zeros(4))
        self.assertTrue(diag["success"])
        self.assertEqual(diag["update_norm_ratio"], 1)
        raw = np.array([-0.0, 7.0])
        new, diag = project_numpy(raw, np.empty((0, 2)), 0.0)
        self.assertEqual(new.tobytes(), raw.tobytes())
        self.assertTrue(diag["success"])

    def test_vector_relaxation(self):
        new, diag = project_numpy([2.0, 4.0], np.eye(2), [0.25, 0.75])
        np.testing.assert_allclose(new, [0.5, 3])
        self.assertTrue(diag["success"])

    def test_invalid_inputs_fail_visibly(self):
        cases = (
            ([], np.empty((1, 0)), 0),
            ([1.0], [[1.0], [1.0], [1.0], [1.0]], 0),
            ([np.nan], [[1.0]], 0),
            ([1.0], [[np.inf]], 0),
            ([1.0], [[1.0]], -0.1),
            ([1.0], [[1.0]], np.nan),
            ([1.0], [[1.0]], 1.1),
            ([1.0], [[1.0]], [0.0, 0.25]),
            ([1j], [[1.0]], 0),
            ([1.0], np.empty((0, 1)), np.nan),
        )
        for raw, gradients, rho in cases:
            with self.assertRaises((ValueError, ProjectionError)):
                project_numpy(raw, gradients, rho)

    def test_unresolved_nearly_opposite_constraints_do_not_silently_freeze(self):
        # The small angle is below resolvable Gram rank. Neither dropping the
        # second constraint nor returning zero is an authorized fallback.
        with self.assertRaises(ProjectionError) as caught:
            project_numpy([1.0, 1.0], [[1.0, 0.0], [-1.0, 1e-8]], 0.0)
        self.assertFalse(caught.exception.diagnostics["success"])
        self.assertEqual(caught.exception.diagnostics["valid_candidates"], 0)

    def test_numpy_import_does_not_load_jax(self):
        if os.environ.get("EXP030_TEST_JAX") != "1":
            self.assertNotIn("jax", sys.modules)


@unittest.skipUnless(os.environ.get("EXP030_TEST_JAX") == "1", "explicit optional JAX engineering check")
class JaxProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("jax") is None:
            raise unittest.SkipTest("JAX unavailable; no dependency installation")
        import jax
        import jax.numpy as jnp
        cls.jax, cls.jnp = jax, jnp

    def test_jitted_tree_matches_reference_and_preserves_none(self):
        jax, jnp = self.jax, self.jnp
        update = {"a": jnp.array([2.0, 1.0]), "b": (None, jnp.array([-3.0]))}
        first = {"a": jnp.array([1.0, 0.0]), "b": (None, jnp.array([0.0]))}
        second = {"a": jnp.array([-1.0, 1.0]), "b": (None, jnp.array([0.0]))}
        compiled = jax.jit(project_tree)
        for rho in (0.0, 0.25, 0.75):
            new, diag = compiled(update, (first, second), jnp.asarray(rho))
            expected, _ = project_numpy([2, 1, -3], [[1, 0, 0], [-1, 1, 0]], rho)
            np.testing.assert_allclose(np.concatenate([np.asarray(new["a"]), np.asarray(new["b"][1])]), expected, atol=3e-5)
            self.assertIsNone(new["b"][0])
            self.assertTrue(bool(diag["success"]))

    def test_jitted_feasible_input_preserves_bits(self):
        raw = {"x": self.jnp.array([-1.125, 2.0625, -0.0]), "none": None}
        gradient = {"x": self.jnp.array([1.0, -1.0, 0.0]), "none": None}
        new, diag = self.jax.jit(project_tree)(raw, (gradient,), 0.0)
        self.assertEqual(np.asarray(new["x"]).tobytes(), np.asarray(raw["x"]).tobytes())
        self.assertTrue(bool(diag["success"]))

    def test_jitted_rank_degenerate_and_invalid_runtime(self):
        raw = self.jnp.array([2.0, -1.0])
        gradients = (self.jnp.array([1.0, 0.0]), self.jnp.array([-1.0, 0.0]), self.jnp.zeros(2))
        compiled = self.jax.jit(project_tree)
        new, diag = compiled(raw, gradients, 0.0)
        np.testing.assert_allclose(new, [0.0, -1.0], atol=3e-5)
        self.assertTrue(bool(diag["success"]))
        bad, bad_diag = compiled(raw, gradients, self.jnp.asarray(-0.25))
        self.assertFalse(bool(bad_diag["success"]))
        self.assertTrue(np.all(np.isnan(np.asarray(bad))))

    def test_tree_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            project_tree({"x": self.jnp.ones(2), "n": None}, ({"x": self.jnp.ones(2)},), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
