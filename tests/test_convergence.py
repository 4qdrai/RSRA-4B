"""
Tests for the Convergence Analysis Module
===========================================

Validates:
- Banach contraction sequences converge monotonically
- Multiple initializations converge to the same fixed point
- Convergence rate matches the theoretical Banach bound
- Monotone operator achieves linear convergence
"""

from __future__ import annotations

import numpy as np
import pytest

from rsra.simulations.convergence_analysis import (
    ConvergenceResult,
    compute_fixed_point,
    create_contraction_operator,
    run_banach_contraction,
    run_monotone_operator,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def banach_result_rho05() -> ConvergenceResult:
    """Run Banach contraction with rho=0.5 for testing."""
    return run_banach_contraction(rho=0.5, dim=32, max_iter=100, n_inits=10)


@pytest.fixture
def banach_result_rho03() -> ConvergenceResult:
    """Run Banach contraction with rho=0.3 for testing."""
    return run_banach_contraction(rho=0.3, dim=32, max_iter=100, n_inits=10)


@pytest.fixture
def banach_result_rho09() -> ConvergenceResult:
    """Run Banach contraction with rho=0.9 for testing."""
    return run_banach_contraction(rho=0.9, dim=32, max_iter=200, n_inits=10)


# ── Test: Contraction Operator Construction ──────────────────────────────────
class TestContractionOperator:
    """Tests for the contraction operator factory."""

    def test_spectral_norm_matches_rho(self) -> None:
        """Verify that the constructed operator has exact spectral norm rho."""
        rng = np.random.default_rng(42)
        for rho in [0.3, 0.5, 0.7, 0.9]:
            A, _ = create_contraction_operator(dim=64, rho=rho, rng=rng)
            actual_norm = np.linalg.svd(A, compute_uv=False).max()
            np.testing.assert_allclose(actual_norm, rho, rtol=1e-10)

    def test_rho_out_of_range_raises(self) -> None:
        """Rho must be in (0, 1)."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="rho must be in"):
            create_contraction_operator(dim=32, rho=1.0, rng=rng)
        with pytest.raises(ValueError, match="rho must be in"):
            create_contraction_operator(dim=32, rho=0.0, rng=rng)
        with pytest.raises(ValueError, match="rho must be in"):
            create_contraction_operator(dim=32, rho=-0.5, rng=rng)

    def test_fixed_point_satisfies_equation(self) -> None:
        """Verify h* = A @ h* + b."""
        rng = np.random.default_rng(42)
        A, b = create_contraction_operator(dim=64, rho=0.5, rng=rng)
        h_star = compute_fixed_point(A, b)
        residual = np.linalg.norm(A @ h_star + b - h_star)
        assert residual < 1e-10, f"Fixed point residual too large: {residual}"


# ── Test: Monotonic Convergence ──────────────────────────────────────────────
class TestMonotonicConvergence:
    """Tests that Banach contraction produces monotonically decreasing errors."""

    def test_errors_decrease_monotonically(
        self, banach_result_rho05: ConvergenceResult
    ) -> None:
        """Each iteration should bring us closer to h* (on average)."""
        mean_errors = banach_result_rho05.errors.mean(axis=0)
        # Check monotonic decrease (allow tiny numerical noise)
        for k in range(1, len(mean_errors)):
            assert mean_errors[k] <= mean_errors[k - 1] + 1e-14, (
                f"Non-monotonic at step {k}: "
                f"{mean_errors[k]} > {mean_errors[k-1]}"
            )

    def test_per_trajectory_monotonic(
        self, banach_result_rho05: ConvergenceResult
    ) -> None:
        """Each individual trajectory should converge monotonically."""
        for i in range(banach_result_rho05.errors.shape[0]):
            traj = banach_result_rho05.errors[i]
            for k in range(1, len(traj)):
                assert traj[k] <= traj[k - 1] + 1e-12, (
                    f"Trajectory {i} non-monotonic at step {k}"
                )

    def test_final_error_is_small(
        self, banach_result_rho05: ConvergenceResult
    ) -> None:
        """After sufficient iterations, error should be negligible."""
        final_errors = banach_result_rho05.errors[:, -1]
        assert np.all(final_errors < 1e-10), (
            f"Final errors too large: max = {final_errors.max():.2e}"
        )


# ── Test: Same Fixed Point from Different Initializations ────────────────────
class TestFixedPointUniqueness:
    """Tests that all initializations converge to the same fixed point."""

    def test_all_inits_converge_to_same_point(
        self, banach_result_rho05: ConvergenceResult
    ) -> None:
        """
        After convergence, the difference between any two trajectories'
        endpoints should be negligible — proving uniqueness of h*.
        """
        # Final errors are all distances to the analytically computed h*
        final_errors = banach_result_rho05.errors[:, -1]
        # All should be very close to zero
        assert final_errors.max() < 1e-10
        # And the spread should be tiny
        assert final_errors.std() < 1e-12

    def test_different_seeds_same_fixed_point(self) -> None:
        """Different random seeds produce different A,b but same convergence."""
        result_seed1 = run_banach_contraction(
            rho=0.5, dim=32, max_iter=100, n_inits=5, seed=42
        )
        result_seed2 = run_banach_contraction(
            rho=0.5, dim=32, max_iter=100, n_inits=5, seed=123
        )
        # Both should converge (final error << 1)
        assert result_seed1.errors.mean(axis=0)[-1] < 1e-10
        assert result_seed2.errors.mean(axis=0)[-1] < 1e-10


# ── Test: Convergence Rate Matches Theoretical Bound ─────────────────────────
class TestConvergenceRate:
    """Tests that empirical convergence rate matches the Banach theorem bound."""

    def test_errors_below_theoretical_bound(
        self, banach_result_rho05: ConvergenceResult
    ) -> None:
        """Mean errors should be at or below the theoretical rho^k bound."""
        mean_errors = banach_result_rho05.errors.mean(axis=0)
        bound = banach_result_rho05.theoretical_bound
        # Allow 20% slack for averaging across initializations,
        # and floor at machine epsilon since fp64 noise dominates
        # once values drop below ~1e-15.
        eps_floor = 1e-14
        for k in range(len(mean_errors)):
            effective_bound = max(bound[k] * 1.2, eps_floor)
            assert mean_errors[k] <= effective_bound + 1e-15, (
                f"Step {k}: mean error {mean_errors[k]:.2e} > "
                f"bound {effective_bound:.2e}"
            )

    def test_faster_rho_converges_faster(self) -> None:
        """Smaller rho should converge in fewer iterations."""
        r03 = run_banach_contraction(
            rho=0.3, dim=32, max_iter=100, n_inits=5
        )
        r07 = run_banach_contraction(
            rho=0.7, dim=32, max_iter=100, n_inits=5
        )
        # At iteration 20, rho=0.3 should have much smaller error
        err_03_at_20 = r03.errors.mean(axis=0)[20]
        err_07_at_20 = r07.errors.mean(axis=0)[20]
        assert err_03_at_20 < err_07_at_20 * 0.1, (
            f"rho=0.3 error ({err_03_at_20:.2e}) not significantly "
            f"smaller than rho=0.7 ({err_07_at_20:.2e}) at step 20"
        )

    def test_iterations_to_eps_ordering(
        self, banach_result_rho05: ConvergenceResult
    ) -> None:
        """Tighter tolerance should require more iterations."""
        eps_iters = banach_result_rho05.iterations_to_eps
        assert eps_iters[1e-4] <= eps_iters[1e-8]
        assert eps_iters[1e-8] <= eps_iters[1e-12]

    def test_theoretical_iteration_count(self) -> None:
        """
        Verify iteration count formula:
        k = ceil(log(eps/||h0-h*||) / log(rho)).
        """
        rho = 0.5
        result = run_banach_contraction(
            rho=rho, dim=32, max_iter=200, n_inits=10
        )
        init_err = result.errors.mean(axis=0)[0]
        for eps, k_computed in result.iterations_to_eps.items():
            if init_err > eps:
                k_expected = int(np.ceil(
                    np.log(eps / init_err) / np.log(rho)
                ))
                assert k_computed == k_expected, (
                    f"For eps={eps}: expected {k_expected}, "
                    f"got {k_computed}"
                )


# ── Test: Monotone Operator Convergence ──────────────────────────────────────
class TestMonotoneOperator:
    """Tests for the monotone operator (forward-backward splitting) method."""

    def test_monotone_converges(self) -> None:
        """Monotone operator should reduce error over iterations."""
        results = run_monotone_operator(
            dim=32, max_iter=100, n_inits=5
        )
        for mono in results:
            mean_errors = mono.errors.mean(axis=0)
            # Error at end should be much smaller than at start
            assert mean_errors[-1] < mean_errors[0] * 0.01, (
                f"Monotone didn't converge sufficiently: "
                f"start={mean_errors[0]:.2e}, end={mean_errors[-1]:.2e}"
            )

    def test_linear_rate_is_subunity(self) -> None:
        """Estimated linear convergence rate should be < 1."""
        results = run_monotone_operator(
            dim=32, max_iter=100, n_inits=5
        )
        for mono in results:
            assert 0 < mono.linear_rate < 1.0, (
                f"Linear rate {mono.linear_rate} not in (0, 1)"
            )

    def test_stronger_convexity_converges_faster(self) -> None:
        """Higher strong-convexity parameter mu → faster convergence."""
        results = run_monotone_operator(
            dim=32, max_iter=100, n_inits=5
        )
        # results[0] = easy (mu=0.1), results[2] = hard (mu=1.0)
        # "Hard" has higher mu → should converge faster
        easy_final = results[0].errors.mean(axis=0)[-1]
        hard_final = results[2].errors.mean(axis=0)[-1]
        assert hard_final <= easy_final + 1e-10, (
            f"Higher mu should converge faster: "
            f"easy={easy_final:.2e}, hard={hard_final:.2e}"
        )


# ── Test: Reproducibility ────────────────────────────────────────────────────
class TestReproducibility:
    """Tests that results are deterministic with fixed seeds."""

    def test_same_seed_same_results(self) -> None:
        """Running with the same seed should produce identical results."""
        r1 = run_banach_contraction(
            rho=0.5, dim=32, max_iter=50, n_inits=5, seed=42
        )
        r2 = run_banach_contraction(
            rho=0.5, dim=32, max_iter=50, n_inits=5, seed=42
        )
        np.testing.assert_array_equal(r1.errors, r2.errors)
        np.testing.assert_array_equal(r1.fixed_point, r2.fixed_point)
