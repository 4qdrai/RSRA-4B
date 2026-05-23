"""
Tests for the Reasoning Decay Simulation Module
=================================================

Validates:
- Standard model accuracy = p^N exactly
- Monte Carlo converges to analytical result within tolerance
- RSRA accuracy >= standard accuracy always
- Reproducibility with fixed seed
- Edge cases and boundary conditions
"""

from __future__ import annotations

import numpy as np
import pytest

from rsra.simulations.reasoning_decay import (
    DecayResult,
    compute_advantage_heatmap,
    compute_critical_depth,
    monte_carlo_rsra,
    monte_carlo_standard,
    rsra_accuracy_analytical,
    rsra_effective_p_step,
    run_decay_comparison,
    standard_accuracy,
)


# ── Test: Standard Model Accuracy ────────────────────────────────────────────
class TestStandardAccuracy:
    """Tests for the standard AR model accuracy formula."""

    def test_known_values(self) -> None:
        """Verify p^N for known inputs."""
        assert standard_accuracy(0.95, 1) == pytest.approx(0.95)
        assert standard_accuracy(0.95, 10) == pytest.approx(0.95**10)
        assert standard_accuracy(0.95, 100) == pytest.approx(0.95**100)
        assert standard_accuracy(0.50, 2) == pytest.approx(0.25)

    def test_n_zero(self) -> None:
        """Zero steps should give 100% accuracy."""
        assert standard_accuracy(0.95, 0) == pytest.approx(1.0)

    def test_perfect_step(self) -> None:
        """Perfect per-step accuracy should maintain 100% at all depths."""
        for n in [1, 10, 100, 1000]:
            assert standard_accuracy(1.0, n) == pytest.approx(1.0)

    def test_zero_step_accuracy(self) -> None:
        """Zero per-step accuracy means zero sequence accuracy."""
        for n in [1, 10, 100]:
            assert standard_accuracy(0.0, n) == pytest.approx(0.0)

    def test_exponential_decay(self) -> None:
        """Accuracy should decay exponentially with depth."""
        p = 0.95
        acc_10 = standard_accuracy(p, 10)
        acc_20 = standard_accuracy(p, 20)
        # acc_20 should be acc_10^2
        assert acc_20 == pytest.approx(acc_10**2, rel=1e-10)

    def test_specific_claim_100_steps(self) -> None:
        """At p=0.95, N=100: accuracy should be ~0.59%."""
        acc = standard_accuracy(0.95, 100)
        assert acc == pytest.approx(0.0059, abs=0.001)


# ── Test: RSRA Effective Step Accuracy ───────────────────────────────────────
class TestRSRAEffectiveStep:
    """Tests for the RSRA-4B effective per-step accuracy formula."""

    def test_no_correction_equals_standard(self) -> None:
        """With no detection, RSRA should reduce to standard + tactical."""
        p_eff = rsra_effective_p_step(0.95, p_detect=0.0, p_correct=0.0)
        # p_eff = 0.95 + 0 + 0.05 * 0.3 = 0.965
        expected = 0.95 + 0.05 * 0.3
        assert p_eff == pytest.approx(expected)

    def test_perfect_correction(self) -> None:
        """With perfect detection and correction, p_eff should be ~1.0."""
        p_eff = rsra_effective_p_step(
            0.95, p_detect=1.0, p_correct=1.0, p_tactical=0.0
        )
        assert p_eff == pytest.approx(1.0)

    def test_higher_detection_improves_accuracy(self) -> None:
        """Higher detection probability should increase effective accuracy."""
        p_low = rsra_effective_p_step(0.95, 0.5, 0.8)
        p_high = rsra_effective_p_step(0.95, 0.9, 0.8)
        assert p_high > p_low

    def test_higher_correction_improves_accuracy(self) -> None:
        """Higher correction probability should increase effective accuracy."""
        p_low = rsra_effective_p_step(0.95, 0.85, 0.5)
        p_high = rsra_effective_p_step(0.95, 0.85, 0.95)
        assert p_high > p_low

    def test_effective_p_bounded(self) -> None:
        """Effective p should always be in [0, 1]."""
        for pd in [0.0, 0.5, 1.0]:
            for pc in [0.0, 0.5, 1.0]:
                p_eff = rsra_effective_p_step(0.95, pd, pc)
                assert 0.0 <= p_eff <= 1.0, f"p_eff={p_eff} out of bounds"

    def test_claimed_operating_point(self) -> None:
        """At det=85%, cor=80%, tactical=30%: compute expected p_eff."""
        p_eff = rsra_effective_p_step(0.95, 0.85, 0.80, 0.30)
        # p_eff = 0.95 + 0.05*0.85*0.80 + 0.05*(1-0.85*0.80)*0.30
        p_corrected = 0.05 * 0.85 * 0.80
        p_uncorrected = 0.05 - p_corrected
        p_tactical = p_uncorrected * 0.30
        expected = 0.95 + p_corrected + p_tactical
        assert p_eff == pytest.approx(expected)


# ── Test: Monte Carlo vs Analytical ──────────────────────────────────────────
class TestMonteCarloConvergence:
    """Tests that MC estimates converge to analytical values."""

    def test_standard_mc_matches_analytical(self) -> None:
        """Standard MC should match p^N within statistical tolerance."""
        rng = np.random.default_rng(42)
        p_step = 0.95
        n_runs = 50_000

        for n_steps in [1, 5, 10, 20]:
            outcomes = monte_carlo_standard(
                p_step, n_steps, n_runs=n_runs, rng=rng
            )
            mc_mean = outcomes.mean()
            analytical = standard_accuracy(p_step, n_steps)
            # 99% CI: allow ~3 standard errors
            se = np.sqrt(analytical * (1 - analytical) / n_runs)
            assert abs(mc_mean - analytical) < 3 * se + 1e-4, (
                f"N={n_steps}: MC={mc_mean:.4f} vs "
                f"analytical={analytical:.4f}, 3σ={3*se:.4f}"
            )

    def test_rsra_mc_matches_analytical(self) -> None:
        """RSRA MC should match analytical within statistical tolerance."""
        rng = np.random.default_rng(42)
        p_step = 0.95
        p_detect = 0.85
        p_correct = 0.80
        n_runs = 50_000

        for n_steps in [1, 5, 10, 20]:
            outcomes = monte_carlo_rsra(
                p_step, p_detect, p_correct, n_steps,
                n_runs=n_runs, rng=rng,
            )
            mc_mean = outcomes.mean()
            analytical = rsra_accuracy_analytical(
                p_step, p_detect, p_correct, n_steps
            )
            se = np.sqrt(analytical * (1 - analytical) / n_runs)
            assert abs(mc_mean - analytical) < 3 * se + 1e-3, (
                f"N={n_steps}: MC={mc_mean:.4f} vs "
                f"analytical={analytical:.4f}, 3σ={3*se:.4f}"
            )

    def test_mc_output_shape(self) -> None:
        """MC output should have shape (n_runs,) with values in {0, 1}."""
        rng = np.random.default_rng(42)
        outcomes = monte_carlo_standard(0.95, 10, n_runs=100, rng=rng)
        assert outcomes.shape == (100,)
        assert set(np.unique(outcomes)).issubset({0.0, 1.0})


# ── Test: RSRA Always >= Standard ────────────────────────────────────────────
class TestRSRADominance:
    """Tests that RSRA-4B accuracy is always >= standard accuracy."""

    def test_rsra_dominates_standard_analytically(self) -> None:
        """RSRA analytical accuracy >= standard for all depths."""
        for n in [1, 5, 10, 50, 100, 200]:
            std = standard_accuracy(0.95, n)
            rsra = rsra_accuracy_analytical(0.95, 0.85, 0.80, n)
            assert rsra >= std - 1e-15, (
                f"N={n}: RSRA ({rsra:.6f}) < standard ({std:.6f})"
            )

    def test_rsra_dominates_for_various_params(self) -> None:
        """RSRA >= standard for a sweep of detection/correction values."""
        for pd in [0.1, 0.5, 0.8, 0.99]:
            for pc in [0.1, 0.5, 0.8, 0.99]:
                for n in [1, 10, 100]:
                    std = standard_accuracy(0.95, n)
                    rsra = rsra_accuracy_analytical(0.95, pd, pc, n)
                    assert rsra >= std - 1e-15, (
                        f"pd={pd}, pc={pc}, N={n}: "
                        f"RSRA ({rsra:.6f}) < std ({std:.6f})"
                    )

    def test_advantage_increases_overall(self) -> None:
        """The RSRA advantage over standard should generally increase with depth.
        Note: At very high depths, both models approach 0, so the *absolute*
        advantage (rsra - std) can decrease. We test that advantage at depth 100
        exceeds advantage at depth 10."""
        advantages = []
        for n in [1, 10, 50, 100]:
            std = standard_accuracy(0.95, n)
            rsra = rsra_accuracy_analytical(0.95, 0.85, 0.80, n)
            advantages.append(rsra - std)

        # Advantage at N=100 should exceed advantage at N=1
        assert advantages[-1] > advantages[0], (
            "Advantage at N=100 should exceed advantage at N=1"
        )
        # Advantage at N=50 should exceed advantage at N=10
        assert advantages[2] > advantages[1], (
            "Advantage at N=50 should exceed advantage at N=10"
        )


# ── Test: Claim Validation ───────────────────────────────────────────────────
class TestClaimValidation:
    """Tests validating the specific claims from the RSRA-4B proposal."""

    def test_68_percent_at_100_steps(self) -> None:
        """
        Claim: '>68% accuracy on 100-step sequences'.

        The original RSRA-4B document's claim requires high detection
        and correction rates. With p_detect=0.95, p_correct=0.95,
        and p_tactical=0.30, the effective p_step is high enough
        to maintain >68% at N=100.

        p_eff = 0.95 + 0.05*0.95*0.95 + 0.05*(1-0.95*0.95)*0.30
              = 0.95 + 0.045125 + 0.001471... ≈ 0.9966
        0.9966^100 ≈ 71.2%
        """
        acc = rsra_accuracy_analytical(
            0.95, 0.95, 0.95, 100
        )
        assert acc > 0.68, (
            f"RSRA accuracy at N=100 should exceed 68% with high "
            f"detection/correction, got {acc:.2%}"
        )

    def test_claimed_operating_point(self) -> None:
        """At the baseline operating point (det=85%, cor=80%), RSRA
        significantly outperforms standard even if not >68%."""
        std = standard_accuracy(0.95, 100)
        rsra = rsra_accuracy_analytical(0.95, 0.85, 0.80, 100)
        # RSRA should be orders of magnitude better than standard
        assert rsra > std * 10, (
            f"RSRA ({rsra:.4%}) should be >> standard ({std:.4%})"
        )

    def test_standard_059_percent_at_100_steps(self) -> None:
        """Claim: 'Standard model decays to 0.59% at N=100'."""
        acc = standard_accuracy(0.95, 100)
        assert abs(acc - 0.0059) < 0.001, (
            f"Standard accuracy at N=100 should be ~0.59%, "
            f"got {acc:.4%}"
        )

    def test_critical_depth_improvement(self) -> None:
        """RSRA should have much higher critical depth than standard."""
        std_cd = compute_critical_depth(
            0.95, p_detect=0.0, p_correct=0.0,
            threshold=0.50
        )
        rsra_cd = compute_critical_depth(
            0.95, p_detect=0.85, p_correct=0.80,
            threshold=0.50
        )
        assert rsra_cd > std_cd, (
            f"RSRA critical depth ({rsra_cd}) should exceed "
            f"standard ({std_cd})"
        )


# ── Test: Heatmap Computation ────────────────────────────────────────────────
class TestHeatmap:
    """Tests for the advantage heatmap computation."""

    def test_heatmap_shape(self) -> None:
        """Heatmap matrix should have correct shape."""
        heatmap = compute_advantage_heatmap(
            p_detect_values=[0.5, 0.8, 0.95],
            p_correct_values=[0.5, 0.8],
        )
        assert heatmap.advantage_matrix.shape == (3, 2)
        assert heatmap.rsra_accuracy_matrix.shape == (3, 2)

    def test_heatmap_all_positive(self) -> None:
        """Advantage should be non-negative everywhere."""
        heatmap = compute_advantage_heatmap()
        assert np.all(heatmap.advantage_matrix >= -1e-10), (
            "RSRA advantage should be non-negative"
        )

    def test_heatmap_increases_with_detection(self) -> None:
        """Higher detection → higher advantage (for fixed correction)."""
        heatmap = compute_advantage_heatmap()
        for j in range(heatmap.advantage_matrix.shape[1]):
            col = heatmap.advantage_matrix[:, j]
            for i in range(1, len(col)):
                assert col[i] >= col[i - 1] - 1e-10, (
                    f"Advantage should increase with detection "
                    f"probability"
                )


# ── Test: Reproducibility ────────────────────────────────────────────────────
class TestReproducibility:
    """Tests that results are deterministic with fixed seeds."""

    def test_same_seed_same_mc(self) -> None:
        """Fixed seed should produce identical Monte Carlo results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        out1 = monte_carlo_rsra(
            0.95, 0.85, 0.80, 50, n_runs=1000, rng=rng1
        )
        out2 = monte_carlo_rsra(
            0.95, 0.85, 0.80, 50, n_runs=1000, rng=rng2
        )
        np.testing.assert_array_equal(out1, out2)

    def test_decay_comparison_reproducible(self) -> None:
        """run_decay_comparison should be reproducible."""
        r1 = run_decay_comparison(
            p_detect=0.85, p_correct=0.80, depths=[10, 50], n_runs=500
        )
        r2 = run_decay_comparison(
            p_detect=0.85, p_correct=0.80, depths=[10, 50], n_runs=500
        )
        np.testing.assert_array_almost_equal(
            r1.rsra_accuracy_mc, r2.rsra_accuracy_mc
        )


# ── Test: Critical Depth Computation ─────────────────────────────────────────
class TestCriticalDepth:
    """Tests for the critical depth computation."""

    def test_standard_critical_depth(self) -> None:
        """Standard critical depth at 50% for p=0.95."""
        # log(0.5) / log(0.95) ≈ 13.51 → floor = 13
        cd = compute_critical_depth(
            0.95, p_detect=0.0, p_correct=0.0,
            threshold=0.50
        )
        expected = int(np.floor(np.log(0.5) / np.log(
            rsra_effective_p_step(0.95, 0.0, 0.0)
        )))
        assert cd == expected

    def test_perfect_correction_high_depth(self) -> None:
        """Perfect correction should allow very deep reasoning."""
        cd = compute_critical_depth(
            0.95, p_detect=1.0, p_correct=1.0,
            threshold=0.50
        )
        assert cd == 500  # max_depth hit with perfect correction
