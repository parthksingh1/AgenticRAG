"""Judge-calibration maths.

The eval suite's credibility rests on these numbers, so they are tested against
hand-worked cases rather than against themselves. Where a property is easier to
state than an example — kappa is symmetric, a perfect judge has zero ECE — it is
checked with hypothesis over the whole input space.
"""

from __future__ import annotations

import itertools
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from src.services import calibration as cal

# A score sequence with enough spread that the fitted line is meaningful.
SCORES = st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=50)


class TestEce:
    """Expected calibration error."""

    def test_a_perfect_judge_scores_zero(self) -> None:
        """Identical judge and human scores means no calibration error."""
        values = [0.1, 0.35, 0.7, 0.95]
        assert cal.ece(values, values) == 0.0

    def test_a_maximally_wrong_judge_scores_one(self) -> None:
        """Confident and always wrong is the worst possible calibration."""
        assert cal.ece([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_bins_are_weighted_by_population(self) -> None:
        """A bin holding one case cannot outweigh one holding nine.

        Nine cases are perfectly calibrated and one is off by 1.0, so the error
        is one tenth of that, not half of it.
        """
        judge = [0.05] * 9 + [0.95]
        human = [0.05] * 9 + [0.0]
        assert cal.ece(judge, human, bins=2) == pytest.approx(0.095, abs=1e-6)

    def test_an_empty_sample_is_zero_not_an_error(self) -> None:
        """No labels means nothing measured, which is not an error condition."""
        assert cal.ece([], []) == 0.0

    def test_unpaired_sequences_are_rejected(self) -> None:
        """Zipping mismatched lengths would silently compute the wrong number."""
        with pytest.raises(ValueError, match="same length"):
            cal.ece([0.5, 0.5], [0.5])

    @given(SCORES)
    def test_a_judge_matching_the_humans_always_has_zero_error(self, values: list[float]) -> None:
        """Whatever the distribution, agreeing with the humans is zero error."""
        assert cal.ece(values, values) == 0.0

    @given(SCORES, SCORES)
    def test_error_is_bounded_to_the_unit_interval(
        self, judge: list[float], human: list[float]
    ) -> None:
        """ECE over scores in [0, 1] cannot leave [0, 1]."""
        if len(judge) != len(human):
            return
        assert 0.0 <= cal.ece(judge, human) <= 1.0


class TestCohensKappa:
    """Chance-corrected agreement."""

    def test_total_agreement_is_one(self) -> None:
        """Two raters who always agree, with both labels present, score 1."""
        assert cal.cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0

    def test_chance_level_agreement_is_zero(self) -> None:
        """Agreeing exactly as often as chance predicts scores 0."""
        assert cal.cohens_kappa([1, 0, 1, 0], [1, 0, 0, 1]) == 0.0

    def test_systematic_disagreement_is_negative(self) -> None:
        """Worse than chance is a real signal, not clamped to zero."""
        assert cal.cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) < 0

    def test_no_variance_is_undefined_rather_than_perfect(self) -> None:
        """Both raters labelling everything the same demonstrates no agreement.

        Reporting 1.0 here would claim strong agreement from a dataset that shows
        none, which is exactly the failure mode kappa exists to prevent.
        """
        assert cal.cohens_kappa([1, 1, 1], [1, 1, 1]) is None

    def test_an_empty_sample_is_undefined(self) -> None:
        """Nothing to compare means no coefficient."""
        assert cal.cohens_kappa([], []) is None

    @given(st.lists(st.integers(0, 1), min_size=2, max_size=40))
    def test_kappa_is_symmetric(self, a: list[int]) -> None:
        """Which rater is listed first cannot change the agreement."""
        b = list(reversed(a))
        assert cal.cohens_kappa(a, b) == cal.cohens_kappa(b, a)


class TestPearson:
    """Correlation."""

    def test_a_perfect_linear_relationship_is_one(self) -> None:
        """A judge that is a scaled copy of the humans correlates perfectly."""
        assert cal.pearson_r([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0

    def test_a_perfect_inverse_relationship_is_minus_one(self) -> None:
        """A judge that is exactly backwards is still perfectly correlated."""
        assert cal.pearson_r([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0

    def test_a_constant_series_has_no_correlation(self) -> None:
        """Zero variance means the coefficient is undefined, not zero."""
        assert cal.pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None

    def test_a_single_point_has_no_correlation(self) -> None:
        """One point defines no relationship."""
        assert cal.pearson_r([0.5], [0.5]) is None


class TestFitLine:
    """Least-squares recalibration."""

    def test_it_recovers_a_known_line(self) -> None:
        """A judge scoring double the humans is corrected by halving."""
        slope, intercept = cal.fit_line([0.2, 0.4, 0.6], [0.1, 0.2, 0.3])
        assert slope == pytest.approx(0.5)
        assert intercept == pytest.approx(0.0, abs=1e-9)

    def test_a_constant_judge_falls_back_to_the_identity(self) -> None:
        """There is no line through a vertical point cloud.

        The identity leaves the scores unchanged, which is wrong but honest;
        fitting a horizontal line would collapse every future score to one value.
        """
        assert cal.fit_line([0.5, 0.5, 0.5], [0.1, 0.5, 0.9]) == (1.0, 0.0)

    def test_a_single_point_falls_back_to_the_identity(self) -> None:
        """One point does not determine a slope."""
        assert cal.fit_line([0.5], [0.9]) == (1.0, 0.0)


class TestCalibrationObject:
    """The assembled calibration."""

    def test_a_perfect_judge_gets_full_weight(self) -> None:
        """Zero error means a full vote."""
        result = cal.calibrate("j", [0.8, 0.6, 0.9, 0.2], [0.8, 0.6, 0.9, 0.2])
        assert result.expected_calibration_error == 0.0
        assert result.weight == 1.0
        assert result.mean_absolute_error == 0.0

    def test_a_small_sample_is_flagged_unreliable(self) -> None:
        """Below the minimum the numbers are noise and say so."""
        result = cal.calibrate("j", [0.5] * 5, [0.5] * 5)
        assert not result.is_reliable

    def test_a_large_sample_is_reliable(self) -> None:
        """At the threshold the calibration is actionable."""
        size = cal.MIN_RELIABLE_SAMPLE
        result = cal.calibrate("j", [0.5] * size, [0.5] * size)
        assert result.is_reliable

    def test_apply_clamps_to_the_unit_interval(self) -> None:
        """A fitted line can leave [0, 1]; a reported score never should."""
        c = cal.Calibration("j", 50, 0.1, None, None, 0.1, 2.0, 0.5, 0.9, {})
        assert c.apply(1.0) == 1.0
        assert c.apply(0.0) == 0.5

    def test_apply_maps_toward_the_human_scale(self) -> None:
        """An over-confident judge is pulled down toward what humans said."""
        judge = [0.9, 0.8, 0.95, 0.85]
        human = [0.5, 0.4, 0.55, 0.45]
        result = cal.calibrate("optimist", judge, human)
        assert result.apply(0.9) == pytest.approx(0.5, abs=0.05)


class TestWeightingAndCombination:
    """How judges are weighted against each other."""

    def test_weight_decreases_monotonically_with_error(self) -> None:
        """More miscalibration is always less vote, never more."""
        weights = [cal.weight_for(e) for e in (0.0, 0.1, 0.3, 0.6, 1.0)]
        assert weights == sorted(weights, reverse=True)
        assert weights[0] == 1.0

    def test_a_negative_error_cannot_buy_extra_weight(self) -> None:
        """ECE is non-negative; a bad input must not exceed a full vote."""
        assert cal.weight_for(-5.0) == 1.0

    def test_equal_weights_give_the_mean(self) -> None:
        """Two equally trusted judges average."""
        assert cal.combine({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 1.0}) == 0.5

    def test_a_worthless_judge_is_ignored(self) -> None:
        """Zero weight removes a judge's influence without removing its score."""
        assert cal.combine({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0}) == 1.0

    def test_an_unknown_judge_counts_as_fully_weighted(self) -> None:
        """A judge added this week has no calibration yet.

        Dropping it instead would silently change what the metric means in the
        middle of an experiment.
        """
        assert cal.combine({"new": 0.4}, {}) == 0.4

    def test_all_weights_zero_falls_back_to_an_unweighted_mean(self) -> None:
        """Better an honest average than a division by zero."""
        assert cal.combine({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 0.0}) == 0.5

    def test_no_scores_is_zero(self) -> None:
        """Nothing to combine."""
        assert cal.combine({}, {"a": 1.0}) == 0.0

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=5),
            st.floats(min_value=0.0, max_value=1.0),
            min_size=1,
            max_size=5,
        )
    )
    def test_a_combination_never_leaves_the_range_of_its_inputs(
        self, scores: dict[str, float]
    ) -> None:
        """A weighted mean lies between the smallest and largest input."""
        combined = cal.combine(scores, {})
        assert min(scores.values()) - 1e-6 <= combined <= max(scores.values()) + 1e-6


class TestReliabilityBins:
    """The reliability diagram."""

    def test_every_bin_is_present_even_when_empty(self) -> None:
        """The chart needs the empty bins to show the gap."""
        result = cal.reliability_bins([0.05], [0.0], bins=4)
        assert len(result["bins"]) == 4
        assert result["bins"][0]["count"] == 1
        assert result["bins"][3]["count"] == 0
        assert result["bins"][3]["judge_mean"] is None

    def test_a_score_of_exactly_one_lands_in_the_last_bin(self) -> None:
        """Half-open bins would otherwise drop the perfect score entirely."""
        result = cal.reliability_bins([1.0], [1.0], bins=2)
        assert result["bins"][1]["count"] == 1


class TestScoreExtraction:
    """Reading scores back out of stored verdicts."""

    def test_both_stored_shapes_are_read(self) -> None:
        """Older rows hold a bare number; newer ones hold a verdict object.

        Both are read rather than migrated, because reshaping historical eval
        data destroys the comparability the data exists for.
        """
        assert cal._score_of({"score": 0.8, "reasoning": "..."}) == 0.8
        assert cal._score_of(0.5) == 0.5

    def test_unusable_values_are_skipped(self) -> None:
        """A judge that answered in prose contributes nothing, silently."""
        assert cal._score_of("very good") is None
        assert cal._score_of(None) is None
        assert cal._score_of({"reasoning": "no score"}) is None


def test_bin_edges_tile_the_unit_interval() -> None:
    """The bins must cover [0, 1] exactly, with no gap and no overlap."""
    edges = cal._bin_edges(cal.DEFAULT_BINS)
    assert edges[0][0] == 0.0
    assert math.isclose(edges[-1][1], 1.0)
    for (_, upper), (lower, _) in itertools.pairwise(edges):
        assert math.isclose(upper, lower)
