"""Sanity checks for FairnessAuditor's metrics and reweighing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xai_loan.fairness.audit import FairnessAuditor, bucket_age_band, sensitive_feature_series


class _StubModel:
    """A fake classifier that returns pre-set predictions, for isolating
    the fairness math from any real model training."""

    def __init__(self, predictions: np.ndarray) -> None:
        self._predictions = predictions

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predictions


def test_audit_detects_biased_predictions() -> None:
    n = 200
    sensitive = pd.Series(["A"] * (n // 2) + ["B"] * (n // 2), name="group")
    y_true = pd.Series(np.zeros(n, dtype=int))
    # group B always predicted positive, group A always predicted negative:
    # maximal demographic parity gap.
    y_pred = np.array([0] * (n // 2) + [1] * (n // 2))

    auditor = FairnessAuditor()
    result = auditor.audit(_StubModel(y_pred), pd.DataFrame(index=range(n)), y_true, sensitive)

    assert result["demographic_parity_difference"] == pytest.approx(1.0)
    assert set(result["false_positive_rate_by_group"].keys()) == {"A", "B"}
    assert result["false_positive_rate_by_group"]["A"] == pytest.approx(0.0)
    assert result["false_positive_rate_by_group"]["B"] == pytest.approx(1.0)
    assert result["false_positive_rate_parity_difference"] == pytest.approx(1.0)


def test_audit_finds_no_disparity_when_predictions_match_across_groups() -> None:
    # exactly 50% positive rate within EACH group by construction (not by
    # chance), so a perfect classifier has zero demographic parity gap.
    n_per_group = 100
    sensitive = pd.Series(["A"] * n_per_group + ["B"] * n_per_group)
    y_true = pd.Series(([0, 1] * (n_per_group // 2)) * 2)
    y_pred = y_true.to_numpy()  # perfect predictions, identical behavior per group

    auditor = FairnessAuditor()
    result = auditor.audit(
        _StubModel(y_pred), pd.DataFrame(index=range(2 * n_per_group)), y_true, sensitive
    )

    assert result["demographic_parity_difference"] == pytest.approx(0.0, abs=1e-9)
    assert result["equalized_odds_difference"] == pytest.approx(0.0, abs=1e-9)


def test_audit_metrics_bounded_between_zero_and_one() -> None:
    rng = np.random.default_rng(1)
    n = 150
    sensitive = pd.Series(rng.choice(["A", "B", "C"], n), name="group")
    y_true = pd.Series(rng.integers(0, 2, n))
    y_pred = rng.integers(0, 2, n)

    auditor = FairnessAuditor()
    result = auditor.audit(_StubModel(y_pred), pd.DataFrame(index=range(n)), y_true, sensitive)

    assert 0.0 <= result["demographic_parity_difference"] <= 1.0
    assert 0.0 <= result["equalized_odds_difference"] <= 1.0
    for rate in result["false_positive_rate_by_group"].values():
        assert 0.0 <= rate <= 1.0


def test_apply_reweighing_mismatched_lengths_raises() -> None:
    auditor = FairnessAuditor()
    with pytest.raises(ValueError):
        auditor.apply_reweighing(
            pd.DataFrame({"a": [1, 2, 3]}),
            pd.Series([0, 1]),
            pd.Series(["A", "B"]),
        )


def test_apply_reweighing_upweights_underrepresented_group_label_combo() -> None:
    # group A: 100 rows, 20 labeled 1 (P(label=1|A) = 0.2)
    # group B: 20 rows, 16 labeled 1 (P(label=1|B) = 0.8)
    # overall P(label=1) = 36/120 = 0.3, P(A) = 100/120, P(B) = 20/120
    #
    # (A, label=1) is rarer within A than the overall rate -> underrepresented
    # relative to independence -> hand-computed weight = (100/120)*(0.3)/(20/120) = 1.5
    # (B, label=1) is far more common within B than the overall rate -> overrepresented
    # -> hand-computed weight = (20/120)*(0.3)/(16/120) = 0.375
    sensitive = pd.Series(["A"] * 100 + ["B"] * 20)
    y = pd.Series([0] * 80 + [1] * 20 + [0] * 4 + [1] * 16)
    X = pd.DataFrame(index=sensitive.index)

    auditor = FairnessAuditor()
    weights = auditor.apply_reweighing(X, y, sensitive)

    weight_a1 = weights[(sensitive == "A") & (y == 1)].iloc[0]
    weight_b1 = weights[(sensitive == "B") & (y == 1)].iloc[0]

    assert weight_a1 == pytest.approx(1.5)
    assert weight_b1 == pytest.approx(0.375)
    assert weight_a1 > 1.0  # underrepresented combo -> upweighted
    assert weight_b1 < 1.0  # overrepresented combo -> downweighted
    assert (weights > 0).all()


def test_apply_reweighing_equal_distribution_gives_uniform_weights() -> None:
    # two groups with identical label distribution -> every combo's
    # joint probability already matches its marginal product -> weight 1.
    sensitive = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])
    y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    X = pd.DataFrame(index=sensitive.index)

    auditor = FairnessAuditor()
    weights = auditor.apply_reweighing(X, y, sensitive)

    assert np.allclose(weights.to_numpy(), 1.0)


def test_bucket_age_band_splits_at_cutoff() -> None:
    ages = pd.Series([18, 24, 25, 26, 40])
    bands = bucket_age_band(ages, cutoff=25)
    assert bands.tolist() == ["under_25", "under_25", "25_or_over", "25_or_over", "25_or_over"]


def test_sensitive_feature_series_buckets_age_but_passes_through_sex() -> None:
    X = pd.DataFrame({"sex": ["male", "female", "male"], "age": [20, 30, 40]})
    pairs = sensitive_feature_series(["sex", "age"], X)

    pairs_by_name = dict(pairs)
    assert list(pairs_by_name["sex"]) == ["male", "female", "male"]
    assert list(pairs_by_name["age"]) == ["under_25", "25_or_over", "25_or_over"]


def test_sensitive_feature_series_converts_days_birth_to_age_band() -> None:
    # DAYS_BIRTH is negative days since birth; -7305 days = 20 years,
    # -10950 days = 30 years.
    X = pd.DataFrame({"DAYS_BIRTH": [-7305, -10950]})
    pairs = sensitive_feature_series(["DAYS_BIRTH"], X)

    assert dict(pairs)["DAYS_BIRTH"].tolist() == ["under_25", "25_or_over"]
