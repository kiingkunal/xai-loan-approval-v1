"""Project-wide paths and constants.

Centralizes filesystem locations and default hyperparameters so every
module agrees on where data/models/reports live and which random seed
and thresholds to use, without each module hardcoding its own copy.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
"""Root of the repository (parent of ``src/``)."""

DATA_DIR: Path = PROJECT_ROOT / "data"
"""Directory holding downloaded datasets (gitignored)."""

MODELS_DIR: Path = PROJECT_ROOT / "models"
"""Directory holding trained model artifacts (gitignored)."""

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
"""Directory holding generated figures and tables (gitignored)."""

GERMAN_CREDIT_DIR: Path = DATA_DIR / "german_credit"
"""Directory holding the UCI German Credit dataset files."""

HOME_CREDIT_DIR: Path = DATA_DIR / "home_credit"
"""Directory holding the Home Credit Default Risk dataset files."""

RANDOM_STATE: int = 42
"""Random seed used everywhere for reproducibility (splits, SMOTE, models)."""

TEST_SIZE: float = 0.2
"""Fraction of data held out as the test set during train/test splitting."""

DEFAULT_MODELS: tuple[str, ...] = ("logreg", "rf", "xgb")
"""Model identifiers trained by default via ``models.train.train_models``."""

TRUST_SCORE_WEIGHTS: dict[str, float] = {
    "model_confidence": 0.4,
    "shap_lime_agreement": 0.3,
    "counterfactual_feasibility": 0.3,
}
"""Default weights for the three trust score components (must sum to 1.0)."""

TRUST_SCORE_THRESHOLDS: dict[str, float] = {
    "auto_decision": 80.0,
    "human_review": 50.0,
}
"""Trust score cutoffs: ``>= auto_decision`` auto-decides, ``< human_review``
triggers strong human review, the band between is routine human review."""
