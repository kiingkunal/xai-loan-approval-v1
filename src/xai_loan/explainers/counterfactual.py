"""DiCE-based counterfactual explanations.

DiCE perturbs raw applicant features directly, so it needs the model
wrapped together with preprocessing — see `xai_loan.models.pipeline.PipelineModel`,
which exposes predict/predict_proba over the raw feature space. Without
this, counterfactuals would come back in one-hot/scaled units (e.g.
"income = -0.3"), meaningless to an applicant or loan officer — DiCE
needs to search in the same units the narrative generator and dashboard
report in.
"""

from __future__ import annotations

from typing import Self

import dice_ml
import pandas as pd
from raiutils.exceptions import UserConfigValidationException

from xai_loan.models.pipeline import PipelineModel


class CounterfactualGenerator:
    """Generates minimal-change counterfactuals using DiCE.

    Attributes:
        method: DiCE generation strategy — ``"random"`` (fast, default),
            ``"genetic"``, or ``"kdtree"``.
    """

    def __init__(self, method: str = "random") -> None:
        """Initialize the generator.

        Args:
            method: Which DiCE search strategy to use.
        """
        self.method = method
        self._explainer: dice_ml.Dice | None = None
        self._feature_cols: list[str] = []

    def fit(
        self,
        model: object,
        preprocessor: object,
        train_df: pd.DataFrame,
        categorical_cols: list[str],
        numeric_cols: list[str],
        target_col: str,
    ) -> Self:
        """Build the DiCE explainer over the raw (pre-preprocessing) feature space.

        Args:
            model: A fitted classifier exposing `predict_proba`, trained
                on `preprocessor`-transformed features.
            preprocessor: The `LoanDataPreprocessor` already fitted on
                this same training data.
            train_df: Raw training DataFrame (un-preprocessed) — DiCE
                uses it to learn realistic feature ranges and categories
                to search within.
            categorical_cols: Raw categorical feature columns.
            numeric_cols: Raw numeric feature columns.
            target_col: Name of the target column in `train_df`.

        Returns:
            self, for chaining.
        """
        self._feature_cols = list(categorical_cols) + list(numeric_cols)
        pipeline_model = PipelineModel(preprocessor, model, self._feature_cols)

        data = dice_ml.Data(
            dataframe=train_df[self._feature_cols + [target_col]],
            continuous_features=list(numeric_cols),
            outcome_name=target_col,
        )
        dice_model = dice_ml.Model(model=pipeline_model, backend="sklearn")
        self._explainer = dice_ml.Dice(data, dice_model, method=self.method)
        return self

    def generate(
        self,
        instance: pd.DataFrame,
        n: int = 3,
        desired_class: int = 1,
        features_to_vary: list[str] | str = "all",
    ) -> list[dict[str, object]]:
        """Generate counterfactuals that flip the prediction toward `desired_class`.

        Args:
            instance: A single-row raw (un-preprocessed) feature
                DataFrame for the applicant being explained.
            n: Number of counterfactuals to request.
            desired_class: Target class label the counterfactuals should
                predict. This framework's convention is 0 = approve,
                1 = reject — so explaining a rejected applicant means
                passing ``desired_class=0``.
            features_to_vary: Restricts which raw columns DiCE is allowed
                to change, or ``"all"`` (DiCE's default) to allow every
                feature. Use this to keep counterfactuals actionable —
                e.g. a model using building-statistic or document-flag
                features has no business recommending the applicant
                change one of those; restricting the search to fields a
                person could actually act on (income, loan amount, ...)
                produces both a more realistic counterfactual *and* a
                plain-English narrative that makes sense, since
                `narrative.generate_narrative` describes whatever
                feature actually changed.

        Returns:
            A list of up to `n` dicts, each a modified version of
            `instance`'s features with the minimum change DiCE found to
            reach `desired_class`. Returns an empty list if DiCE can't
            find any valid counterfactual.

        Raises:
            RuntimeError: If called before `fit`.
        """
        if self._explainer is None:
            raise RuntimeError("CounterfactualGenerator.generate() called before fit().")

        try:
            result = self._explainer.generate_counterfactuals(
                instance[self._feature_cols],
                total_CFs=n,
                desired_class=desired_class,
                features_to_vary=features_to_vary,
            )
        except UserConfigValidationException:
            # DiCE raises this when it finds zero counterfactuals for the
            # query (distinct from finding fewer than `n`, which it
            # returns normally) -- expected and more likely once
            # `features_to_vary` restricts the search space, not a bug.
            return []

        cfs_df = result.cf_examples_list[0].final_cfs_df
        if cfs_df is None:
            return []
        return cfs_df[self._feature_cols].to_dict(orient="records")
