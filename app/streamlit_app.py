"""Streamlit demo dashboard tying the whole framework together.

Run with: streamlit run app/streamlit_app.py

Requires `scripts/train_all.py` to have been run first — this app only
loads saved models, it never trains anything. Per-dataset artifacts
(preprocessor, model, explainers, the DiCE counterfactual generator) are
loaded once via `st.cache_resource` and reused across every interaction;
the form submission triggers `run_prediction_pipeline`, whose result is
stashed in `st.session_state` so the Explain/Counterfactuals tabs can
render it without recomputing SHAP/LIME/DiCE on every Streamlit rerun
(Streamlit re-executes the whole script on every interaction, including
switching tabs — only the *prediction* itself should be expensive, not
viewing tabs of an already-computed result).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from field_specs import FIELD_SPECS, CategoricalField, NumericField
from sklearn.model_selection import train_test_split

from xai_loan.data import feature_labels
from xai_loan.data.loader import load_german_credit, load_home_credit
from xai_loan.data.preprocessor import split_features_target
from xai_loan.explainers.counterfactual import CounterfactualGenerator
from xai_loan.explainers.lime_explainer import LIMEExplainer
from xai_loan.explainers.narrative import generate_narrative
from xai_loan.explainers.shap_explainer import SHAPExplainer
from xai_loan.fairness.audit import FairnessAuditor, sensitive_feature_series
from xai_loan.models.pipeline import PipelineModel
from xai_loan.models.registry import load_model
from xai_loan.trust.score import compute_trust_score
from xai_loan.utils.config import RANDOM_STATE, TEST_SIZE

_BEST_MODEL_NAME = "xgb_tuned"
_HOME_CREDIT_UI_SAMPLE_SIZE = 20_000
_BACKGROUND_SAMPLE_SIZE = 100

_DATASET_OPTIONS = {"German Credit": "german", "Home Credit (large)": "home_credit"}


@st.cache_resource(show_spinner="Loading dataset and trained models...")
def load_dataset_bundle(dataset_label: str) -> dict[str, object]:
    """Load a dataset's data, trained artifacts, and fitted explainers.

    Cached per `dataset_label` so switching datasets in the sidebar
    triggers exactly one reload, not one per Streamlit rerun.

    Args:
        dataset_label: One of the keys in `_DATASET_OPTIONS`.

    Returns:
        A dict bundling everything every tab needs: the raw df,
        metadata, preprocessor, best model, a `PipelineModel` wrapper,
        the train/test split, and pre-fitted SHAP/LIME/counterfactual
        explainers.
    """
    prefix = _DATASET_OPTIONS[dataset_label]
    if prefix == "german":
        df, metadata = load_german_credit()
    else:
        df, metadata = load_home_credit(sample_size=_HOME_CREDIT_UI_SAMPLE_SIZE)

    preprocessor = load_model(f"{prefix}_preprocessor")
    model = load_model(f"{prefix}_{_BEST_MODEL_NAME}")

    X, y = split_features_target(df, metadata["target_col"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    feature_cols = metadata["categorical_cols"] + metadata["numeric_cols"]
    pipeline_model = PipelineModel(preprocessor, model, feature_cols)

    background_raw = X_train.sample(n=min(_BACKGROUND_SAMPLE_SIZE, len(X_train)), random_state=RANDOM_STATE)
    background_transformed = preprocessor.transform(background_raw)
    shap_explainer = SHAPExplainer().fit(model, background_transformed)
    lime_explainer = LIMEExplainer().fit(model, background_transformed)

    cf_train_df = X_train.copy()
    cf_train_df[metadata["target_col"]] = y_train
    counterfactual_generator = CounterfactualGenerator().fit(
        model,
        preprocessor,
        cf_train_df,
        metadata["categorical_cols"],
        metadata["numeric_cols"],
        metadata["target_col"],
    )

    # The Predict form only shows a curated subset of fields (see
    # field_specs.py) -- whatever isn't shown still needs a value to hand
    # the model a complete row, so fill every feature with its
    # training-set mode (categorical) / median (numeric) up front.
    default_values: dict[str, object] = {}
    for col in metadata["categorical_cols"]:
        default_values[col] = X_train[col].mode().iloc[0]
    for col in metadata["numeric_cols"]:
        default_values[col] = float(X_train[col].median())

    # Restricts DiCE's counterfactual search to the same human-meaningful
    # columns the narrative is allowed to cite (see feature_labels.py) --
    # German Credit's allowlist covers its whole feature set, so this is
    # effectively "all" there; Home Credit's allowlist is the curated
    # ~14-field subset, so DiCE never recommends changing a building
    # statistic or document flag.
    features_to_vary = sorted(feature_labels.NARRATIVE_FEATURE_ALLOWLIST.get(prefix, set()) & set(feature_cols))

    return {
        "prefix": prefix,
        "df": df,
        "metadata": metadata,
        "features_to_vary": features_to_vary,
        "preprocessor": preprocessor,
        "model": model,
        "pipeline_model": pipeline_model,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "shap_explainer": shap_explainer,
        "lime_explainer": lime_explainer,
        "counterfactual_generator": counterfactual_generator,
        "default_values": default_values,
    }


def run_prediction_pipeline(bundle: dict[str, object], raw_instance: pd.DataFrame) -> dict[str, object]:
    """Run one applicant through the full prediction + explanation stack.

    Args:
        bundle: Output of `load_dataset_bundle`.
        raw_instance: Single-row raw (un-preprocessed) applicant DataFrame.
    Returns:
        A dict with everything the Predict/Explain/Counterfactuals tabs
        need to render: ``raw_instance``, ``preprocessed_instance``,
        ``prediction``, ``probability_reject``, ``shap_exp``,
        ``lime_exp``, ``counterfactuals``, ``narrative``, ``trust_result``.
    """
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]

    preprocessed_instance = preprocessor.transform(raw_instance)
    probabilities = model.predict_proba(preprocessed_instance)[0]
    prediction = int(model.predict(preprocessed_instance)[0])

    shap_exp = bundle["shap_explainer"].local_explanation(preprocessed_instance)
    lime_exp = bundle["lime_explainer"].local_explanation(preprocessed_instance)
    counterfactuals = bundle["counterfactual_generator"].generate(
        raw_instance, n=3, desired_class=1 - prediction, features_to_vary=bundle["features_to_vary"]
    )
    narrative = generate_narrative(
        raw_instance,
        prediction,
        shap_exp,
        counterfactuals,
        categorical_cols=bundle["metadata"]["categorical_cols"],
        dataset=bundle["prefix"],
    )
    trust_result = compute_trust_score(
        raw_instance, bundle["pipeline_model"], shap_exp, lime_exp, counterfactuals
    )

    return {
        "raw_instance": raw_instance,
        "preprocessed_instance": preprocessed_instance,
        "prediction": prediction,
        "probability_reject": float(probabilities[1]),
        "shap_exp": shap_exp,
        "lime_exp": lime_exp,
        "counterfactuals": counterfactuals,
        "narrative": narrative,
        "trust_result": trust_result,
    }


def render_predict_tab(bundle: dict[str, object]) -> None:
    """Render the applicant form, then the prediction + trust score + narrative."""
    fields = FIELD_SPECS[bundle["prefix"]]

    st.header("Predict")
    st.write("Enter applicant details, then submit for a prediction with a full explanation.")
    if bundle["prefix"] == "home_credit":
        st.caption(
            "Showing the fields a loan officer would realistically collect. The "
            "remaining ~100 technical features in this dataset (external bureau "
            "scores, document-submission flags, etc.) are filled with training-set "
            "averages -- in a real deployment those would come from internal "
            "systems or a credit bureau API, not a form."
        )

    sections: dict[str, list[CategoricalField | NumericField]] = {}
    for spec in fields:
        sections.setdefault(spec.section, []).append(spec)

    with st.form("applicant_form"):
        ui_values: dict[str, object] = {}
        for section_name, section_fields in sections.items():
            st.subheader(section_name)
            form_cols = st.columns(2)
            for i, spec in enumerate(section_fields):
                target_col = form_cols[i % 2]
                if isinstance(spec, CategoricalField):
                    code_labels = spec.code_labels or {}
                    options = sorted(code_labels.keys()) or sorted(
                        bundle["df"][spec.raw_col].dropna().unique().tolist()
                    )
                    ui_values[spec.raw_col] = target_col.selectbox(
                        spec.label, options, format_func=lambda value, labels=code_labels: labels.get(value, value)
                    )
                else:
                    ui_values[spec.raw_col] = target_col.number_input(
                        spec.label,
                        value=float(spec.default),
                        min_value=float(spec.min_value),
                        max_value=float(spec.max_value),
                        step=float(spec.step),
                    )

        submitted = st.form_submit_button("Predict")

    if submitted:
        raw_row = dict(bundle["default_values"])
        for spec in fields:
            value = ui_values[spec.raw_col]
            raw_row[spec.raw_col] = spec.to_raw(value) if isinstance(spec, NumericField) else value
        raw_instance = pd.DataFrame([raw_row])
        st.session_state["last_result"] = run_prediction_pipeline(bundle, raw_instance)

    result = st.session_state.get("last_result")
    if result is None:
        st.info("Submit the form above to see a prediction.")
        return

    prediction = result["prediction"]
    trust_result = result["trust_result"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Decision", "Rejected" if prediction == 1 else "Approved")
    col2.metric("Probability of rejection", f"{result['probability_reject']:.1%}")
    col3.metric("Trust score", f"{trust_result['score']}/100", trust_result["verdict"])

    st.subheader("Plain-English Explanation")
    st.write(result["narrative"])

    st.subheader("Trust Score Detail")
    st.json(trust_result)


def render_explain_tab(bundle: dict[str, object]) -> None:
    """Render SHAP waterfall + LIME bar chart side by side for the last prediction."""
    st.header("Explain")
    result = st.session_state.get("last_result")
    if result is None:
        st.info("Make a prediction in the Predict tab first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SHAP")
        bundle["shap_explainer"].plot_waterfall(result["preprocessed_instance"])
        st.pyplot(plt.gcf())
        plt.close("all")

    with col2:
        st.subheader("LIME")
        lime_exp = result["lime_exp"]
        lime_df = pd.DataFrame({"feature": lime_exp["feature_names"], "weight": lime_exp["weights"]})
        lime_df["direction"] = lime_df["weight"].apply(lambda w: "pushes toward reject" if w > 0 else "pushes toward approve")
        lime_df = lime_df.sort_values("weight")
        fig = px.bar(
            lime_df,
            x="weight",
            y="feature",
            color="direction",
            orientation="h",
            title="LIME feature contributions",
        )
        st.plotly_chart(fig, width="stretch")


def render_counterfactuals_tab(bundle: dict[str, object]) -> None:
    """Render the original applicant alongside DiCE's alternative scenarios."""
    st.header("Counterfactuals")
    result = st.session_state.get("last_result")
    if result is None:
        st.info("Make a prediction in the Predict tab first.")
        return

    counterfactuals = result["counterfactuals"]
    if not counterfactuals:
        st.warning("No realistic counterfactual was found that would flip this decision.")
        return

    rows = [dict(result["raw_instance"].iloc[0])] + counterfactuals
    labels = ["Original"] + [f"Alternative {i + 1}" for i in range(len(counterfactuals))]
    st.dataframe(pd.DataFrame(rows, index=labels))


def render_fairness_tab(bundle: dict[str, object]) -> None:
    """Render group-fairness metrics for the held-out test set, per protected attribute."""
    st.header("Fairness Audit")
    metadata = bundle["metadata"]
    X_test_transformed = bundle["preprocessor"].transform(bundle["X_test"])

    auditor = FairnessAuditor()
    for label, sensitive_series in sensitive_feature_series(metadata["protected_cols"], bundle["X_test"]):
        st.subheader(f"Protected attribute: {label}")
        audit_result = auditor.audit(bundle["model"], X_test_transformed, bundle["y_test"], sensitive_series)

        col1, col2, col3 = st.columns(3)
        col1.metric("Demographic parity diff", f"{audit_result['demographic_parity_difference']:.3f}")
        col2.metric("Equalized odds diff", f"{audit_result['equalized_odds_difference']:.3f}")
        col3.metric("FPR parity diff", f"{audit_result['false_positive_rate_parity_difference']:.3f}")

        st.bar_chart(pd.Series(audit_result["selection_rate_by_group"], name="rejection rate by group"))


def render_about_tab() -> None:
    """Render a short description of the framework and decision-support layer."""
    st.header("About")
    st.markdown(
        """
This dashboard demonstrates an explainable loan-decision workflow for tabular
credit data. It combines model predictions with SHAP, LIME, counterfactual
analysis, group-fairness metrics, and a human-readable decision summary.

**Plain-English Narrative Generator** translates feature attributions and
counterfactuals into concise explanations that a non-technical reviewer can
understand.

**Trust Score** combines model confidence, SHAP/LIME agreement, and
counterfactual feasibility into a 0-100 score with a routing verdict, helping
identify decisions that should be reviewed by a human.

This is a research and portfolio-grade decision-support demo, not a production
lending system.
        """
    )


def main() -> None:
    """Build the page layout and dispatch to each tab's render function."""
    st.set_page_config(page_title="Explainable Loan Approval", layout="wide")
    st.title("Explainable Loan Approval")

    dataset_label = st.sidebar.selectbox("Dataset", list(_DATASET_OPTIONS.keys()), index=0)
    if st.session_state.get("dataset_label") != dataset_label:
        st.session_state["last_result"] = None
        st.session_state["dataset_label"] = dataset_label

    bundle = load_dataset_bundle(dataset_label)

    tab_predict, tab_explain, tab_counterfactuals, tab_fairness, tab_about = st.tabs(
        ["Predict", "Explain", "Counterfactuals", "Fairness", "About"]
    )
    with tab_predict:
        render_predict_tab(bundle)
    with tab_explain:
        render_explain_tab(bundle)
    with tab_counterfactuals:
        render_counterfactuals_tab(bundle)
    with tab_fairness:
        render_fairness_tab(bundle)
    with tab_about:
        render_about_tab()


main()
