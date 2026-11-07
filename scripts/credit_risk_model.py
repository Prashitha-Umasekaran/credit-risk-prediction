
# ============================================================
# CREDIT RISK MODEL - EXPERIMENTAL ANALYSIS
# Complete Analysis, Model Evaluation and Visualization
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Delinquency_prediction_dataset.xlsx"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "outputs", "reports"
)

VISUAL_DIR = os.path.join(
    BASE_DIR, "visuals"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_excel(DATA_PATH)

target_col = "Delinquent_Account"

if target_col not in df.columns:
    raise ValueError(
        f"Target column '{target_col}' was not found."
    )

df = df.dropna(subset=[target_col]).copy()

print("\n===== DATA LOADED =====")
print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df[target_col].value_counts())


# ============================================================
# 3. PREPARE TARGET
# ============================================================

target_values = list(df[target_col].unique())

if len(target_values) != 2:
    raise ValueError(
        "The target column must contain exactly two classes."
    )

if set(target_values).issubset({0, 1, 0.0, 1.0}):
    y = df[target_col].astype(int)

else:
    normalized = {
        str(value).strip().lower(): value
        for value in target_values
    }

    positive_labels = {
        "1",
        "yes",
        "true",
        "delinquent",
        "default",
        "defaulted"
    }

    positive_values = [
        value
        for label, value in normalized.items()
        if label in positive_labels
    ]

    if len(positive_values) != 1:
        raise ValueError(
            "Could not safely identify the delinquent class. "
            "Please check the target values."
        )

    y = (
        df[target_col] == positive_values[0]
    ).astype(int)

X = df.drop(columns=[target_col])

if X.shape[1] == 0:
    raise ValueError(
        "No predictor columns were found."
    )


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n===== DATA SPLIT =====")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 5. PREPROCESSING DEFINITION
# ============================================================

numeric_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=np.number
).columns.tolist()

numeric_transformer = SklearnPipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = SklearnPipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 6. ORIGINAL CLASS DISTRIBUTION
# ============================================================

original_counts = y.value_counts().reindex(
    [0, 1],
    fill_value=0
)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    ["Non-delinquent", "Delinquent"],
    original_counts.values
)

ax.set_title(
    "Original Customer Class Distribution",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Account Class")
ax.set_ylabel("Number of Records")

ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

ax.bar_label(bars, padding=4)

fig.tight_layout()

fig.savefig(
    os.path.join(
        VISUAL_DIR,
        "class_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 7. DEFINE SMOTE AND MODELS
# ============================================================

minority_count = y_train.value_counts().min()

if minority_count < 2:
    raise ValueError(
        "Not enough minority-class training records for SMOTE."
    )

smote = SMOTE(
    random_state=42,
    k_neighbors=min(5, minority_count - 1)
)

logistic_pipeline = ImbPipeline(
    steps=[
        ("preprocess", clone(preprocessor)),
        ("smote", clone(smote)),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)

random_forest_pipeline = ImbPipeline(
    steps=[
        ("preprocess", clone(preprocessor)),
        ("smote", clone(smote)),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# ============================================================
# 8. BASELINE MODEL
# ============================================================

baseline = DummyClassifier(
    strategy="most_frequent"
)

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)


# ============================================================
# 9. TRAIN MACHINE LEARNING MODELS
# ============================================================

print("\n===== TRAINING LOGISTIC REGRESSION =====")

logistic_pipeline.fit(
    X_train,
    y_train
)

logistic_predictions = logistic_pipeline.predict(
    X_test
)

print("\n===== TRAINING RANDOM FOREST =====")

random_forest_pipeline.fit(
    X_train,
    y_train
)

rf_predictions = random_forest_pipeline.predict(
    X_test
)


# ============================================================
# 10. SMOTE DISTRIBUTION VISUALIZATION
# ============================================================

# Use a separate preprocessing copy for visualization.
# This does not modify the trained model pipelines.

visual_preprocessor = clone(preprocessor)

X_train_processed = visual_preprocessor.fit_transform(
    X_train
)

visual_smote = clone(smote)

X_train_resampled, y_train_resampled = (
    visual_smote.fit_resample(
        X_train_processed,
        y_train
    )
)

before_counts = y_train.value_counts().reindex(
    [0, 1],
    fill_value=0
)

after_counts = pd.Series(
    y_train_resampled
).value_counts().reindex(
    [0, 1],
    fill_value=0
)

labels = [
    "Non-delinquent",
    "Delinquent"
]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

bars_before = ax.bar(
    x - width / 2,
    before_counts.values,
    width,
    label="Before SMOTE"
)

bars_after = ax.bar(
    x + width / 2,
    after_counts.values,
    width,
    label="After SMOTE"
)

ax.set_title(
    "Training Class Distribution Before and After SMOTE",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel("Account Class")
ax.set_ylabel("Number of Training Records")

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.legend()
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

ax.bar_label(bars_before, padding=3)
ax.bar_label(bars_after, padding=3)

fig.tight_layout()

fig.savefig(
    os.path.join(
        VISUAL_DIR,
        "smote_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

predictions = {
    "Baseline Model": baseline_predictions,
    "Logistic Regression": logistic_predictions,
    "Random Forest": rf_predictions
}

results = []

for model_name, prediction in predictions.items():

    results.append({
        "Model": model_name,

        "Accuracy": accuracy_score(
            y_test,
            prediction
        ),

        "Balanced Accuracy": balanced_accuracy_score(
            y_test,
            prediction
        ),

        "Precision": precision_score(
            y_test,
            prediction,
            pos_label=1,
            zero_division=0
        ),

        "Delinquent Recall": recall_score(
            y_test,
            prediction,
            pos_label=1,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            prediction,
            pos_label=1,
            zero_division=0
        )
    })

    print(f"\n===== {model_name} =====")

    print(
        classification_report(
            y_test,
            prediction,
            labels=[0, 1],
            target_names=[
                "Non-delinquent",
                "Delinquent"
            ],
            zero_division=0
        )
    )

results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")

print(
    results_df.round(4).to_string(index=False)
)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_performance.csv"
    ),
    index=False
)


# ============================================================
# 12. MODEL COMPARISON VISUALIZATION
# ============================================================

metric_columns = [
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Delinquent Recall",
    "F1 Score"
]

comparison_df = results_df.set_index(
    "Model"
)[metric_columns]

fig, ax = plt.subplots(figsize=(11, 6))

comparison_df.plot(
    kind="bar",
    ax=ax,
    width=0.8
)

ax.set_title(
    "Credit Risk Model Performance Comparison",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Machine Learning Model")
ax.set_ylabel("Performance Score")

ax.set_ylim(0, 1.05)

ax.legend(
    title="Evaluation Metrics",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

ax.tick_params(
    axis="x",
    rotation=0
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        VISUAL_DIR,
        "model_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 13. RANDOM FOREST CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    rf_predictions,
    labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(7, 5))

image = ax.imshow(cm, interpolation="nearest")

ax.set_title(
    "Random Forest Confusion Matrix",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels([
    "Non-delinquent",
    "Delinquent"
])

ax.set_yticklabels([
    "Non-delinquent",
    "Delinquent"
])

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14
        )

fig.colorbar(
    image,
    ax=ax,
    label="Number of Records"
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        VISUAL_DIR,
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 14. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_model = random_forest_pipeline.named_steps[
    "model"
]

feature_names = (
    random_forest_pipeline
    .named_steps["preprocess"]
    .get_feature_names_out()
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_importance.csv"
    ),
    index=False
)

top_features = importance_df.head(
    15
).sort_values(
    by="Importance",
    ascending=True
)

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(
    top_features["Feature"],
    top_features["Importance"]
)

ax.set_title(
    "Random Forest Feature Importance - Top 15",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Feature Importance")
ax.set_ylabel("Features")

ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)

ax.bar_label(
    bars,
    fmt="%.3f",
    padding=3
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        VISUAL_DIR,
        "feature_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 15. SAVE TRAINED MODEL PIPELINES
# ============================================================

joblib.dump(
    logistic_pipeline,
    os.path.join(
        MODEL_DIR,
        "logistic_model.pkl"
    )
)

joblib.dump(
    random_forest_pipeline,
    os.path.join(
        MODEL_DIR,
        "random_forest_model.pkl"
    )
)


# ============================================================
# 16. SAVE TEST PREDICTIONS
# ============================================================

output = X_test.copy()

output["Actual"] = y_test.values

output["Baseline_Predicted"] = (
    baseline_predictions
)

output["Logistic_Regression_Predicted"] = (
    logistic_predictions
)

output["Random_Forest_Predicted"] = (
    rf_predictions
)

output.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "delinquency_predictions_output.xlsx"
    ),
    index=False
)


# ============================================================
# 17. COMPLETION SUMMARY
# ============================================================

print("\n===== ANALYSIS COMPLETED =====")

print("\nGenerated visualizations:")

visual_files = [
    "class_distribution.png",
    "smote_distribution.png",
    "model_comparison.png",
    "confusion_matrix.png",
    "feature_importance.png"
]

for filename in visual_files:
    print(
        " -",
        os.path.join(VISUAL_DIR, filename)
    )

print("\nGenerated reports:")
print(" - model_performance.csv")
print(" - feature_importance.csv")
print(" - delinquency_predictions_output.xlsx")

print("\nSaved models:")
print(" - logistic_model.pkl")
print(" - random_forest_model.pkl")

print("\nOutput directory:", OUTPUT_DIR)
print("Visual directory:", VISUAL_DIR)
print("Model directory:", MODEL_DIR)

print("\nAll analysis tasks completed successfully.")
