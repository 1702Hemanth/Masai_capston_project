# =====================================================================
# TITANIC DATASET - MACHINE LEARNING MODELING
# =====================================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
# =====================================================================
# LOAD CLEANED DATA
# =====================================================================

print("=" * 70)
print("TITANIC DATASET - MODELING")
print("=" * 70)

df = pd.read_csv("analytics/titanic.csv")

print("\nCleaned dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

# =====================================================================
# DEFINE FEATURES AND TARGET
# =====================================================================

target = "survived"

X = df.drop(columns=[target])
y = df[target]

print("\nTarget variable:", target)
print("Features:", list(X.columns))

print("\nTarget class distribution:")
print(y.value_counts())
print("\nTarget class percentages:")
print((y.value_counts(normalize=True) * 100).round(2))

# =====================================================================
# STRATIFIED TRAIN-TEST SPLIT
# =====================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("STRATIFIED TRAIN-TEST SPLIT")
print("=" * 70)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nTraining target percentages:")
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nTesting target percentages:")
print((y_test.value_counts(normalize=True) * 100).round(2))

print("""
Justification:
Stratification is used to preserve approximately the same proportion
of survived and non-survived passengers in both the training and testing
sets. This is important because the Titanic target variable contains
two classes with an unequal distribution.
""")

print("\nStep 1 completed successfully.")

# =====================================================================
# PREPROCESSING PIPELINE
# =====================================================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

print("\n" + "=" * 70)
print("PREPROCESSING PIPELINE")
print("=" * 70)

# Remove target leakage and redundant derived columns
features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features]
y = df[target]

# Repeat the split using only selected features
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Numerical and categorical columns
numeric_features = [
    "age",
    "fare",
    "pclass",
    "sibsp",
    "parch"
]

categorical_features = [
    "sex",
    "embarked"
]

# Numerical preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combined preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nSelected features:")
print(features)

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

print("""
Preprocessing design:
Numerical features use median imputation followed by StandardScaler.
Categorical features use most-frequent imputation followed by one-hot encoding.
The preprocessing transformer is fitted only on the training data through
the machine-learning pipelines.
""")

print("\nStep 2 completed successfully.")

# =====================================================================
# CLASSIFICATION MODELS
# =====================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

print("\n" + "=" * 70)
print("CLASSIFICATION MODELS")
print("=" * 70)

# ---------------------------------------------------------------------
# 1. Logistic Regression
# ---------------------------------------------------------------------

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ]
)

logistic_pipeline.fit(X_train, y_train)

print("\nLogistic Regression trained successfully.")

# ---------------------------------------------------------------------
# 2. Decision Tree
# ---------------------------------------------------------------------

decision_tree_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
        )
    ]
)

decision_tree_pipeline.fit(X_train, y_train)

print("Decision Tree trained successfully.")

# ---------------------------------------------------------------------
# 3. Random Forest
# ---------------------------------------------------------------------

random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

random_forest_pipeline.fit(X_train, y_train)

print("Random Forest trained successfully.")

# =====================================================================
# DECISION TREE VISUALIZATION
# =====================================================================

print("\nCreating Decision Tree visualization...")

# Transform training data using fitted preprocessing
X_train_transformed = (
    decision_tree_pipeline
    .named_steps["preprocessor"]
    .transform(X_train)
)

# Get transformed feature names
feature_names = (
    decision_tree_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

tree_model = decision_tree_pipeline.named_steps["classifier"]

plt.figure(figsize=(22, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Classifier")
plt.tight_layout()

plt.savefig(
    "analytics/charts/decision_tree.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Decision Tree visualization saved: decision_tree.png")

print("\nAll three classification models trained successfully.")
print("Step 3 completed successfully.")

# =====================================================================
# MODEL EVALUATION
# =====================================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

models = {
    "Logistic Regression": logistic_pipeline,
    "Decision Tree": decision_tree_pipeline,
    "Random Forest": random_forest_pipeline
}

results = []

plt.figure(figsize=(9, 7))

for model_name, model in models.items():

    # Predictions
    y_pred = model.predict(X_test)

    # Probability predictions for ROC/AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": auc
    })

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc:.3f})"
    )

# ROC reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Classification Models")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "analytics/charts/roc_curves.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# =====================================================================
# COMPARISON TABLE
# =====================================================================

comparison_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 70)

print("\n")
print(comparison_df.round(4).to_string(index=False))

# Save comparison table
comparison_df.to_csv(
    "analytics/model/classification_comparison.csv",
    index=False
)

print("\nComparison table saved:")
print("analytics/model/classification_comparison.csv")

print("\nROC curves saved:")
print("analytics/charts/roc_curves.png")

print("\nStep 4 completed successfully.")

# =====================================================================
# CLASS IMBALANCE ANALYSIS
# =====================================================================

print("\n" + "=" * 70)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 70)

print("\nClass distribution:")
class_counts = y_train.value_counts().sort_index()
print(class_counts)

print("\nClass percentages:")
class_percentages = (
    y_train.value_counts(normalize=True)
    .sort_index()
    * 100
)
print(class_percentages.round(2))

# ---------------------------------------------------------------------
# Baseline: Always predict the majority class
# ---------------------------------------------------------------------

majority_class = y_train.mode()[0]

baseline_predictions = [majority_class] * len(y_test)

baseline_precision = precision_score(
    y_test,
    baseline_predictions,
    zero_division=0
)

baseline_recall = recall_score(
    y_test,
    baseline_predictions,
    zero_division=0
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    zero_division=0
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print("\n" + "-" * 60)
print("MAJORITY-CLASS BASELINE")
print("-" * 60)

print(f"\nMajority class: {majority_class}")
print(f"Accuracy : {baseline_accuracy:.4f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall   : {baseline_recall:.4f}")
print(f"F1 Score : {baseline_f1:.4f}")

# ---------------------------------------------------------------------
# Balanced Logistic Regression
# ---------------------------------------------------------------------

balanced_logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

balanced_logistic_pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------------
# SMOTE Logistic Regression
# ---------------------------------------------------------------------

smote_logistic_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

smote_logistic_pipeline.fit(X_train, y_train)

smote_predictions = smote_logistic_pipeline.predict(X_test)

smote_precision = precision_score(
    y_test,
    smote_predictions,
    zero_division=0
)

smote_recall = recall_score(
    y_test,
    smote_predictions,
    zero_division=0
)

smote_f1 = f1_score(
    y_test,
    smote_predictions,
    zero_division=0
)

smote_accuracy = accuracy_score(
    y_test,
    smote_predictions
)

smote_probability = smote_logistic_pipeline.predict_proba(X_test)[:, 1]

smote_auc = roc_auc_score(
    y_test,
    smote_probability
)

print("\n" + "-" * 60)
print("SMOTE LOGISTIC REGRESSION")
print("-" * 60)

print(f"\nAccuracy : {smote_accuracy:.4f}")
print(f"Precision: {smote_precision:.4f}")
print(f"Recall   : {smote_recall:.4f}")
print(f"F1 Score : {smote_f1:.4f}")
print(f"ROC-AUC  : {smote_auc:.4f}")

balanced_predictions = balanced_logistic_pipeline.predict(X_test)

balanced_precision = precision_score(
    y_test,
    balanced_predictions
)

balanced_recall = recall_score(
    y_test,
    balanced_predictions
)

balanced_f1 = f1_score(
    y_test,
    balanced_predictions
)

balanced_accuracy = accuracy_score(
    y_test,
    balanced_predictions
)

print("\n" + "-" * 60)
print("BALANCED LOGISTIC REGRESSION")
print("-" * 60)

print(f"\nAccuracy : {balanced_accuracy:.4f}")
print(f"Precision: {balanced_precision:.4f}")
print(f"Recall   : {balanced_recall:.4f}")
print(f"F1 Score : {balanced_f1:.4f}")

# ---------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------

imbalance_results = pd.DataFrame([
    {
        "Model": "Majority Baseline",
        "Accuracy": baseline_accuracy,
        "Precision": baseline_precision,
        "Recall": baseline_recall,
        "F1": baseline_f1
    },
    {
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(
            y_test,
            logistic_pipeline.predict(X_test)
        ),
        "Precision": precision_score(
            y_test,
            logistic_pipeline.predict(X_test)
        ),
        "Recall": recall_score(
            y_test,
            logistic_pipeline.predict(X_test)
        ),
        "F1": f1_score(
            y_test,
            logistic_pipeline.predict(X_test)
        )
    },
    {
        "Model": "Balanced Logistic Regression",
        "Accuracy": balanced_accuracy,
        "Precision": balanced_precision,
        "Recall": balanced_recall,
        "F1": balanced_f1
    },
    {
    "Model": "SMOTE Logistic Regression",
    "Accuracy": smote_accuracy,
    "Precision": smote_precision,
    "Recall": smote_recall,
    "F1": smote_f1
}
])

print("\n" + "=" * 70)
print("CLASS IMBALANCE COMPARISON")
print("=" * 70)

print("\n")
print(imbalance_results.round(4).to_string(index=False))

imbalance_results.to_csv(
    "analytics/model/class_imbalance_comparison.csv",
    index=False
)

print("\nComparison saved:")
print("analytics/model/class_imbalance_comparison.csv")

print("""
Conclusion:
The Titanic target has an unequal distribution between the two classes.
The majority-class baseline provides a reference point for evaluating
whether the trained classifier learns useful patterns beyond simply
predicting the majority class.
The balanced Logistic Regression gives additional weight to the minority
class, which can change the precision, recall and F1-score trade-off.
The final conclusion should be based on the measured test-set metrics above.
""")

print("\nStep 5 completed successfully.")

# =====================================================================
# RANDOM FOREST GRID SEARCH WITH OOB SCORE
# =====================================================================

from sklearn.model_selection import GridSearchCV

print("\n" + "=" * 70)
print("RANDOM FOREST GRID SEARCH")
print("=" * 70)

# Random Forest with OOB scoring enabled
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                oob_score=True,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# Hyperparameter grid
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10],
    "classifier__max_features": ["sqrt", "log2"]
}

print("\nStarting GridSearchCV...")
print("Parameters being tuned:")
print("- n_estimators")
print("- max_depth")
print("- max_features")

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nGridSearchCV completed successfully.")

print("\nBest Parameters:")
print(grid_search.best_params_)

print(f"\nBest Cross-Validation F1 Score: {grid_search.best_score_:.4f}")

# Extract best fitted Random Forest
best_rf_pipeline = grid_search.best_estimator_

best_rf_model = best_rf_pipeline.named_steps["classifier"]

print(f"\nBest Random Forest OOB Score: {best_rf_model.oob_score_:.4f}")

# Test-set evaluation of tuned model
best_rf_predictions = best_rf_pipeline.predict(X_test)

best_rf_precision = precision_score(
    y_test,
    best_rf_predictions
)

best_rf_recall = recall_score(
    y_test,
    best_rf_predictions
)

best_rf_f1 = f1_score(
    y_test,
    best_rf_predictions
)

best_rf_accuracy = accuracy_score(
    y_test,
    best_rf_predictions
)

best_rf_prob = best_rf_pipeline.predict_proba(X_test)[:, 1]

best_rf_auc = roc_auc_score(
    y_test,
    best_rf_prob
)

print("\nTuned Random Forest Test Results:")
print(f"Accuracy : {best_rf_accuracy:.4f}")
print(f"Precision: {best_rf_precision:.4f}")
print(f"Recall   : {best_rf_recall:.4f}")
print(f"F1 Score : {best_rf_f1:.4f}")
print(f"ROC-AUC  : {best_rf_auc:.4f}")

print("\nStep 6 completed successfully.")

# =====================================================================
# MULTIVARIATE LINEAR REGRESSION - PREDICTING FARE
# =====================================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\n" + "=" * 70)
print("MULTIVARIATE LINEAR REGRESSION - FARE")
print("=" * 70)

# Features used to predict fare
regression_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "sex",
    "embarked"
]

X_reg = df[regression_features]
y_reg = df["fare"]

# Train-test split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

# Regression preprocessing
regression_numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]

regression_categorical_features = [
    "sex",
    "embarked"
]

regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            ),
            regression_numeric_features
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            ),
            regression_categorical_features
        )
    ]
)

# Regression pipeline
regression_pipeline = Pipeline(
    steps=[
        ("preprocessor", regression_preprocessor),
        ("regressor", LinearRegression())
    ]
)

# Train
regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

# Predictions
y_reg_pred = regression_pipeline.predict(X_reg_test)

# Metrics
mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)

# Number of observations and predictors
n = len(y_reg_test)

X_reg_test_transformed = regression_pipeline.named_steps[
    "preprocessor"
].transform(X_reg_test)

p = X_reg_test_transformed.shape[1]

# Adjusted R²
adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - p - 1)
)

print("\nRegression features:")
print(regression_features)

print("\nRegression results:")
print(f"MAE        : {mae:.4f}")
print(f"RMSE       : {rmse:.4f}")
print(f"R²         : {r2:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")

# =====================================================================
# RESIDUAL ANALYSIS
# =====================================================================

residuals = y_reg_test - y_reg_pred

plt.figure(figsize=(9, 6))

plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot - Fare Regression")

plt.tight_layout()

plt.savefig(
    "analytics/charts/fare_regression_residuals.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nResidual plot saved:")
print("analytics/charts/fare_regression_residuals.png")

print("\nResidual interpretation:")
print(
    "The residuals show an increasing spread as predicted fare increases, "
    "with several large positive residuals at higher predicted fares. "
    "This funnel-like pattern provides visual evidence of heteroscedasticity, "
    "meaning that the variance of the prediction errors is not constant."
)

print("\nStep 7 completed successfully.")

# =====================================================================
# FINAL MODEL COMPARISON
# =====================================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

# Classification results
classification_final = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(
            y_test,
            logistic_pipeline.predict(X_test)
        ),
        "Precision": precision_score(
            y_test,
            logistic_pipeline.predict(X_test)
        ),
        "Recall": recall_score(
            y_test,
            logistic_pipeline.predict(X_test)
        ),
        "F1": f1_score(
            y_test,
            logistic_pipeline.predict(X_test)
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            logistic_pipeline.predict_proba(X_test)[:, 1]
        )
    },
    {
        "Model": "Decision Tree",
        "Accuracy": accuracy_score(
            y_test,
            decision_tree_pipeline.predict(X_test)
        ),
        "Precision": precision_score(
            y_test,
            decision_tree_pipeline.predict(X_test)
        ),
        "Recall": recall_score(
            y_test,
            decision_tree_pipeline.predict(X_test)
        ),
        "F1": f1_score(
            y_test,
            decision_tree_pipeline.predict(X_test)
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            decision_tree_pipeline.predict_proba(X_test)[:, 1]
        )
    },
    {
        "Model": "Random Forest",
        "Accuracy": accuracy_score(
            y_test,
            random_forest_pipeline.predict(X_test)
        ),
        "Precision": precision_score(
            y_test,
            random_forest_pipeline.predict(X_test)
        ),
        "Recall": recall_score(
            y_test,
            random_forest_pipeline.predict(X_test)
        ),
        "F1": f1_score(
            y_test,
            random_forest_pipeline.predict(X_test)
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            random_forest_pipeline.predict_proba(X_test)[:, 1]
        )
    },
    {
        "Model": "Tuned Random Forest",
        "Accuracy": best_rf_accuracy,
        "Precision": best_rf_precision,
        "Recall": best_rf_recall,
        "F1": best_rf_f1,
        "ROC-AUC": best_rf_auc
    }
])

print("\nCLASSIFICATION METRICS")
print("-" * 70)
print(classification_final.round(4).to_string(index=False))

# Regression results
regression_final = pd.DataFrame([
    {
        "Model": "Multivariate Linear Regression",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adjusted_r2
    }
])

print("\nREGRESSION METRICS")
print("-" * 70)
print(regression_final.round(4).to_string(index=False))

# Save final tables
classification_final.to_csv(
    "analytics/model/final_classification_results.csv",
    index=False
)

regression_final.to_csv(
    "analytics/model/regression_results.csv",
    index=False
)

print("\nSaved:")
print("analytics/model/final_classification_results.csv")
print("analytics/model/regression_results.csv")

print("\nStep 8 completed successfully.")

# ==============================================================
# STEP 9: SAVE AND RELOAD COMPLETE PIPELINE
# ==============================================================

import joblib

print("\n" + "=" * 70)
print("SAVE AND RELOAD FINAL PIPELINE")
print("=" * 70)

# Use the tuned Random Forest pipeline as the final fitted pipeline
full_pipeline = best_rf_pipeline

pipeline_path = "analytics/model/titanic_survival_pipeline.joblib"

# Save complete preprocessing + estimator pipeline
joblib.dump(full_pipeline, pipeline_path)

print(f"\nPipeline saved successfully:")
print(pipeline_path)

# Reload the complete pipeline
loaded_pipeline = joblib.load(pipeline_path)

print("Pipeline reloaded successfully.")

# Raw passenger input
raw_passenger = pd.DataFrame({
    "pclass": [3],
    "sex": ["female"],
    "age": [25],
    "sibsp": [0],
    "parch": [0],
    "fare": [15.0],
    "embarked": ["S"]
})

# Prediction directly from raw data
prediction = loaded_pipeline.predict(raw_passenger)
prediction_probability = loaded_pipeline.predict_proba(raw_passenger)[0, 1]

print("\nRaw input passenger:")
print(raw_passenger)

print("\nPrediction:")
print("Survived" if prediction[0] == 1 else "Did Not Survive")

print(f"Survival probability: {prediction_probability:.4f}")

print("\nStep 9 completed successfully.")