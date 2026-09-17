import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



# ============================================================
# 1. LOAD TITANIC DATASET
# ============================================================

print("=" * 70)
print("TITANIC DATASET - LOADING AND PROFILING")
print("=" * 70)

# The raw Titanic dataset is loaded exactly once.
df = sns.load_dataset("titanic")

print("\nDataset loaded successfully.")


# ============================================================
# 2. SAVE OFFLINE FALLBACK IMMEDIATELY
# ============================================================

df.to_csv("analytics/titanic.csv", index=False)

print("Offline fallback saved to: analytics/titanic.csv")


# ============================================================
# 3. BASIC PROFILING
# ============================================================

print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print(df.shape)


print("\n" + "=" * 70)
print("DATASET INFO")
print("=" * 70)

df.info()


print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(df.describe(include="all"))


# ============================================================
# 4. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE PERCENTAGES")
print("=" * 70)

missing_percent = (df.isnull().mean() * 100)

missing_percent = missing_percent[missing_percent > 0]

if missing_percent.empty:
    print("No missing values found.")
else:
    for column, percentage in missing_percent.items():
        print(f"{column}: {percentage:.2f}%")


# ============================================================
# 5. MISSING VALUE HANDLING
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE HANDLING")
print("=" * 70)

# Store the measured missing percentages before cleaning.
missing_rates = (df.isnull().mean() * 100)

for column, rate in missing_rates.items():

    if rate == 0:
        continue

    print(f"\nColumn: {column}")
    print(f"Missing rate: {rate:.2f}%")

    # Under 5% -> drop rows
    if rate < 5:
        print("Strategy: Drop rows with missing values (<5%).")
        df = df.dropna(subset=[column])

    # 5% to 30% -> impute
    elif rate <= 30:

        if pd.api.types.is_numeric_dtype(df[column]):
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)

            print(
                f"Strategy: Median imputation (5%-30%). "
                f"Median = {median_value:.2f}"
            )

        else:
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)

            print(
                f"Strategy: Mode imputation (5%-30%). "
                f"Mode = {mode_value}"
            )

    # More than 30% -> explicitly decide
    else:
      print(
        "Strategy: Drop column (>30% missing) because "
        "imputation would be unreliable."
        )
      df = df.drop(columns=[column])


# ============================================================
# 6. VERIFY CLEANED DATA
# ============================================================

print("\n" + "=" * 70)
print("DATA AFTER MISSING VALUE HANDLING")
print("=" * 70)

print(f"Shape after cleaning: {df.shape}")

print("\nRemaining missing values:")

remaining_missing = df.isnull().sum()

remaining_missing = remaining_missing[remaining_missing > 0]

if remaining_missing.empty:
    print("No missing values remain.")
else:
    print(remaining_missing)


# ============================================================
# 7. SAVE CLEANED DATA
# ============================================================

df.to_csv("analytics/titanic.csv", index=False)

print("\nCleaned dataset saved to: analytics/titanic.csv")

print("\nEDA Part 1 completed successfully.")

# ============================================================
# 8. UNIVARIATE ANALYSIS - AGE AND FARE
# ============================================================

print("\n" + "=" * 70)
print("UNIVARIATE ANALYSIS - AGE AND FARE")
print("=" * 70)


# ------------------------------------------------------------
# AGE HISTOGRAM
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("analytics/charts/age_histogram.png", dpi=300)
plt.close()


# ------------------------------------------------------------
# AGE BOX PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["age"])
plt.title("Age Box Plot")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig("analytics/charts/age_boxplot.png", dpi=300)
plt.close()


# ------------------------------------------------------------
# FARE HISTOGRAM
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(df["fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("analytics/charts/fare_histogram.png", dpi=300)
plt.close()


# ------------------------------------------------------------
# FARE BOX PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["fare"])
plt.title("Fare Box Plot")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig("analytics/charts/fare_boxplot.png", dpi=300)
plt.close()


# ============================================================
# IQR OUTLIER ANALYSIS
# ============================================================

def calculate_iqr_outliers(series, column_name):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
    ]

    print(f"\n{column_name} IQR Analysis")
    print(f"Q1: {q1:.4f}")
    print(f"Q3: {q3:.4f}")
    print(f"IQR: {iqr:.4f}")
    print(f"Lower bound: {lower_bound:.4f}")
    print(f"Upper bound: {upper_bound:.4f}")
    print(f"Number of outliers: {len(outliers)}")

    return len(outliers)


age_outliers = calculate_iqr_outliers(df["age"], "Age")
fare_outliers = calculate_iqr_outliers(df["fare"], "Fare")


# ============================================================
# FARE STATISTICS
# ============================================================

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode().iloc[0]

print("\n" + "=" * 70)
print("FARE STATISTICS")
print("=" * 70)

print(f"Mean: {fare_mean:.4f}")
print(f"Median: {fare_median:.4f}")
print(f"Mode: {fare_mode:.4f}")


# ============================================================
# FARE SKEWNESS
# ============================================================

fare_skewness = df["fare"].skew()

print(f"Skewness coefficient: {fare_skewness:.4f}")

if fare_mean > fare_median > fare_mode:
    skewness_description = "right-skewed"
elif fare_mean < fare_median < fare_mode:
    skewness_description = "left-skewed"
else:
    skewness_description = "approximately symmetric"

print(f"Fare distribution: {skewness_description}")

print(
    "\nInterpretation: "
    f"The fare mean ({fare_mean:.2f}), median ({fare_median:.2f}), "
    f"and mode ({fare_mode:.2f}) indicate that the distribution is "
    f"{skewness_description}."
)


print("\nAge and Fare univariate analysis completed successfully.")

# ============================================================
# 9. BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("BIVARIATE ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# SURVIVAL RATE BY SEX
# ------------------------------------------------------------

print("\nSurvival Rate by Sex:")

survival_by_sex = (
    df.groupby("sex", observed=True)["survived"]
    .mean()
    .mul(100)
    .round(2)
)

print(survival_by_sex)


# ------------------------------------------------------------
# SURVIVAL RATE BY PCLASS
# ------------------------------------------------------------

print("\nSurvival Rate by Passenger Class:")

survival_by_pclass = (
    df.groupby("pclass")["survived"]
    .mean()
    .mul(100)
    .round(2)
)

print(survival_by_pclass)


# ------------------------------------------------------------
# SURVIVAL RATE BY SEX AND PCLASS
# USING BOOLEAN MASKING
# ------------------------------------------------------------

print("\nSurvival Rate by Sex and Passenger Class:")

groups = [
    ("Female, Class 1", (df["sex"] == "female") & (df["pclass"] == 1)),
    ("Female, Class 2", (df["sex"] == "female") & (df["pclass"] == 2)),
    ("Female, Class 3", (df["sex"] == "female") & (df["pclass"] == 3)),
    ("Male, Class 1", (df["sex"] == "male") & (df["pclass"] == 1)),
    ("Male, Class 2", (df["sex"] == "male") & (df["pclass"] == 2)),
    ("Male, Class 3", (df["sex"] == "male") & (df["pclass"] == 3)),
]

survival_by_sex_pclass = {}

for name, mask in groups:
    rate = df.loc[mask, "survived"].mean() * 100
    survival_by_sex_pclass[name] = round(rate, 2)

    print(f"{name}: {rate:.2f}%")


# ------------------------------------------------------------
# CORRELATION MATRIX
# EXACTLY SIX REQUIRED COLUMNS
# ------------------------------------------------------------

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = df[correlation_columns].corr()

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

print(correlation_matrix.round(4))


# ------------------------------------------------------------
# FIND TWO STRONGEST CORRELATIONS
# ------------------------------------------------------------

correlation_pairs = []

for i in range(len(correlation_columns)):
    for j in range(i + 1, len(correlation_columns)):

        col1 = correlation_columns[i]
        col2 = correlation_columns[j]

        coefficient = correlation_matrix.loc[col1, col2]

        correlation_pairs.append(
            (col1, col2, coefficient, abs(coefficient))
        )


correlation_pairs.sort(
    key=lambda x: x[3],
    reverse=True
)

top_two_correlations = correlation_pairs[:2]

print("\nTwo strongest correlations:")

for rank, (col1, col2, coefficient, absolute_value) in enumerate(
    top_two_correlations,
    start=1
):
    print(
        f"{rank}. {col1} vs {col2}: "
        f"correlation = {coefficient:.4f}, "
        f"|correlation| = {absolute_value:.4f}"
    )


# ------------------------------------------------------------
# CORRELATION HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title("Titanic Numeric Feature Correlation Matrix")
plt.tight_layout()

plt.savefig(
    "analytics/charts/correlation_heatmap.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# CORRELATION INTERPRETATION
# ------------------------------------------------------------

print("\nCorrelation Interpretation:")

for col1, col2, coefficient, absolute_value in top_two_correlations:

    direction = "positive" if coefficient > 0 else "negative"

    print(
        f"- {col1} and {col2} have the strongest absolute "
        f"correlation ({coefficient:.4f}), indicating a "
        f"{direction} relationship."
    )


print("\nBivariate analysis completed successfully.")

# =====================================================================
# MULTIVARIATE ANALYSIS
# =====================================================================

print("\n" + "=" * 70)
print("MULTIVARIATE ANALYSIS")
print("=" * 70)

# Chart 1: Survival Rate by Sex and Passenger Class
plt.figure(figsize=(8, 6))

sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title("Survival Rate by Passenger Class and Sex")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("analytics/charts/survival_by_sex_class.png")
plt.close()

print("\nChart 1 saved: survival_by_sex_class.png")

print("""
Interpretation:
The chart shows that survival rates varied substantially by both sex and passenger class.
Female passengers generally had higher survival rates than male passengers within the same class.
Survival was also higher in first class than in second and third class for both sexes.
""")

# Chart 2: Age vs Fare colored by Survival
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    style="sex",
    alpha=0.7
)

plt.title("Age vs Fare by Survival and Sex")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.legend(title="Survival / Sex")

plt.tight_layout()
plt.savefig("analytics/charts/age_fare_survival.png")
plt.close()

print("\nChart 2 saved: age_fare_survival.png")

print("""
Interpretation:
The scatter plot examines the relationship between passenger age and fare while distinguishing survivors from non-survivors.
Survivors appear across different age and fare ranges, while higher fares are concentrated among passengers paying more expensive tickets.
The additional sex distinction helps show how survival patterns varied across passengers with different ages and fares.
""")

# Chart 3: Family Size vs Survival by Passenger Class

df["family_size"] = df["sibsp"] + df["parch"] + 1

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="family_size",
    y="survived",
    hue="pclass"
)

plt.title("Survival Rate by Family Size and Passenger Class")
plt.xlabel("Family Size")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("analytics/charts/family_size_class_survival.png")
plt.close()

print("\nChart 3 saved: family_size_class_survival.png")

print("""
Interpretation:
The chart compares survival rates across different family sizes while separating passengers by passenger class.
Survival patterns vary across family sizes and passenger classes, showing that family structure and class can jointly describe differences in survival.
Very large family groups have fewer observations, so their survival rates should be interpreted cautiously.
""")

# Chart 4: Survival by Embarkation Port, Sex and Passenger Class

plt.figure(figsize=(11, 6))

sns.barplot(
    data=df,
    x="embarked",
    y="survived",
    hue="sex",
    errorbar=None
)

plt.title("Survival Rate by Embarkation Port and Sex")
plt.xlabel("Port of Embarkation")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("analytics/charts/survival_by_embarked_sex.png")
plt.close()

print("\nChart 4 saved: survival_by_embarked_sex.png")

print("""
Interpretation:
The chart compares survival rates across embarkation ports while separating passengers by sex.
Survival rates differ between male and female passengers across the three embarkation ports.
These differences should be interpreted together with passenger class because passenger composition was not identical across ports.
""")

# =====================================================================
# STANDARDIZATION SANITY CHECK
# =====================================================================

print("\n" + "=" * 70)
print("STANDARDIZATION SANITY CHECK")
print("=" * 70)

# Keep original values for comparison
age_mean_before = df["age"].mean()
age_std_before = df["age"].std()

fare_mean_before = df["fare"].mean()
fare_std_before = df["fare"].std()

# Z-score standardization
df["age_standardized"] = (
    (df["age"] - age_mean_before) / age_std_before
)

df["fare_standardized"] = (
    (df["fare"] - fare_mean_before) / fare_std_before
)

# Calculate after-standardization statistics
age_mean_after = df["age_standardized"].mean()
age_std_after = df["age_standardized"].std()

fare_mean_after = df["fare_standardized"].mean()
fare_std_after = df["fare_standardized"].std()

print("\nAGE")
print(f"Before standardization - Mean: {age_mean_before:.4f}")
print(f"Before standardization - Std:  {age_std_before:.4f}")
print(f"After standardization  - Mean: {age_mean_after:.4f}")
print(f"After standardization  - Std:  {age_std_after:.4f}")

print("\nFARE")
print(f"Before standardization - Mean: {fare_mean_before:.4f}")
print(f"Before standardization - Std:  {fare_std_before:.4f}")
print(f"After standardization  - Mean: {fare_mean_after:.4f}")
print(f"After standardization  - Std:  {fare_std_after:.4f}")

print("""
Interpretation:
Z-score standardization transforms the Age and Fare variables to approximately zero mean and unit standard deviation.
The standardized values are suitable for comparing variables measured on different scales.
This transformation is used here only as an EDA sanity check and is not used to fit the final machine-learning model.
""")

print("\nEDA MODULE COMPLETED SUCCESSFULLY.")