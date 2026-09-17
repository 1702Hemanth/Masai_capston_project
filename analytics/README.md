# Module 2 — Titanic Analytics and Modeling

## 1. Objective

This module performs exploratory data analysis (EDA), data cleaning,
classification modeling, class-imbalance analysis, hyperparameter tuning,
and multivariate regression using the classic Titanic dataset.

The module covers:

- Data profiling and cleaning
- Missing-value analysis
- Univariate and bivariate analysis
- Multivariate visualization
- Correlation analysis
- Feature standardization sanity check
- Classification modeling
- Class-imbalance analysis
- Random Forest hyperparameter tuning
- Multivariate linear regression
- Model comparison
- Saving and reloading a complete machine-learning pipeline

---

## 2. Dataset Loading

The Titanic dataset is loaded using Seaborn:

```python
df = sns.load_dataset("titanic")
The dataset is loaded once and immediately saved as:
analytics/titanic.csv
Subsequent modeling uses the cleaned Titanic CSV rather than loading the
dataset again.
The original dataset contains 891 rows and 15 columns.
After cleaning, the final dataset contains:
Rows    : 889
Columns : 14
Part A — Exploratory Data Analysis

3. Data Profiling

The following profiling information is generated:
- df.info()
- df.describe()
- df.shape
- Missing-value percentages for affected columns
The target variable is:
survived
Class distribution after cleaning:
Class	Count	Percentage
0 — Not Survived	549	61.75%
1 — Survived	340	38.25%


The target variable is therefore moderately imbalanced, with non-survivors
forming the larger class.
4. Data Cleaning
Missing values were handled according to the percentage of missing values.
Age
Approximately 19.87% of Age values were missing.
Since the missing percentage falls between 5% and 30%, missing Age values
were imputed using the median.
Age median = 28.00
Embarked
Approximately 0.22% of Embarked values were missing.
Because the missing percentage is below 5%, the affected rows were removed.
Embark Town
Approximately 0.22% of Embark Town values were missing.
The affected rows were removed because the missing percentage was below 5%.
Deck
Approximately 77.22% of Deck values were missing.
Because more than 30% of the values were missing, the Deck column was
dropped rather than attempting unreliable imputation.
Final Dataset
After cleaning:
Final shape = (889, 14)
Missing values = 0
5. Age and Fare Analysis
Age Distribution
An age histogram and boxplot were generated.
The Age distribution is concentrated mainly around the adult passenger
range, while the boxplot identifies observations outside the IQR-based
range.
Age IQR
Q1       = 22.0000
Q3       = 35.0000
IQR      = 13.0000
Lower    = 2.5000
Upper    = 54.5000
Outliers = 65
The IQR method identifies 65 observations outside the lower and upper
bounds. These observations were retained because they represent legitimate
passenger ages rather than automatically being treated as invalid data.
Fare Distribution
The Fare histogram shows a strongly concentrated distribution at lower
fares with a long right tail.
The Fare boxplot also identifies a substantial number of high-fare
observations.
Fare IQR
Q1       = 7.8958
Q3       = 31.0000
IQR      = 23.1042
Lower    = -26.7605
Upper    = 65.6563
Outliers = 114
The negative lower bound is a mathematical result of the IQR method.
Since Fare cannot logically be negative, the important practical finding
is the presence of high-fare observations above the upper bound.
Fare Summary Statistics
Mean   = 32.0967
Median = 14.4542
Mode   = 8.0500
The ordering:
Mean > Median > Mode
indicates a positively/right-skewed Fare distribution.
The calculated skewness coefficient is:
4.8014
This confirms strong positive skewness.
6. Survival Analysis
Survival Rate by Sex
The survival rates were calculated using boolean masking.
Female = 74.04%
Male   = 18.89%
The observed survival rate is substantially different between the two
sex groups in this dataset.
Survival Rate by Passenger Class
Class 1 = 62.62%
Class 2 = 47.28%
Class 3 = 24.24%
The observed survival rate decreases across passenger classes from
first class to third class.
Survival Rate by Sex and Passenger Class
Sex	Class	Survival Rate
Female	1	96.74%
Female	2	92.11%
Female	3	50.00%
Male	1	36.89%
Male	2	15.74%
Male	3	13.54%


The combination of sex and passenger class provides more detailed
information than either variable alone. Female passengers have higher
observed survival rates within each passenger class, while third-class
passengers have lower survival rates than passengers in higher classes.
7. Correlation Analysis
The required six numerical variables were used:
survived
pclass
age
sibsp
parch
fare
The resulting correlation matrix is:
	survived	pclass	age	sibsp	parch	fare
survived	1.0000	-0.3355	-0.0698	-0.0340	0.0832	0.2553
pclass	-0.3355	1.0000	-0.3365	0.0817	0.0168	-0.5482
age	-0.0698	-0.3365	1.0000	-0.2325	-0.1715	0.0937
sibsp	-0.0340	0.0817	-0.2325	1.0000	0.4145	0.1609
parch	0.0832	0.0168	-0.1715	0.4145	1.0000	0.2175
fare	0.2553	-0.5482	0.0937	0.1609	0.2175	1.0000


Strongest Absolute Correlations
The two strongest off-diagonal correlations are:
1. pclass and fare: -0.5482
2. sibsp and parch: +0.4145
The negative pclass-fare correlation indicates that passenger class and fare
are inversely associated in this dataset: higher numerical pclass values
represent lower passenger classes and are generally associated with lower
fares.
The positive sibsp-parch correlation indicates that passengers travelling
with siblings/spouses tended also to have parents/children recorded in the
dataset.
A correlation heatmap is saved as:
analytics/charts/correlation_heatmap.png
8. Multivariate Visualizations
Four distinct multivariate charts were generated.
8.1 Survival by Sex and Passenger Class
File:
analytics/charts/survival_by_sex_class.png
The visualization shows survival differences across both sex and passenger
class. Female passengers show higher observed survival rates within each
class, while third-class passengers generally show lower survival rates.
8.2 Age, Fare and Survival
File:
analytics/charts/age_fare_survival.png
This visualization examines the relationship between Age and Fare while
using survival and sex as additional dimensions. Surviving passengers are
distributed across different age and fare levels, while high-fare
observations include several passengers with substantially different
survival outcomes.
8.3 Family Size, Class and Survival
File:
analytics/charts/family_size_class_survival.png
Family size was calculated as:
family_size = sibsp + parch + 1
The chart compares survival across family size and passenger class.
Survival patterns vary with both family size and class, showing that
considering multiple passenger characteristics provides more information
than examining a single variable independently.
8.4 Survival by Embarked Port and Sex
File:
analytics/charts/survival_by_embarked_sex.png
The visualization compares survival across embarkation ports and sex.
Differences between male and female survival rates remain visible across
the embarkation groups, demonstrating interaction between categorical
passenger characteristics.
9. Standardization Sanity Check
Age and Fare were standardized using the z-score transformation as an
EDA-only sanity check.
Age
Before standardization:
Mean = 29.3152
Std  = 12.9849
After standardization:
Mean = 0.0000
Std  = 1.0000
Fare
Before standardization:
Mean = 32.0967
Std  = 49.6975
After standardization:
Mean = 0.0000
Std  = 1.0000
This confirms that z-score standardization centers the variables around
zero and scales their standard deviation to approximately one.
This transformation is an EDA sanity check. The final machine-learning
models perform their own preprocessing through the training pipeline.
Part B — Machine Learning Modeling
10. Feature Selection
The classification target is:
survived
The following features were selected:
pclass
sex
age
sibsp
parch
fare
embarked
Redundant or leakage-prone columns such as alive were excluded from the
modeling features.
11. Stratified Train-Test Split
The data was divided into:
Training samples = 711
Testing samples  = 178
A stratified 80/20 split was used with:
random_state = 42
Training distribution:
Not Survived = 439 (61.74%)
Survived     = 272 (38.26%)
Testing distribution:
Not Survived = 110 (61.80%)
Survived     = 68  (38.20%)
Stratification preserves approximately the same class proportions in the
training and testing sets. This is important because the target contains
two classes with an unequal distribution.
12. Preprocessing Pipeline
Numerical features:
age
fare
pclass
sibsp
parch
Numerical preprocessing:
Median Imputation
        ↓
StandardScaler
Categorical features:
sex
embarked
Categorical preprocessing:
Most-Frequent Imputation
        ↓
One-Hot Encoding
A ColumnTransformer and machine-learning Pipeline were used so that
preprocessing is fitted using training data only.
This avoids data leakage from the test set.
13. Classification Models
Three initial classification models were trained:
1. Logistic Regression
2. Decision Tree
3. Random Forest
A Decision Tree visualization was generated with feature names and class
names.
File:
analytics/charts/decision_tree.png
14. Classification Model Evaluation
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
Classification Comparison
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	0.8090	0.7833	0.6912	0.7344	0.8610
Decision Tree	0.7472	0.7347	0.5294	0.6154	0.8259
Random Forest	0.7978	0.7581	0.6912	0.7231	0.8212
Tuned Random Forest	0.8146	0.7869	0.7059	0.7442	0.8283


ROC curves are saved as:
analytics/charts/roc_curves.png
15. Class Imbalance Analysis
The training dataset contains:
Not Survived = 61.74%
Survived     = 38.26%
A majority-class baseline was created.
Majority-Class Baseline
Accuracy  = 0.6180
Precision = 0.0000
Recall    = 0.0000
F1 Score  = 0.0000
The baseline demonstrates the performance obtained by always predicting
the majority class.
Balanced Logistic Regression
A Logistic Regression model using:
class_weight="balanced"
was also trained.
Results:
Accuracy  = 0.7921
Precision = 0.7183
Recall    = 0.7500
F1 Score  = 0.7338
Compared with the regular Logistic Regression, the balanced model increases
recall from 0.6912 to 0.7500 while reducing precision from 0.7833 to
0.7183. The F1 score remains very similar, changing from 0.7344 to 0.7338.
This demonstrates the trade-off created by giving additional weight to the
minority class.
Comparison file:
analytics/model/class_imbalance_comparison.csv
16. Random Forest GridSearchCV
GridSearchCV was used to tune the Random Forest.
Parameters searched:
n_estimators
max_depth
max_features
The search used:
5-fold cross-validation
Scoring = F1
18 parameter combinations
90 total fits
Best Parameters
n_estimators = 300
max_depth    = 10
max_features = sqrt
Cross-Validation
Best CV F1 = 0.7464
Out-of-Bag Score
OOB Score = 0.8340
Tuned Random Forest Test Results
Accuracy  = 0.8146
Precision = 0.7869
Recall    = 0.7059
F1 Score  = 0.7442
ROC-AUC   = 0.8283
The tuned Random Forest improves the test accuracy and F1 score compared
with the untuned Random Forest.
17. Multivariate Linear Regression — Fare
A multivariate linear regression model was developed to predict Fare.
Features:
pclass
age
sibsp
parch
sex
embarked
The model was evaluated using:
- MAE
- RMSE
- R²
- Adjusted R²
Regression Results
Model	MAE	RMSE	R²	Adjusted R²
Multivariate Linear Regression	21.1386	41.7465	0.3468	0.3118


The R² value of 0.3468 indicates that the model explains approximately
34.68% of the variation in Fare on the test data.
The adjusted R² is 0.3118 after accounting for the number of predictors.
18. Residual Analysis and Heteroscedasticity
Residual plot:
analytics/charts/fare_regression_residuals.png
The residuals show an increasing spread as predicted Fare increases, with
several large positive residuals at higher predicted Fare values.
The funnel-like pattern provides visual evidence of heteroscedasticity,
meaning that the variance of the regression errors is not constant.
19. Final Model Comparison
Classification
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	0.8090	0.7833	0.6912	0.7344	0.8610
Decision Tree	0.7472	0.7347	0.5294	0.6154	0.8259
Random Forest	0.7978	0.7581	0.6912	0.7231	0.8212
Tuned Random Forest	0.8146	0.7869	0.7059	0.7442	0.8283


Regression
Model	MAE	RMSE	R²	Adjusted R²
Multivariate Linear Regression	21.1386	41.7465	0.3468	0.3118


20. Final Classifier Recommendation
Based on the measured test-set results, the Tuned Random Forest is selected
as the final classifier for this project. It achieved an accuracy of
0.8146, precision of 0.7869, recall of 0.7059, and F1 score of 0.7442.
Logistic Regression produced the highest ROC-AUC of 0.8610 and a similar
F1 score of 0.7344, making it a strong alternative when probability
discrimination is emphasized. The Tuned Random Forest was selected for the
saved deployment pipeline because its test accuracy and F1 score were
slightly higher than the other evaluated classifiers.
21. Pipeline Saving and Reloading
The complete fitted preprocessing and Tuned Random Forest pipeline is
saved using joblib.
Saved pipeline:
analytics/model/titanic_survival_pipeline.joblib
The pipeline was successfully reloaded and tested using raw passenger data.
Example input:
pclass     = 3
sex        = female
age        = 25
sibsp      = 0
parch      = 0
fare       = 15.0
embarked   = S
Reloaded pipeline output:
Prediction          = Survived
Survival probability = 0.5739
Because the preprocessing and estimator are stored together in the pipeline,
raw input can be passed directly to the reloaded model without manually
repeating the preprocessing steps.
22. Generated Files
Python Scripts
analytics/01_eda.py
analytics/02_modeling.py
Dataset
analytics/titanic.csv
Charts
analytics/charts/age_histogram.png
analytics/charts/age_boxplot.png
analytics/charts/fare_histogram.png
analytics/charts/fare_boxplot.png
analytics/charts/correlation_heatmap.png
analytics/charts/survival_by_sex_class.png
analytics/charts/age_fare_survival.png
analytics/charts/family_size_class_survival.png
analytics/charts/survival_by_embarked_sex.png
analytics/charts/decision_tree.png
analytics/charts/roc_curves.png
analytics/charts/fare_regression_residuals.png
Model Results
analytics/model/classification_comparison.csv
analytics/model/class_imbalance_comparison.csv
analytics/model/final_classification_results.csv
analytics/model/regression_results.csv
analytics/model/titanic_survival_pipeline.joblib
23. How to Run
From the project root:
python analytics/01_eda.py
Then:
python analytics/02_modeling.py
The scripts generate the cleaned dataset, charts, evaluation tables,
regression results, and saved machine-learning pipeline.
24. Module Summary
This module demonstrates a complete analytics and machine-learning
workflow, beginning with dataset profiling and cleaning and continuing
through exploratory analysis, classification, imbalance handling,
hyperparameter tuning, regression, evaluation, and pipeline persistence.
The Titanic classification experiments achieved test accuracies between
0.7472 and 0.8146 across the evaluated models. The Tuned Random Forest
achieved an F1 score of 0.7442, while Logistic Regression achieved the
highest ROC-AUC of 0.8610.
The Fare regression model achieved an R² of 0.3468, with the residual plot
showing visual evidence of heteroscedasticity.
The final trained preprocessing and classification pipeline was successfully
saved, reloaded, and us