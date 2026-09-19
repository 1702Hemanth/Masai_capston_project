# Zepto Data & AI Capstone Project

A complete data engineering, analytics, machine learning, and GenAI
support-assistant project built as a three-module capstone.

---

## Project Overview

This project implements an end-to-end data and AI workflow for Zepto,
covering:

1. **Data Pipeline** — web scraping, cleaning, SQLite database creation,
   SQL analytics, and validation.
2. **Analytics & Machine Learning** — Titanic data profiling, EDA,
   visualization, classification, class-imbalance analysis, Random Forest
   tuning, regression, and model serialization.
3. **Support Assistant** — an offline RAG application using local
   embeddings, ChromaDB, LangGraph, Pydantic, and FastAPI.

The project is organized as a single repository with separate directories
for each module.

---

# Repository Structure

text
Masai_capston_project/
│
├── data_pipeline/
│   ├── scraper.py
│   ├── database.py
│   ├── queries.py
│   ├── run_pipeline.py
│   ├── cleaned_books.csv
│   └── README.md
│
├── analytics/
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── titanic.csv
│   ├── README.md
│   ├── charts/
│   │   ├── age_histogram.png
│   │   ├── age_boxplot.png
│   │   ├── fare_histogram.png
│   │   ├── fare_boxplot.png
│   │   ├── correlation_heatmap.png
│   │   ├── survival_by_sex_class.png
│   │   ├── age_fare_survival.png
│   │   ├── family_size_class_survival.png
│   │   ├── survival_by_embarked_sex.png
│   │   ├── decision_tree.png
│   │   ├── roc_curves.png
│   │   └── fare_regression_residuals.png
│   └── model/
│       ├── classification_comparison.csv
│       ├── class_imbalance_comparison.csv
│       ├── final_classification_results.csv
│       ├── regression_results.csv
│       └── titanic_survival_pipeline.joblib
│
├── support_assistant/
│   ├── docs/
│   │   ├── doc_01.txt
│   │   ├── doc_02.txt
│   │   ├── doc_03.txt
│   │   ├── doc_04.txt
│   │   ├── doc_05.txt
│   │   ├── doc_06.txt
│   │   ├── doc_07.txt
│   │   └── doc_08.txt
│   ├── chroma_db/
│   ├── ingest.py
│   ├── rag.py
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md

support_assistant/chroma_db/ is generated locally by ingest.py and is
ignored by Git.

## Module 1 — Data Pipeline

Objective
The Data Pipeline module demonstrates a complete data engineering workflow:
Website
   ↓
Web Scraping
   ↓
Data Cleaning
   ↓
CSV
   ↓
SQLite Database
   ↓
SQL Queries
   ↓
Validation
The source website used for the scraping exercise is:
https://books.toscrape.com
The pipeline collects book information and converts the scraped data into a
clean structured dataset.
Technologies
- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite
- SQL
Main Files
scraper.py
Responsible for:
- Downloading book pages
- Extracting book information
- Converting prices
- Cleaning scraped values
- Creating the cleaned CSV dataset
database.py
Responsible for:
- Creating the SQLite database
- Creating database tables
- Loading cleaned data
- Inserting books and categories
queries.py
Contains SQL queries used to analyze the database.
run_pipeline.py
Runs the complete data pipeline in sequence.
Running Module 1
From the project root:
python data_pipeline/run_pipeline.py
The database is generated locally by the pipeline.

## Module 2 — Analytics & Machine Learning

Objective
The Analytics module performs exploratory data analysis and machine learning
using the classic Titanic dataset.
The workflow includes:
Titanic Dataset
      ↓
Profiling
      ↓
Cleaning
      ↓
EDA
      ↓
Visualization
      ↓
Feature Preparation
      ↓
Classification
      ↓
Class Imbalance Analysis
      ↓
Hyperparameter Tuning
      ↓
Regression
      ↓
Pipeline Serialization
Technologies
- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Joblib
Dataset
The classic Titanic dataset is initially loaded using:
sns.load_dataset("titanic")

The cleaned dataset is stored as:
analytics/titanic.csv
The saved CSV acts as the reproducible dataset used by the modeling stage.
EDA
The analysis includes:
- Dataset shape
- Dataset information
- Descriptive statistics
- Missing-value analysis
- Missing-value treatment
- Age distribution
- Fare distribution
- IQR outlier analysis
- Fare mean, median, and mode
- Fare skewness
- Survival rate by sex
- Survival rate by passenger class
- Survival rate by sex and passenger class
- Correlation analysis
- Correlation heatmap
- Multivariate visualizations
- Standardization sanity check
Machine Learning Models
Three classification models were evaluated:
1. Logistic Regression
2. Decision Tree
3. Random Forest
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
Class Imbalance
The training data was analyzed for class imbalance.
A majority-class baseline and balanced Logistic Regression were compared
using precision, recall, and F1 score.
Random Forest Tuning
GridSearchCV was used to tune:
- n_estimators
- max_depth
- max_features
The tuned Random Forest used OOB evaluation.
Regression
A multivariate Linear Regression model was developed to predict fare.
The evaluation metrics include:
- MAE
- RMSE
- R²
- Adjusted R²
Residual analysis was also performed to examine heteroscedasticity.
Final Classification Results
The tested classifiers produced the following test-set results:
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	0.8090	0.7833	0.6912	0.7344	0.8610
Decision Tree	0.7472	0.7347	0.5294	0.6154	0.8259
Random Forest	0.7978	0.7581	0.6912	0.7231	0.8212
Tuned Random Forest	0.8146	0.7869	0.7059	0.7442	0.8283


The detailed analysis and interpretations are documented in:
analytics/README.md
Saved Model
The complete fitted preprocessing and classification pipeline is saved as:
analytics/model/titanic_survival_pipeline.joblib
The pipeline was reloaded using Joblib and tested with raw passenger input.

## Module 3 — Support Assistant

Objective
The Support Assistant is an offline Retrieval-Augmented Generation system for
Zepto customer-support policy questions.
The pipeline is:
Policy Documents
      ↓
Document Loading
      ↓
Chunking
      ↓
Local Embeddings
      ↓
ChromaDB
      ↓
Intent Classification
      ↓
Conditional LangGraph Routing
      ↓
Retrieval / Direct Answer
      ↓
Pydantic Validation
      ↓
FastAPI JSON Response
Technologies
- Python
- Sentence Transformers
- all-MiniLM-L6-v2
- ChromaDB
- LangGraph
- Pydantic
- FastAPI
- Uvicorn
- Docker
Policy Corpus
Eight policy documents are included:
Document	Topic
doc_01.txt	Delivery Policy
doc_02.txt	Returns & Refunds
doc_03.txt	Membership Tiers
doc_04.txt	Order Tracking
doc_05.txt	Order Cancellation
doc_06.txt	Damaged or Missing Items
doc_07.txt	Gift Cards
doc_08.txt	Customer Support Hours


Embeddings
The application uses the local:
all-MiniLM-L6-v2
model to generate document and query embeddings.
No LLM API key is required for the graded baseline.
ChromaDB
Embeddings are stored in:
zepto_policies
ChromaDB collection.
The database is generated locally using:
python support_assistant/ingest.py
The generated chroma_db/ directory is excluded from Git.
LangGraph
The graph contains three required nodes:
classify_intent
retrieve_and_answer
direct_answer
The routing is:
                  classify_intent
                  /              \
                 /                \
       policy_question       general_question
              |                    |
              v                    v
   retrieve_and_answer       direct_answer
Mock LLM Mode
The graded baseline is completely offline.
By default:
MOCK_LLM=1
or when the variable is unset.
In this mode:
- Intent classification uses a deterministic keyword heuristic.
- Policy questions use real ChromaDB retrieval.
- The final answer is generated deterministically.
- General questions receive a fixed response.
- No LLM API is contacted.
The optional real-LLM path can be activated with:
MOCK_LLM=0
but the project does not depend on it.
Structured Response
The final response follows:
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 1.0
}
The response is validated using Pydantic.
FastAPI
Start the application:
uvicorn support_assistant.main:app --host 127.0.0.1 --port 7860
The API is available at:
http://127.0.0.1:7860
Endpoint
POST /ask
Request
{
  "query": "What is the delivery fee for orders below INR 149?"
}
Example Policy Response
{
  "answer": "Based on the retrieved context: Delivery Policy: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order vol",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_07"
  ],
  "confidence": 1.0
}
Example General Question
Request:
{
  "query": "What is the capital of India?"
}
Response:
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
Docker
A Dockerfile is provided at:
support_assistant/Dockerfile
Build:
docker build -t zepto-support-assistant ./support_assistant
Run:
docker run --rm -p 7860:7860 zepto-support-assistant
The container exposes port:
7860
Docker was not available in the development environment, so the Docker
build could not be executed during development.

## Design Decisions

 # 1. SQLite for Module 1

SQLite was selected because it is lightweight, local, and sufficient for the
structured dataset used in the data-pipeline module.

# 2. Pandas for Data Processing

Pandas provides convenient tools for:
- Data cleaning
- Transformation
- CSV processing
- Statistical analysis
- Data validation

# 3. Scikit-learn Pipelines for Module 2

Preprocessing and estimators are combined into pipelines so that training
transformations are applied consistently during prediction.

# 4. Joblib for Model Persistence

Joblib is used to save and reload the complete fitted classification
pipeline.

# 5. ChromaDB for Module 3

ChromaDB provides local vector storage and similarity retrieval without
requiring a hosted vector database.

# 6. Sentence Transformers

all-MiniLM-L6-v2 provides local embeddings without requiring an external
embedding API.

# 7. LangGraph

LangGraph provides explicit state-based orchestration and conditional
routing between policy retrieval and direct-answer paths.

# 8. Deterministic Mock Mode

The MOCK_LLM baseline makes the Support Assistant reproducible and
completely offline.
This ensures the required functionality does not depend on:
- API keys
- Paid services
- External LLM providers