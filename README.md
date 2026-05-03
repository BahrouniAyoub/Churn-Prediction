# Customer Churn Prediction

A machine learning project for predicting customer churn in a telecommunications company using the Telco Customer Churn dataset. This project implements a custom **Gravitational Search Algorithm (GSA)** for feature selection and evaluates **15 classification models** to identify the best predictor.

## Project Overview

This notebook performs an end-to-end churn prediction pipeline:

1. **Data Preparation** - Loading, cleaning, and exploratory analysis
2. **Feature Engineering** - Encoding, scaling, and outlier detection
3. **Feature Selection** - Custom Gravitational Search Algorithm (GSA)
4. **Model Training** - 15 classifiers with 5-fold cross-validation
5. **Evaluation** - Comprehensive metrics comparison on the test set

## Dataset

- **Source:** Telco Customer Churn dataset
- **Size:** 7,043 customers, 21 columns
- **Target:** `Churn` (Yes/No) - whether a customer left the company
- **Features:** Demographics, account info, services, billing, and payment details

## Pipeline

### Phase 1: Data Preparation
- Load and inspect the dataset
- Exploratory Data Analysis with `ydata-profiling`
- Handle missing values in `TotalCharges` (fill with 0)
- Replace comma separators and convert to numeric
- Drop `customerID` column
- Duplicate detection (0 duplicates found)
- IQR outlier detection on numerical columns (0 outliers found)

### Phase 2: Encoding
- **Label Encoding:** Target variable `Churn` (No=0, Yes=1)
- **Binary Mapping:** `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`
- **One-Hot Encoding:** `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod`
- Result: 30 features after encoding

### Phase 3: Gravitational Search Algorithm (GSA) Feature Selection
- Custom `GSAFeatureSelector` class implementing binary GSA
- Fitness function minimizes error rate using Logistic Regression
- Parameters: 20 agents, 30 iterations, G0=100, alpha=10
- **Result:** 16 out of 30 features selected (Best accuracy: 0.8243)

**Selected Features:**
- `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `PaperlessBilling`
- `MonthlyCharges`, `TotalCharges`
- `MultipleLines_No phone service`, `InternetService_Fiber optic`
- `OnlineBackup_Yes`, `TechSupport_No internet service`, `TechSupport_Yes`
- `StreamingTV_No internet service`, `StreamingMovies_Yes`
- `Contract_One year`, `PaymentMethod_Credit card (automatic)`

### Phase 4: Train/Test Split & Scaling
- Stratified 80/20 split (random_state=42)
- MinMaxScaler on numerical columns (fit on train only to avoid data leakage)
- Final shapes: Train (5,634, 16), Test (1,409, 16)

### Phase 5: K-Fold Cross Validation (k=5)
All 15 models evaluated with StratifiedKFold on the training set:

| Model | CV Accuracy |
|---|---|
| **CatBoost Classifier** | **80.30%** |
| Logistic Regression | 80.25% |
| SVM Classifier Poly | 80.12% |
| AdaBoost Classifier | 79.73% |
| SVM Classifier Linear | 79.13% |
| Random Forest | 78.72% |
| Logistic Regression (AdaBoost) | 78.49% |
| Random Forest (AdaBoost) | 78.42% |
| XGBoost Classifier | 78.33% |
| KNN Classifier | 77.25% |
| AdaBoost (Extra Tree) | 76.87% |
| Extra Tree Classifier | 76.84% |
| Naive Bayes (Gaussian) | 73.91% |
| SVM (AdaBoost) | 73.46% |
| Decision Tree | 72.91% |

### Phase 6: Test Set Evaluation

| Model | Accuracy | Recall | Precision | F-Measure | AUC Score |
|---|---|---|---|---|---|
| **CatBoost Classifier** | **79.56%** | **51.07%** | **64.53%** | **57.01%** | **84.20%** |
| AdaBoost Classifier | 79.21% | 51.60% | 63.28% | 56.85% | 83.80% |
| Logistic Regression | 79.70% | 51.07% | 64.97% | 57.19% | 83.46% |
| SVM Classifier Linear | 78.99% | 53.74% | 62.04% | 57.59% | 82.04% |
| XGBoost Classifier | 77.57% | 48.40% | 59.54% | 53.39% | 81.11% |

## Best Model: CatBoost Classifier

| Metric | Value |
|---|---|
| Accuracy | 79.56% |
| Recall | 51.07% |
| Precision | 64.53% |
| F-Measure | 57.01% |
| **AUC Score** | **84.20%** |

CatBoost achieved the highest AUC score (84.20%), outperforming the reference paper's reported results for AdaBoost (81.71%, AUC=84%) and XGBoost (80.80%, AUC=84%).

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
catboost
ydata-profiling
matplotlib
seaborn
```

## Usage

1. Upload the `Customer Churn.csv` file to your Google Drive at `/content/drive/MyDrive/IA/`
2. Open the notebook in Google Colab
3. Run all cells sequentially

## Outputs

The notebook generates:
- EDA report (ydata-profiling)
- GSA convergence curve
- K-Fold CV accuracy comparison bar chart
- Test set metric bar charts (Accuracy, Recall, Precision, F-Measure)
- Confusion matrices for all 15 models
- ROC/AUC curves for all 15 models
- Final champion model summary
