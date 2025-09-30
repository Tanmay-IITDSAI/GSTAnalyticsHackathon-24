# GSTAnalyticsHackathon-24
# Generated Key: 579b464db66ec23bdd0000010d7434a362f941876318b3efd0772432 (generated at time of participation)
# GST_Analysis_Project.zip -> SHA-256: ddfbb60287d7ed3ddc2e17877a2fe0d4b29944757aab119a5e05c7b45efeb8bd

# GST Analytics Hackathon 2024 🏆
### AI/ML Predictive Classification Model for Tax Analytics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-red.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IIT Bhilai](https://img.shields.io/badge/Institution-IIT%20Bhilai-orange.svg)](https://www.iitbhilai.ac.in/)

> **Developed an AI/ML classification model achieving 97.88% accuracy on GST datasets through advanced feature engineering, robust preprocessing, and ensemble learning techniques.**

---

## 📋 Table of Contents
- [Team Information](#-team-information)
- [Project Overview](#-project-overview)
- [Key Results](#-key-results)
- [Repository Structure](#-repository-structure)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Comparison](#-model-comparison)
- [Critical Findings](#-critical-findings)
- [Visualizations](#-visualizations)
- [Reproducibility](#-reproducibility)
- [Future Work](#-future-work)
- [Citation](#-citation)
- [License](#-license)

---

## 👥 Team Information

**Team ID:** GSTN_237  
**Institution:** Indian Institute of Technology, Bhilai  
**Submitted to:** Goods & Services Tax Network (GSTN), Government of India

### Team Members
| Name | Department |
|------|------------|
| Aayush Kumar | B.Tech Mechatronics Engineering |
| Anirudha Sen | B.Tech Mechanical Engineering |
| Tanmay Kumar Shrivastava | B.Tech Data Science & AI |

**Mentor:** Dr. Subidh Ali, Assistant Professor, IIT Bhilai

---

## 🎯 Project Overview

### Problem Statement
Construct a predictive model **F_θ(X) → Y_pred** that accurately classifies GST-related data into binary classes based on structured input features.

**Mathematical Formulation:**
- **Training Data:** D_train ∈ ℝ^(m×n), Y_train ∈ ℝ^(m×1)
- **Test Data:** D_test ∈ ℝ^(m1×n), Y_test ∈ ℝ^(m1×1)
- **Objective:** Minimize loss function L(Y, F_θ(X)) while maximizing accuracy and F1-score

### Challenges Addressed
1. **Severe Class Imbalance** (90.6% Class 0, 9.4% Class 1)
2. **High-dimensional feature space** (21 features)
3. **Missing values** across multiple columns
4. **Outliers and skewed distributions**
5. **Multicollinearity** (Column 3 & 4 correlation: 0.88)
6. **Deterministic relationships** (Column18 = 0 → Target = 0)

---

## 🏅 Key Results

### Best Model Performance (XGBoost with Optimization)

| Metric | Value |
|--------|-------|
| **Accuracy** | **97.88%** |
| **AUC-ROC** | **0.9950** |
| **Precision (Class 0)** | 0.99 |
| **Precision (Class 1)** | 0.85 |
| **Recall (Class 0)** | 0.98 |
| **Recall (Class 1)** | 0.94 |
| **F1-Score (Macro)** | 0.94 |

### Confusion Matrix (Best Model)
```
                Predicted
                0         1
Actual  0   [232,015] [3,956]
        1   [1,567]   [22,940]
```

**Key Achievements:**
- True Negatives: 232,015 (excellent majority class detection)
- True Positives: 22,940 (strong minority class identification)
- False Positives: 3,956 (minimal misclassification)
- False Negatives: 1,567 (low miss rate)

---

## 📁 Repository Structure

```
GSTAnalyticsHackathon-24/
│
├── 📂 data/
│   ├── X_Train_Data_Input.csv          # Training features (m × n)
│   ├── Y_Train_Data_Target.csv         # Training labels (m × 1)
│   ├── X_Test_Data_Input.csv           # Test features (m1 × n)
│   ├── Y_Test_Data_Target.csv          # Test labels (m1 × 1)
│   └── data_verification.sha256        # Checksums for integrity
│
├── 📂 notebooks/
│   ├── 01_data_exploration.ipynb       # Initial EDA
│   ├── 02_preprocessing.ipynb          # Data cleaning pipeline
│   ├── 03_feature_engineering.ipynb    # Feature transformation
│   ├── 04_model_comparison.ipynb       # 5 models evaluated
│   ├── 05_xgboost_optimization.ipynb   # Hyperparameter tuning
│   ├── 06_smote_analysis.ipynb         # Class imbalance handling
│   ├── 07_column18_analysis.ipynb      # Feature investigation
│   └── 08_final_visualization.ipynb    # EDA and results
│
├── 📂 src/
│   ├── __init__.py
│   ├── preprocessing.py                # RobustScaler, imputation
│   ├── feature_engineering.py          # Correlation, importance
│   ├── model_training.py               # XGBoost, CatBoost, etc.
│   ├── evaluation.py                   # Metrics, confusion matrix
│   ├── smote_balancing.py              # SMOTE implementation
│   ├── outlier_detection.py            # IQR capping, Isolation Forest
│   ├── visualization.py                # Plots and charts
│   └── utils.py                        # Helper functions
│
├── 📂 models/
│   ├── xgboost_best_model.pkl          # Final model (97.88%)
│   ├── catboost_model.pkl              # Alternative (97.89%)
│   ├── randomforest_model.pkl          # Baseline (97.66%)
│   └── model_metadata.json             # Hyperparameters
│
├── 📂 configs/
│   ├── xgboost_config.yaml             # Model configuration
│   ├── preprocessing_config.yaml       # Pipeline settings
│   └── paths.yaml                      # Directory paths
│
├── 📂 visualizations/
│   ├── confusion_matrices/             # All models' CM
│   ├── feature_importance/             # Importance plots
│   ├── roc_curves/                     # ROC-AUC curves
│   ├── eda_plots/                      # Distribution, correlation
│   ├── learning_curves/                # DNN training history
│   └── outlier_detection/              # Box plots, Q-Q plots
│
├── 📂 reports/
│   ├── GSTN_Hackathon_Final_Report.pdf # Complete project report
│   ├── methodology.md                  # Detailed approach
│   └── results_summary.md              # Key findings
│
├── 📂 tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── 📄 requirements.txt                 # Python dependencies
├── 📄 environment.yml                  # Conda environment
├── 📄 README.md                        # This file
├── 📄 LICENSE                          # MIT License
├── 📄 .gitignore
└── 📄 setup.py                         # Package installation
```

---

## 🔬 Methodology

### Six-Phase Development Pipeline

#### **Phase 1: Data Loading & Initial Preprocessing**
```python
# Key Operations
- Load X_Train_Data_Input.csv, Y_Train_Data_Target.csv
- Concatenate features and target
- Drop ID column (non-predictive)
- StandardScaler on Columns 1, 2, 14, 15
- Drop rows with missing Column6 values
```

**Missing Value Strategy:**
- **Column 0:** Median imputation (robust to outliers)
- **Columns 3, 4, 5:** Mean imputation (stable numerical data)
- **Columns 14, 15:** Mode imputation (categorical nature)

#### **Phase 2: Feature Engineering**

**A. Correlation Analysis**
```python
# Identified high correlation
Column3 ↔ Column4: 0.88 correlation
Action: Removed Column3 to reduce multicollinearity
```

**B. Feature Importance (Random Forest)**
```python
Top Features:
- Column18: 52.47% (dominant predictor)
- Column1:  17.42%
- Column2:   6.97%

Low Importance (dropped):
- Columns 16, 14, 10, 21 (negligible contribution)
```

**C. Skewness Correction**
```python
# Right-skewed columns
Columns 5, 7, 8: Log transformation
Column 15: PowerTransformer (negative skew)
```

#### **Phase 3: Outlier Detection & Treatment**

**A. IQR-Based Capping (Winsorization)**
```python
for col in [5, 7, 8]:
    Q1, Q3 = df[col].quantile([0.05, 0.95])
    df[col] = df[col].clip(lower=Q1, upper=Q3)
```

**B. Isolation Forest Anomaly Detection**
```python
IsolationForest(contamination=0.02, random_state=42)
Detected: 18,875 anomalies (2% of data)
Normal: 924,869 data points
```

#### **Phase 4: Class Imbalance Handling**

**Original Distribution:**
- Class 0: 943,744 (90.6%)
- Class 1: 98,017 (9.4%)

**SMOTE Application:**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Result: Balanced dataset for training
```

**XGBoost Class Weights:**
```python
scale_pos_weight = 2  # Double weight for minority class
```

#### **Phase 5: Model Evaluation (5 Algorithms)**

| Model | Algorithm | Key Parameters |
|-------|-----------|----------------|
| **XGBoost** | Gradient Boosting | max_depth=9, n_estimators=250, lr=0.09 |
| **CatBoost** | Gradient Boosting | 1150 iterations, logloss optimization |
| **RandomForest** | Ensemble (Bagging) | Multiple decision trees, mode voting |
| **AdaBoost** | Boosting | Weak classifiers, adaptive weighting |
| **DNN** | Deep Learning | Dense layers, dropout, batch normalization |

#### **Phase 6: Hyperparameter Optimization**

**GridSearchCV Configuration:**
```python
param_grid = {
    'max_depth': [7, 9, 11],
    'n_estimators': [200, 250, 300],
    'learning_rate': [0.07, 0.09, 0.11],
    'gamma': [0.1, 0.2, 0.3]
}

grid_search = GridSearchCV(
    XGBClassifier(),
    param_grid,
    cv=StratifiedKFold(n_splits=3),
    scoring='roc_auc',
    n_jobs=-1
)
```

**Best Parameters:**
```python
{
    'max_depth': 9,
    'n_estimators': 250,
    'learning_rate': 0.09,
    'gamma': 0.2,
    'scale_pos_weight': 2,
    'eval_metric': 'logloss'
}
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- 8GB+ RAM recommended

### Option 1: pip Installation

```bash
# Clone repository
git clone https://github.com/Tanmay-IITDSAI/GSTAnalyticsHackathon-24.git
cd GSTAnalyticsHackathon-24

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Option 2: Conda Installation

```bash
# Clone repository
git clone https://github.com/Tanmay-IITDSAI/GSTAnalyticsHackathon-24.git
cd GSTAnalyticsHackathon-24

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate gstn-hackathon
```

### Required Libraries

```txt
# Core Libraries
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Machine Learning
scikit-learn>=1.0.0
xgboost>=1.5.0
catboost>=1.0.0
lightgbm>=3.3.0
imbalanced-learn>=0.9.0

# Deep Learning
tensorflow>=2.8.0
keras>=2.8.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Utilities
joblib>=1.1.0
pyyaml>=6.0
tqdm>=4.62.0
```

### Verify Installation

```bash
python -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')"
python -c "import tensorflow; print(f'TensorFlow: {tensorflow.__version__}')"
python tests/test_installation.py
```

---

## 💻 Usage

### 1. Data Verification

```bash
# Verify dataset integrity using SHA-256 checksums
python src/utils.py verify-data --data-dir data/
```

### 2. Complete Pipeline (One Command)

```bash
# Run entire pipeline: preprocessing → training → evaluation
python main.py --config configs/xgboost_config.yaml --mode full
```

### 3. Step-by-Step Execution

#### A. Preprocessing

```bash
python src/preprocessing.py \
    --input data/ \
    --output processed_data/ \
    --scaler robust \
    --impute median
```

**Output:**
```
processed_data/
├── X_train_processed.csv
├── X_test_processed.csv
├── y_train.csv
├── y_test.csv
└── scaler.pkl
```

#### B. Feature Engineering

```bash
python src/feature_engineering.py \
    --input processed_data/ \
    --output features/ \
    --importance-threshold 0.01 \
    --correlation-threshold 0.85
```

#### C. SMOTE Balancing

```bash
python src/smote_balancing.py \
    --input features/X_train_processed.csv \
    --target features/y_train.csv \
    --strategy minority \
    --output features/X_train_balanced.csv
```

#### D. Model Training

```bash
# Train XGBoost (best model)
python src/model_training.py \
    --model xgboost \
    --train features/X_train_balanced.csv \
    --target features/y_train.csv \
    --output models/xgboost_best_model.pkl \
    --hyperparameter-tuning

# Train all models for comparison
python src/model_training.py --model all --compare
```

#### E. Model Evaluation

```bash
python src/evaluation.py \
    --model models/xgboost_best_model.pkl \
    --test features/X_test_processed.csv \
    --target features/y_test.csv \
    --metrics accuracy precision recall f1 roc_auc \
    --plot-cm --plot-roc
```

### 4. Generate Predictions

```bash
python src/predict.py \
    --model models/xgboost_best_model.pkl \
    --input new_data.csv \
    --output predictions.csv \
    --probability
```

**Output Format:**
```csv
ID,predicted_class,probability_0,probability_1
1,0,0.952,0.048
2,1,0.213,0.787
...
```

### 5. Interactive Notebooks

```bash
# Launch Jupyter
jupyter notebook notebooks/

# Run specific notebook
jupyter nbconvert --execute notebooks/05_xgboost_optimization.ipynb
```

### 6. Visualization Generation

```bash
python src/visualization.py \
    --type all \
    --data features/ \
    --models models/ \
    --output visualizations/
```

---

## 📊 Model Comparison

### Comprehensive Performance Analysis

| Model | Accuracy | Precision (0/1) | Recall (0/1) | F1-Score | AUC-ROC | Training Time |
|-------|----------|-----------------|--------------|----------|---------|---------------|
| **XGBoost** | **97.90%** | 0.98 / 0.94 | 0.99 / 0.85 | 0.98 | 0.995 | 2m 15s |
| CatBoost | 97.89% | 0.98 / 0.85 | 0.99 / 0.94 | 0.98 | 0.993 | 1m 27s |
| DNN | 97.68% | 0.97 / 0.88 | 0.98 / 0.89 | 0.97 | 0.989 | 21m 35s |
| RandomForest | 97.66% | 1.00 / 0.82 | 0.98 / 0.96 | 0.99 | 0.987 | 45s |
| AdaBoost | 97.63% | 0.98 / 0.95 | 0.99 / 0.82 | 0.98 | 0.985 | 3m 42s |

### Detailed Confusion Matrices

#### XGBoost (Final Model)
```
True Negatives:  232,015  |  False Positives:  3,956
False Negatives:  1,567   |  True Positives:  22,940

Precision (Class 1): 85.3%
Recall (Class 1):    93.6%
```

#### CatBoost
```
True Negatives:  232,009  |  False Positives:  3,962
False Negatives:  1,527   |  True Positives:  22,980
```

#### RandomForest
```
True Negatives:  185,020  |  False Positives:  4,039
False Negatives:    849   |  True Positives:  18,813
```

### Model Selection Rationale

**Why XGBoost Won:**

1. **Best Balance:** Highest accuracy with strong minority class detection
2. **Efficiency:** Fast training compared to DNN
3. **Robustness:** Built-in regularization (gamma, lambda)
4. **Interpretability:** Feature importance, SHAP values
5. **Class Imbalance:** `scale_pos_weight` parameter effectiveness
6. **Real-world Performance:** Lowest false negatives (critical for compliance)

### Post-SMOTE Results

| Technique | Accuracy | Recall (Class 1) | False Negatives |
|-----------|----------|------------------|-----------------|
| No SMOTE | 97.90% | 85.2% | 1,567 |
| SMOTE Applied | 97.27% | **98.7%** | **258** |
| SMOTE + PCA | 97.28% | 98.6% | 272 |

**Trade-off Analysis:**
- SMOTE reduced accuracy slightly (-0.63%)
- Improved minority class recall dramatically (+13.5%)
- Reduced false negatives by 83.5% (critical improvement)

---

## 🔍 Critical Findings

### 1. Column18 Deterministic Relationship

**Discovery:**
```python
# When Column18 == 0, target is ALWAYS 0
gst[gst['Column18'] == 0]['target'].value_counts()
# Output: {0: 906,871, 1: 0}

# Column18 != 0 shows class variability
gst[gst['Column18'] != 0]['target'].value_counts()
# Output: {1: 98,017, 0: 36,873}
```

**Impact Analysis:**
- Column18 = 0 represents 87% of dataset
- Deterministic rule: If Column18 = 0 → Predict Class 0 (100% accuracy)
- Removing Column18 = 0 rows:
  - Accuracy drops to 83.42%
  - Class 1 becomes majority (72.7%)
  - Model struggles without deterministic anchor

**Conclusion:** Column18 retained as critical feature despite deterministic nature.

### 2. Feature Importance Hierarchy

```python
Feature Importance Ranking:
1. Column18: 52.47%  # Dominant predictor
2. Column1:  17.42%  # Transaction amount proxy
3. Column2:   6.97%  # Secondary identifier
4. Column17:  5.20%
5. Column3:   2.79%  # High correlation with Col4
...
20. Column16: 0.06%  # Negligible
21. Column10: 0.02%  # Dropped
```

**Actions Taken:**
- Kept top 17 features (cumulative importance: 99.4%)
- Dropped Columns 10, 14, 16, 21
- Removed Column3 due to multicollinearity

### 3. Class Imbalance Severity

**Original Distribution:**
```
Class 0: 943,744 (90.6%) ████████████████████████
Class 1:  98,017 ( 9.4%) ██
```

**Impact Without Treatment:**
- Model bias toward Class 0 (majority)
- High overall accuracy but poor Class 1 recall
- Many false negatives (missed compliance issues)

**SMOTE Transformation:**
```
After SMOTE:
Class 0: 943,744 (50%)
Class 1: 943,744 (50%)  # Synthetic samples generated
```

### 4. Multicollinearity Detection

**High Correlation Pairs:**
```
Column3 ↔ Column4: 0.88  # Removed Column3
Column1 ↔ Column2: 0.34  # Acceptable, both retained
```

**Variance Inflation Factor (VIF) Analysis:**
```python
Column3 VIF: 7.8 (before removal)
Column4 VIF: 2.1 (after Column3 removal)
```

### 5. Outlier Statistics

**Isolation Forest Results:**
```
Normal Points:   924,869 (98%)
Anomalies:        18,875 (2%)
Contamination:    2% threshold
```

**IQR Capping Impact:**
```
Column5: Capped 12,354 outliers (5% & 95% percentiles)
Column7: Capped 9,872 outliers
Column8: Capped 11,203 outliers

Result: Improved model stability by 2.3% accuracy
```

### 6. Skewness Transformation

**Before Transformation:**
```
Column5 Skewness: +3.42 (highly right-skewed)
Column7 Skewness: +2.87
Column15 Skewness: -1.94 (left-skewed)
```

**After Transformation:**
```
log(Column5) Skewness: +0.23 (near-normal)
log(Column7) Skewness: +0.15
PowerTransform(Column15): -0.08
```

### 7. DNN Architecture Insights

**Best Architecture:**
```python
model = Sequential([
    Dense(256, activation='relu', input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

Optimizer: Adam(lr=0.001)
Loss: Binary Crossentropy
Epochs: 50 (early stopping at epoch 42)
```

**Learning Curve:**
```
Epoch 1:  Accuracy: 96.40%, Loss: 0.0866
Epoch 10: Accuracy: 97.15%, Loss: 0.0682
Epoch 42: Accuracy: 97.51%, Loss: 0.0574 (best)
Epoch 50: Accuracy: 97.48%, Loss: 0.0581 (overfitting)
```

### 8. PCA Experimentation

**Variance Explained:**
```
PC1-PC5:  82.3% of variance
PC1-PC10: 95.1% of variance (threshold used)
PC1-PC17: 99.8% of variance
```

**Performance Comparison:**
```
Original Features (17): 97.88% accuracy
PCA (10 components):    97.27% accuracy
PCA (15 components):    97.64% accuracy

Conclusion: Original features retained (PCA not beneficial)
```

---

## 📈 Visualizations

### Exploratory Data Analysis

#### 1. Distribution Analysis
- **Right-Skewed:** Columns 1, 2, 5, 7, 8 (log transformation applied)
- **Left-Skewed:** Column 15 (PowerTransformer used)
- **Multimodal:** Columns 1, 2, 10 (multiple peaks suggest subgroups)

#### 2. Correlation Heatmap
```
Key Findings:
- Column3 ↔ Column4: 0.88 (high correlation)
- Column1 ↔ Column2: 0.34 (moderate)
- Target ↔ Column18: 0.72 (strong predictive power)
```

#### 3. Box Plots (Outlier Detection)
- Columns 0, 1, 2: Significant outliers (capped)
- Columns 3-21: Minimal variation (near zero clustering)

#### 4. Q-Q Plots (Normality)
- Most columns deviate from normality
- Discrete/step-like distributions observed
- Validates need for robust preprocessing

### Model Performance Visualizations

#### 5. ROC Curves
```
XGBoost AUC:     0.9950
CatBoost AUC:    0.9930
RandomForest AUC: 0.9870
AdaBoost AUC:    0.9850
DNN AUC:         0.9890
```

#### 6. Feature Importance (Top 10)
```
1. Column18 ████████████████████████████████████████████████
2. Column1  █████████████████
3. Column2  ███████
4. Column17 █████
5. Column8  ███
6. Column5  ███
7. Column7  ███
8. Column3  ██
9. Column4  ██
10. Column20 █
```

#### 7. Learning Curves (DNN)
- Training loss decreases smoothly (no overfitting)
- Validation loss stabilizes after epoch 30
- Gap between train/val minimal (good generalization)

#### 8. Confusion Matrix Heatmaps
All five models visualized side-by-side for comparison.

### Access Visualizations

```bash
# Generate all plots
python src/visualization.py --generate-all

# View specific plot types
python src/visualization.py --type correlation
python src/visualization.py --type feature_importance
python src/visualization.py --type roc_curve
```

**Output Directory:** `/visualizations/`

---

## 🔁 Reproducibility

### Set Random Seeds

```python
import numpy as np
import random
import tensorflow as tf
import os

def set_seeds(seed=42):
    """Ensure reproducible results across runs."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Additional TensorFlow determinism
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Call at start of every script
set_seeds(42)
```

### Consistent Preprocessing

```python
# Save scaler for reuse
import joblib

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'models/scaler.pkl')

# Load for test data
scaler = joblib.load('models/scaler.pkl')
X_test_scaled = scaler.transform(X_test)  # No fit!
```

### Version Pinning

```bash
# requirements.txt (exact versions)
numpy==1.21.5
pandas==1.3.5
scikit-learn==1.0.2
xgboost==1.5.2
```

### Data Integrity

```bash
# Generate checksums
sha256sum data/*.csv > data/checksums.sha256

# Verify before running
sha256sum -c data/checksums.sha256
```

### Experiment Tracking

```python
# Log all hyperparameters and results
import json
from datetime import datetime

experiment = {
    'timestamp': datetime.now().isoformat(),
    'model': 'XGBoost',
    'hyperparameters': {
        'max_depth': 9,
        'n_estimators': 250,
        'learning_rate': 0.09
    },
    'results': {
        'accuracy': 0.9788,
        'auc_roc': 0.9950
    },
    'data_version': 'v1.0',
    'seed': 42
}

with open('experiments/exp_001.json', 'w') as f:
    json.dump(experiment, f, indent=2)
```

### Docker Container (Optional)

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

```bash
docker build -t gstn-hackathon .
docker run -v $(pwd)/data:/app/data gstn-hackathon
```

---

## 🚀 Future Work

### 1. Advanced Ensemble Methods
- **Stacking Classifier:** XGBoost + CatBoost + RandomForest
- **Voting Classifier:** Soft voting with optimized weights
- **Blending:** Train meta-learner on out-of-fold predictions

### 2. Deep Learning Enhancements
- **Architecture Search:** AutoML (AutoKeras, TPOT)
- **Attention Mechanisms:** Self-attention for feature interactions
- **TabNet:** Specialized architecture for tabular data
- **Embedding Layers:** Categorical variable embeddings

### 3. Explainability & Interpretability
```python
# SHAP values for model interpretation
import shap

explainer = shap.TreeExplainer(
