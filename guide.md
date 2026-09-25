# ECOM6004 Assessment 1 - Complete Setup and Usage Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Initial Setup](#initial-setup)
4. [Project Structure](#project-structure)
5. [File and Function Explanations](#file-and-function-explanations)
6. [Running the Analysis](#running-the-analysis)
7. [Understanding the Output](#understanding-the-output)
8. [Customization](#customization)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance](#maintenance)

---

## Project Overview

This codebase implements the analysis for **ECOM6004 Assessment 1 (Semester 2, 2026)** at Curtin University. It applies binary and ordered response models to the Bank Marketing dataset to answer the research question: **"Which clients should be contacted?"**

### Key Features
- **Binary Models**: Linear Probability Model (LPM), Logit, and Probit
- **Ordered Models**: Ordered Logit and Ordered Probit
- **Data Auditing**: Quality checks before analysis
- **Stratified Split**: 80/20 train/test split with reproducible seeding
- **Decision Log**: Translates test evidence into cautious client decisions
- **Reproducibility**: All results reproducible using student ID as random seed

---

## Prerequisites

### Required Software
- **Python**: Version 3.8 or higher
- **Operating System**: macOS, Linux, or Windows

### Required Python Packages
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
statsmodels>=0.14.0
scipy>=1.10.0
patsy>=0.5.0
```

---

## Initial Setup

### Step 1: Clone or Download the Codebase

If you have access to the repository:
```bash
git clone <repository-url>
cd ECOM6004_Assessment1
```

If you received a zip file:
1. Extract the zip file
2. Navigate to the extracted directory: `ECOM6004_Assessment1`

### Step 2: Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal prompt, indicating the virtual environment is active.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required Python packages.

### Step 4: Verify Installation

```bash
python -c "import pandas, statsmodels, sklearn; print('All packages installed successfully')"
```

If no errors appear, the installation is successful.

### Step 5: Update Your Student ID

Open `main.py` in a text editor and find line 8:
```python
STUDENT_ID = 12345678
```

Replace `12345678` with your actual numeric student ID.

**Important**: The student ID is used as the random seed for the train/test split, ensuring reproducibility.

---

## Project Structure

```
ECOM6004_Assessment1/
├── .windsurfrules                  # AI agent constraints (read-only)
├── requirements.txt                 # Python package dependencies
├── guide.md                        # This comprehensive guide
├── README.md                        # Quick reference guide
├── data/                            # Data directory
│   └── bank_assessment_sem2_2026.csv    # Bank Marketing dataset
├── src/                            # Source code modules
│   ├── __init__.py                 # Package initialization
│   ├── data_loader.py              # Data loading and preparation
│   ├── binary_models.py            # Binary model fitting and evaluation
│   ├── ordered_models.py           # Ordered model fitting and diagnostics
│   └── utils.py                    # Table formatting utilities
├── main.py                         # Main execution script
└── StudentID_A1_GenAI_Disclosure.docx  # Required GenAI disclosure form
```

---

## File and Function Explanations

### 1. `main.py` - Main Execution Script

**Purpose**: Orchestrates the entire analysis pipeline from data loading to reproducibility recording.

**Key Functions**:
- `main()`: Main execution function that runs all analysis steps

**Execution Flow**:
1. Loads and prepares data with audit
2. Fits binary models (LPM, Logit, Probit) on training data
3. Evaluates selected model on test data
4. Creates decision log for client recommendation
5. Fits ordered models on full dataset
6. Diagnoses ordered models
7. Outputs reproducibility record

**How to Run**:
```bash
python main.py
```

---

### 2. `src/data_loader.py` - Data Loading and Preparation

**Purpose**: Loads the Bank Marketing dataset, performs data auditing, applies required transformations, and creates stratified train/test split.

#### Functions:

**`audit_data(df: pd.DataFrame)`**
- **Purpose**: Performs data quality checks
- **Returns**: DataFrame with audit summary including:
  - Total observations
  - Missing values count
  - Duplicate rows count
  - Target variable distribution
  - Response level distribution
  - Age and balance ranges

**`load_and_prepare_bank_data(filepath: str, student_id: int)`**
- **Purpose**: Loads, prepares, and splits the dataset
- **Parameters**:
  - `filepath`: Path to the CSV file
  - `student_id`: Your student ID (used as random seed)
- **Returns**: 
  - `df`: Full prepared dataset
  - `train`: Training set (80%)
  - `test`: Test set (20%)
  - `audit_summary`: Data audit results

**Transformations Applied**:
1. Creates `target`: Binary variable (1 if y='yes', 0 otherwise)
2. Creates `age10`: (age - 40) / 10 (centered and scaled)
3. Creates `balance1000`: balance / 1000 (scaled to thousands)
4. Creates `prior_status`: Categorical variable from `pdays` and `poutcome`
   - 'not previously contacted' (pdays = -1)
   - 'previous success' (poutcome = 'success')
   - 'previous failure' (poutcome = 'failure')
   - 'previous other/unknown' (everything else)
5. Sets categorical reference categories:
   - housing: 'no' as reference
   - loan: 'no' as reference
   - contact: 'telephone' as reference
   - prior_status: 'not previously contacted' as reference

**Train/Test Split**:
- 80% training, 20% test
- Stratified by target variable
- Uses student ID as random seed for reproducibility

---

### 3. `src/binary_models.py` - Binary Response Models

**Purpose**: Fits and evaluates binary response models (LPM, Logit, Probit) for predicting term deposit subscription.

#### Functions:

**`fit_binary_models(train_df: pd.DataFrame)`**
- **Purpose**: Fits LPM, Logit, and Probit models on training data
- **Parameters**:
  - `train_df`: Training dataset
- **Returns**: Dictionary with fitted models:
  - `'lpm'`: Linear Probability Model
  - `'logit'`: Logit model
  - `'probit'`: Probit model

**Model Formula**:
```
target ~ age10 + balance1000 + C(housing) + C(loan) + C(contact) + previous + C(prior_status)
```

**`evaluate_test_set(model, test_df: pd.DataFrame, threshold: float = 0.50)`**
- **Purpose**: Evaluates a fitted model on held-out test set
- **Parameters**:
  - `model`: Fitted model to evaluate
  - `test_df`: Test dataset
  - `threshold`: Classification threshold (default 0.50)
- **Returns**: Dictionary with metrics:
  - `obs_event_rate`: Observed proportion of positive outcomes
  - `mean_pred_prob`: Mean predicted probability
  - `brier_score`: Brier score (lower is better)
  - `sensitivity`: True positive rate
  - `specificity`: True negative rate
  - `balanced_accuracy`: Average of sensitivity and specificity
  - `confusion_matrix`: 2x2 confusion matrix

**Important**: Test evaluation happens **once** with no refitting or threshold tuning, as per assessment requirements.

**`create_decision_log(test_metrics: dict, model_name: str = 'logit')`**
- **Purpose**: Translates test evidence into a cautious client decision
- **Parameters**:
  - `test_metrics`: Dictionary from `evaluate_test_set()`
  - `model_name`: Name of the model used
- **Returns**: DataFrame with decision log including:
  - Model used
  - Test metrics (Brier score, balanced accuracy, sensitivity, specificity)
  - Decision: Whether to proceed with contact strategy
  - Rationale: Explanation of the decision

**Decision Logic**:
- If balanced accuracy > 60%: Proceed with targeted contact strategy
- Otherwise: Caution - use with additional business judgment

---

### 4. `src/ordered_models.py` - Ordered Response Models

**Purpose**: Fits and diagnoses ordered response models (Ordered Logit, Ordered Probit) for predicting response efficiency.

#### Functions:

**`fit_ordered_models(df: pd.DataFrame)`**
- **Purpose**: Fits Ordered Logit and Ordered Probit models on full dataset
- **Parameters**:
  - `df`: Full dataset (not split)
- **Returns**: Dictionary with fitted models:
  - `'ologit'`: Ordered Logit model
  - `'oprobit'`: Ordered Probit model

**Model Formula**:
```
response_level ~ age10 + balance1000 + C(housing) + C(loan) + C(contact) + previous + C(prior_status)
```

**Response Levels**:
- `0`: No subscription
- `1`: Subscription after at least two current-campaign contacts
- `2`: Subscription on the first contact

**Interpretation**: Higher values indicate more efficient conversion (fewer contacts needed).

**Technical Note**: Uses `patsy.dmatrix()` to create design matrix without intercept, respecting categorical reference categories set in data preparation.

**`diagnose_ordered_models(ordered_results: dict)`**
- **Purpose**: Diagnoses ordered model fit and convergence
- **Parameters**:
  - `ordered_results`: Dictionary from `fit_ordered_models()`
- **Returns**: DataFrame with diagnostics:
  - AIC (Akaike Information Criterion)
  - BIC (Bayesian Information Criterion)
  - Log-likelihood
  - Number of observations
  - Convergence status

---

### 5. `src/utils.py` - Utility Functions

**Purpose**: Provides table formatting functions for clean output.

#### Functions:

**`format_summary_table(model_results: dict, model_names: list = None)`**
- **Purpose**: Creates coefficient comparison table across multiple models
- **Parameters**:
  - `model_results`: Dictionary of fitted models
  - `model_names`: List of model names to include
- **Returns**: DataFrame with:
  - Parameter names
  - Coefficients for each model
  - P-values for each model

**`format_metrics_table(metrics: dict)`**
- **Purpose**: Formats evaluation metrics into clean table
- **Parameters**:
  - `metrics`: Dictionary of evaluation metrics
- **Returns**: DataFrame with metrics as rows and values as columns

---

### 6. `requirements.txt` - Dependencies

**Purpose**: Lists all Python packages required to run the analysis.

**Contents**:
```
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
scikit-learn>=1.2.0    # Train/test split
statsmodels>=0.14.0    # Statistical models
scipy>=1.10.0          # Scientific computing
patsy>=0.5.0           # Formula interface for models
```

---

### 7. `.windsurfrules` - AI Agent Constraints

**Purpose**: Contains rules for AI agents assisting with this project.

**Key Rules**:
- Never mix training and test samples during model development
- Test set evaluated once with no refitting or threshold tuning
- Ordered models use full dataset
- Code must run top-to-bottom reproducibly using student ID as seed
- Use `statsmodels.formula.api` for binary models
- Use `statsmodels.miscmodels.ordinal_model.OrderedModel` for ordered models
- Return tabular outputs as pandas DataFrames

**Note**: This file is for AI agent reference and should not be modified.

---

## Running the Analysis

### Complete Execution

After completing the initial setup:

```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Run the analysis
python main.py
```

### Expected Output

The script will print the following sections:

1. **Data Loading & Preparation**
   - Dataset sizes (full, train, test)
   - Data audit summary

2. **Fitting Binary Response Models**
   - AIC values for Logit and Probit
   - Coefficient comparison table (LPM, Logit, Probit)

3. **Out-of-Sample Test Evaluation**
   - Brier score
   - Balanced accuracy
   - Test set metrics table
   - Binary-stage decision log

4. **Ordered Response Models**
   - AIC values for Ordered Logit and Ordered Probit
   - Coefficient comparison table
   - Model diagnostics (AIC, BIC, convergence)

5. **Reproducibility Record**
   - Student ID
   - Random seed
   - Model specifications
   - Formulas used
   - Thresholds

---

## Understanding the Output

### Coefficient Tables

**Binary Models (LPM, Logit, Probit)**:
- Coefficients represent the effect of each predictor on the probability of subscription
- LPM coefficients: Direct change in probability
- Logit/Probit coefficients: Change in log-odds
- P-values indicate statistical significance (typically p < 0.05)

**Ordered Models (Ordered Logit, Ordered Probit)**:
- Coefficients represent the effect on the log-odds of being in a higher response category
- Positive coefficients increase likelihood of higher response levels (more efficient conversion)

### Evaluation Metrics

**Brier Score**: Measures calibration of predicted probabilities (lower is better, range 0-0.25 for binary)

**Balanced Accuracy**: Average of sensitivity and specificity (higher is better, range 0-1)

**Sensitivity**: True positive rate (ability to detect subscriptions)

**Specificity**: True negative rate (ability to detect non-subscriptions)

### Decision Log

The decision log provides:
- Model performance summary
- Recommendation on whether to proceed with contact strategy
- Rationale explaining the decision based on metrics

---

## Customization

### Changing the Model Formula

Edit the `FORMULA` constant in `src/binary_models.py`:
```python
FORMULA = "target ~ age10 + balance1000 + C(housing) + C(loan) + C(contact) + previous + C(prior_status)"
```

Edit the formula in `src/ordered_models.py`:
```python
formula = "age10 + balance1000 + C(housing) + C(loan) + C(contact) + previous + C(prior_status)"
```

### Changing the Test Threshold

Edit the threshold in `main.py` when calling `evaluate_test_set()`:
```python
test_metrics = evaluate_test_set(binary_results['logit'], test_df, threshold=0.50)
```

**Note**: Per assessment requirements, threshold tuning on test data is not allowed.

### Adding New Predictors

1. Add transformation in `src/data_loader.py` in the `load_and_prepare_bank_data()` function
2. Add categorical encoding if needed
3. Update formula in `src/binary_models.py` and `src/ordered_models.py`

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'pandas'`
- **Solution**: Ensure virtual environment is activated and dependencies are installed:
  ```bash
  source venv/bin/activate
  pip install -r requirements.txt
  ```

**Issue**: `FileNotFoundError: data/bank_assessment_sem2_2026.csv`
- **Solution**: Ensure the data file exists in the `data/` directory and you're running from the project root

**Issue**: Convergence warnings for ordered models
- **Solution**: This is normal for complex models. Check the convergence status in the diagnostics output. If models don't converge, consider simplifying the formula.

**Issue**: Different results on different runs
- **Solution**: Ensure `STUDENT_ID` is set correctly in `main.py`. The student ID is used as the random seed for reproducibility.

---

## Maintenance

### Updating Dependencies

To update packages to their latest compatible versions:
```bash
pip install --upgrade -r requirements.txt
```

### Adding New Dependencies

1. Install the package:
   ```bash
   pip install <package-name>
   ```
2. Add to `requirements.txt`:
   ```
   <package-name>=<version>
   ```

### Version Control

If using Git:
```bash
git add .
git commit -m "Description of changes"
git push
```

### Backup

Regularly backup:
- Your modified `main.py` with your student ID
- The completed GenAI disclosure form
- Any custom modifications you make

---

## Academic Integrity Notes

### Reproducibility

- All results are reproducible using the student ID as the random seed
- Keep your student ID consistent across runs
- Document any deviations from the standard analysis

### GenAI Disclosure

- Complete the `StudentID_A1_GenAI_Disclosure.docx` form
- Document any AI assistance used in the analysis
- Submit the disclosure form separately as required

### Code Submission

- Submit only `main.py` as your executable code
- Ensure all code runs top-to-bottom without errors
- Do not include raw code in your report (code stays in `src/` and `main.py`)

---

## Quick Reference

### Essential Commands

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run analysis
python main.py

# Deactivate virtual environment
deactivate
```

### Key Files to Modify

- `main.py`: Update your student ID (line 8)
- `src/data_loader.py`: Add data transformations
- `src/binary_models.py`: Modify binary model formula
- `src/ordered_models.py`: Modify ordered model formula

### Important Constants

- Student ID: Set in `main.py` line 8
- Train/test split: 80/20 (fixed)
- Test threshold: 0.50 (default)
- Random seed: Student ID (for reproducibility)

---

## Support

For issues related to:
- **Python installation**: Consult Python documentation
- **Package installation**: Check package documentation or use `pip show <package>`
- **Statistical concepts**: Refer to ECOM6004 course materials
- **Assessment requirements**: Review the assessment brief

---

**End of Guide**
