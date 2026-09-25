# ECOM6004 Assessment 1 - Project Documentation

## Executive Summary

This project implements a comprehensive econometric analysis for ECOM6004 Assessment 1 (Semester 2, 2026) at Curtin University. The project applies binary and ordered response models to the Bank Marketing dataset to predict customer subscription to term deposits. The analysis follows strict academic standards for reproducibility, model evaluation, and reporting.

**Key Deliverables:**
- Binary response models (LPM, Logit, Probit) with out-of-sample evaluation
- Ordered response models (Ordered Logit, Ordered Probit) for multi-level outcomes
- Stratified train/test split ensuring reproducibility via student ID seeding
- Automated report generation using Quarto for HTML/PDF output

---

## 1. Project Context and Objectives

### 1.1 Academic Context
- **Course:** ECOM6004 - Econometrics
- **Institution:** Curtin University
- **Semester:** Semester 2, 2026
- **Assessment:** Assessment 1 - Binary and Ordered Response Models

### 1.2 Research Objectives
1. **Binary Classification:** Predict whether a customer subscribes to a term deposit (yes/no)
2. **Ordered Classification:** Analyze multi-level response patterns using ordered models
3. **Model Comparison:** Compare Linear Probability Model, Logit, and Probit specifications
4. **Out-of-Sample Evaluation:** Assess model performance on held-out test data

### 1.3 Dataset Overview
- **Source:** Bank Marketing Dataset (UCI Machine Learning Repository)
- **Sample Size:** 45,211 observations
- **Target Variable:** `y` - subscription to term deposit (binary: yes/no)
- **Ordered Variable:** `response_level` - multi-level response categories (0, 1, 2)
- **Key Predictors:** Age, balance, housing loan, personal loan, contact method, prior campaign history

---

## 2. Project Architecture

### 2.1 Directory Structure

```
ECOM6004_Assessment1/
├── .windsurfrules                  # AI agent constraints and coding standards
├── requirements.txt                 # Python package dependencies
├── README.md                        # Quick setup guide
├── project.md                       # This detailed project documentation
├── data/                            # Data storage
│   ├── bank_assessment_sem2_2026.csv    # Primary dataset
│   └── telco_assessment_sem2_2026.csv   # Secondary dataset (unused)
├── src/                             # Source code modules
│   ├── __init__.py                  # Package initialization
│   ├── data_loader.py               # Data loading and preprocessing
│   ├── binary_models.py             # Binary response model estimation
│   ├── ordered_models.py            # Ordered response model estimation
│   └── utils.py                     # Utility functions for formatting
├── main.py                          # Master execution pipeline
├── generate_report.py              # Quarto report generation script
├── StudentID_A1.qmd                 # Quarto markdown source file
├── StudentID_A1.html               # Compiled HTML report
├── StudentID_A1.pdf                 # Compiled PDF report
└── StudentID_A1_GenAI_Disclosure.docx  # GenAI disclosure form
```

### 2.2 Technology Stack

**Core Libraries:**
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.2.0` - Machine learning utilities (train/test split)
- `statsmodels>=0.14.0` - Statistical modeling and econometrics
- `scipy>=1.10.0` - Scientific computing

**Visualization and Reporting:**
- `matplotlib>=3.7.0` - Plotting library
- `seaborn>=0.12.0` - Statistical visualization
- `tabulate>=0.9.0` - Table formatting
- `jinja2>=3.1.0` - Template engine
- `quarto-cli` - Scientific and technical publishing system

**Development Environment:**
- Python 3.9+
- Virtual environment (venv)
- Windsurf IDE with Cascade AI Agent

---

## 3. Methodology

### 3.1 Data Preprocessing Pipeline

#### 3.1.1 Feature Engineering
The `data_loader.py` module implements the following transformations:

1. **Target Variable Creation:**
   ```python
   df['target'] = (df['y'] == 'yes').astype(int)
   ```
   Converts binary response 'yes'/'no' to numeric 1/0.

2. **Age Standardization:**
   ```python
   df['age10'] = (df['age'] - 40) / 10.0
   ```
   Centers age at 40 years and scales by decades. This improves interpretability: coefficient represents effect per 10-year increase from age 40.

3. **Balance Normalization:**
   ```python
   df['balance1000'] = df['balance'] / 1000.0
   ```
   Scales balance by thousands of euros for numerical stability.

4. **Prior Contact Status Construction:**
   ```python
   def get_prior_status(row):
       if row['pdays'] == -1:
           return 'not previously contacted'
       elif row['poutcome'] == 'success':
           return 'previous success'
       elif row['poutcome'] == 'failure':
           return 'previous failure'
       else:
           return 'previous other/unknown'
   ```
   Creates categorical variable summarizing prior campaign outcomes.

#### 3.1.2 Categorical Variable Encoding
All categorical variables are encoded with explicit reference categories:

- **housing:** ['no', 'yes'] - Reference: 'no'
- **loan:** ['no', 'yes'] - Reference: 'no'
- **contact:** ['telephone', 'cellular', 'unknown'] - Reference: 'telephone'
- **prior_status:** ['not previously contacted', 'previous success', 'previous failure', 'previous other/unknown'] - Reference: 'not previously contacted'

This ensures consistent interpretation across models and reproducibility.

#### 3.1.3 Train/Test Split
```python
train, test = train_test_split(
    df, test_size=0.20, random_state=student_id, stratify=df['target']
)
```

**Key Features:**
- **Stratified Sampling:** Maintains class distribution in both splits
- **80/20 Split:** 36,168 training observations, 9,043 test observations
- **Reproducibility:** Student ID serves as random seed
- **No Data Leakage:** Strict separation between training and test phases

### 3.2 Binary Response Models

#### 3.2.1 Model Specification
All binary models use the same specification:

```
target ~ age10 + balance1000 + C(housing) + C(loan) + C(contact) + previous + C(prior_status)
```

**Variables:**
- `age10`: Continuous, centered age
- `balance1000`: Continuous, scaled balance
- `C(housing)`: Categorical, housing loan status
- `C(loan)`: Categorical, personal loan status
- `C(contact)`: Categorical, contact method
- `previous`: Continuous, number of prior contacts
- `C(prior_status)`: Categorical, prior campaign outcome

#### 3.2.2 Model Types

**1. Linear Probability Model (LPM):**
```python
lpm = smf.ols(FORMULA, data=train_df).fit()
```
- Uses ordinary least squares (OLS)
- Assumes linear relationship between predictors and probability
- Coefficients represent marginal effects directly
- Limitation: Can predict probabilities outside [0,1] range

**2. Logit Model:**
```python
logit = smf.logit(FORMULA, data=train_df).fit(disp=False)
```
- Uses logistic function to constrain predictions to [0,1]
- Assumes logistic distribution of error terms
- Coefficients represent log-odds changes
- Marginal effects require additional calculation

**3. Probit Model:**
```python
probit = smf.probit(FORMULA, data=train_df).fit(disp=False)
```
- Uses cumulative normal distribution (probit link)
- Assumes normal distribution of error terms
- Coefficients represent changes in latent variable
- Similar to logit but with different distributional assumptions

#### 3.2.3 Model Comparison
Models are compared using:
- **Akaike Information Criterion (AIC):** Measures model fit with penalty for complexity
- **Coefficient Significance:** p-values for individual predictors
- **Magnitude and Direction:** Consistency across model types

### 3.3 Out-of-Sample Evaluation

#### 3.3.1 Evaluation Metrics
The `evaluate_test_set()` function computes:

1. **Brier Score:**
   ```python
   brier = brier_score_loss(actuals, preds)
   ```
   - Mean squared error of predicted probabilities
   - Range: [0, 0.25] for binary classification
   - Lower values indicate better calibration

2. **Confusion Matrix:**
   ```
                Predicted
                No    Yes
   Actual No   TN    FP
         Yes  FN    TP
   ```

3. **Sensitivity (True Positive Rate):**
   ```python
   sens = tp / (tp + fn)
   ```
   - Proportion of actual positives correctly identified
   - Also called recall or hit rate

4. **Specificity (True Negative Rate):**
   ```python
   spec = tn / (tn + fp)
   ```
   - Proportion of actual negatives correctly identified

5. **Balanced Accuracy:**
   ```python
   bal_acc = (sens + spec) / 2.0
   ```
   - Average of sensitivity and specificity
   - Useful for imbalanced datasets

6. **Observed Event Rate:**
   ```python
   obs_event_rate = actuals.mean()
   ```
   - Baseline prevalence of positive class

7. **Mean Predicted Probability:**
   ```python
   mean_pred_prob = preds.mean()
   ```
   - Average predicted probability across test set

#### 3.3.2 Evaluation Protocol
- **Single Evaluation:** Test set evaluated exactly once
- **No Threshold Tuning:** Default 0.50 threshold used
- **No Refitting:** Models trained on training data only
- **Strict Separation:** Prevents data leakage and overfitting

### 3.4 Ordered Response Models

#### 3.4.1 Model Specification
Ordered models use the full dataset (45,211 observations) with `response_level` as the ordered outcome.

**Design Matrix:**
```python
X = pd.get_dummies(
    df[['age10', 'balance1000', 'housing', 'loan', 'contact', 'previous', 'prior_status']], 
    drop_first=True
).astype(float)
```

**Key Features:**
- **No Intercept:** Cutpoints estimated separately by OrderedModel
- **Dummy Variables:** All categorical variables one-hot encoded
- **Float Conversion:** Ensures numerical compatibility with statsmodels

#### 3.4.2 Model Types

**1. Ordered Logit:**
```python
ologit = OrderedModel(y, X, distr='logit').fit(method='bfgs', disp=False)
```
- Assumes logistic distribution for latent variable
- Cumulative logit model (proportional odds assumption)
- Coefficients represent effect on log-odds of higher categories

**2. Ordered Probit:**
```python
oprobit = OrderedModel(y, X, distr='probit').fit(method='bfgs', disp=False)
```
- Assumes normal distribution for latent variable
- Cumulative probit model
- Similar interpretation to ordered logit with different link function

#### 3.4.3 Cutpoint Estimation
Ordered models estimate K-1 cutpoints (thresholds) for K response categories:
- Cutpoint 1: Separates level 0 from levels 1,2
- Cutpoint 2: Separates levels 0,1 from level 2

These cutpoints are automatically estimated during model fitting.

---

## 4. Implementation Details

### 4.1 Module: `src/data_loader.py`

**Purpose:** Load, preprocess, and split the Bank Marketing dataset.

**Key Functions:**
```python
def load_and_prepare_bank_data(filepath: str, student_id: int):
    """
    Loads, prepares, and splits the Bank Marketing dataset.
    
    Args:
        filepath: Path to CSV file
        student_id: Random seed for reproducibility
    
    Returns:
        full_df: Complete dataset with engineered features
        train_df: Training set (80%)
        test_df: Test set (20%)
    """
```

**Workflow:**
1. Load CSV using pandas
2. Create target variable (binary)
3. Engineer features (age10, balance1000, prior_status)
4. Set categorical reference categories
5. Perform stratified train/test split
6. Return three dataframes

### 4.2 Module: `src/binary_models.py`

**Purpose:** Fit binary response models and evaluate on test set.

**Key Functions:**
```python
def fit_binary_models(train_df: pd.DataFrame):
    """Fits LPM, Logit, and Probit models on training data."""
    
def evaluate_test_set(model, test_df: pd.DataFrame, threshold: float = 0.50):
    """Evaluates model on held-out test set."""
```

**Model Formula:**
```python
FORMULA = "target ~ age10 + balance1000 + C(housing) + C(loan) + C(contact) + previous + C(prior_status)"
```

**Output:**
- Dictionary with fitted models: `{'lpm': ..., 'logit': ..., 'probit': ...}`
- Dictionary with evaluation metrics: `{'brier_score': ..., 'sensitivity': ..., ...}`

### 4.3 Module: `src/ordered_models.py`

**Purpose:** Fit ordered response models on full dataset.

**Key Functions:**
```python
def fit_ordered_models(df: pd.DataFrame):
    """Fits Ordered Logit and Ordered Probit models."""
```

**Implementation Notes:**
- Uses `statsmodels.miscmodels.ordinal_model.OrderedModel`
- Design matrix excludes intercept (cutpoints estimated separately)
- BFGS optimization method for maximum likelihood estimation
- Returns dictionary: `{'ologit': ..., 'oprobit': ...}`

### 4.4 Module: `src/utils.py`

**Purpose:** Format model results for reporting.

**Key Functions:**
```python
def format_summary_table(model_results: dict, model_names: list = None):
    """Creates coefficient comparison table across models."""
    
def format_metrics_table(metrics: dict):
    """Formats evaluation metrics into pandas DataFrame."""
```

**Output Format:**
- Coefficient tables with parameter names, coefficients, and p-values
- Metrics tables with metric names and values
- Clean pandas DataFrame format for Quarto integration

### 4.5 Module: `main.py`

**Purpose:** Master execution pipeline orchestrating the entire analysis.

**Workflow:**
```python
def main():
    # 1. Data Loading & Preparation
    full_df, train_df, test_df = load_and_prepare_bank_data(...)
    
    # 2. Binary Model Fitting
    binary_results = fit_binary_models(train_df)
    
    # 3. Coefficient Comparison
    coef_table = format_summary_table(binary_results, ['lpm', 'logit', 'probit'])
    
    # 4. Test Evaluation
    test_metrics = evaluate_test_set(binary_results['logit'], test_df)
    
    # 5. Ordered Model Fitting
    ordered_results = fit_ordered_models(full_df)
    
    # 6. Ordered Model Comparison
    ordered_coef_table = format_summary_table(ordered_results, ['ologit', 'oprobit'])
```

**Student ID Configuration:**
```python
STUDENT_ID = 12345678  # Replace with actual student ID
```

### 4.6 Module: `generate_report.py`

**Purpose:** Generate Quarto markdown report with embedded analysis.

**Key Functions:**
```python
def generate_quarto_report(student_id: int, output_file: str = "StudentID_A1.qmd"):
    """Generates Quarto markdown report template."""
```

**Report Structure:**
1. Introduction - Context and objectives
2. Data Description - Dataset overview and preprocessing
3. Binary Response Models - Model specification, results, comparison
4. Out-of-Sample Evaluation - Test set metrics
5. Ordered Response Models - Ordered model specification and results
6. Conclusion - Summary of findings
7. References - Bibliography

**Quarto Features:**
- Python code chunks with `echo: false` (no code in output)
- Automatic table formatting
- TOC and section numbering
- HTML and PDF output formats

---

## 5. Coding Standards and Constraints

### 5.1 Academic Integrity Rules

**Core Constraints (from `.windsurfrules`):**

1. **No Data Leakage:**
   - Never mix training (80%) and test (20%) samples during model development
   - Binary models trained exclusively on training data
   - Test set evaluated exactly once with no refitting

2. **Single Evaluation:**
   - Question 3 evaluates test set once only
   - No threshold tuning on test outcomes
   - Default 0.50 threshold used consistently

3. **Full Dataset for Ordered Models:**
   - Question 4 uses complete dataset (45,211 observations)
   - No train/test split for ordered response models
   - Consistent with assessment requirements

4. **Reproducibility:**
   - Student ID serves as random seed for all stochastic operations
   - Code runs top-to-bottom reproducibly
   - No random state variations between runs

5. **Reporting Standards:**
   - Output report contains NO raw code
   - Code remains in `src/` and `main.py`
   - Quarto chunks use `echo: false` to hide code

### 5.2 Statistical Modeling Standards

**Binary Models:**
- Always use `statsmodels.formula.api` for model estimation
- Formula interface ensures consistent specification
- `ols()` for LPM, `logit()` for logit, `probit()` for probit

**Ordered Models:**
- Use `statsmodels.miscmodels.ordinal_model.OrderedModel`
- Exclude raw intercept vectors from design matrix
- Cutpoints estimated separately by the model

**Output Formatting:**
- All tabular outputs formatted as pandas DataFrames
- Clean, publication-ready tables
- Consistent decimal places and formatting

### 5.3 Code Quality Standards

**Modularity:**
- Separation of concerns across modules
- Single responsibility per function
- Clear docstrings for all functions

**Error Handling:**
- Type hints for function parameters
- Explicit data type conversions (`.astype()`)
- Validation of input data

**Documentation:**
- Inline comments for complex logic
- Docstrings following Google style
- README with setup instructions

---

## 6. Results Interpretation Guide

### 6.1 Binary Model Coefficients

**Interpretation by Model Type:**

**LPM Coefficients:**
- Direct interpretation as marginal effects
- Example: `age10 = -0.002` means each 10-year increase from age 40 reduces subscription probability by 0.2 percentage points

**Logit Coefficients:**
- Interpret as log-odds changes
- Example: `age10 = -0.019` means each 10-year increase from age 40 reduces log-odds of subscription by 0.019
- Marginal effects: `∂P/∂X = β * P(1-P)` where P is predicted probability

**Probit Coefficients:**
- Interpret as changes in latent variable
- Similar magnitude to logit but scaled differently
- Marginal effects: `∂P/∂X = β * φ(Xβ)` where φ is standard normal PDF

**Categorical Coefficients:**
- Represent difference from reference category
- Example: `C(housing)[T.yes] = -0.638` (logit) means having a housing loan reduces log-odds by 0.638 compared to no housing loan

### 6.2 Model Fit Statistics

**AIC (Akaike Information Criterion):**
- Lower values indicate better fit
- Penalizes model complexity
- Used for model comparison
- Example: Logit AIC = 22,973 vs Probit AIC = 22,996 → Logit preferred

**P-values:**
- Test null hypothesis that coefficient = 0
- Values < 0.05 indicate statistical significance
- Example: `p < 0.001` for housing coefficient → highly significant

### 6.3 Evaluation Metrics

**Brier Score:**
- Measures calibration of predicted probabilities
- Range: [0, 0.25] for binary classification
- Lower is better
- Example: 0.090 indicates good calibration

**Sensitivity:**
- Ability to identify positive cases (subscriptions)
- Range: [0, 1]
- Higher is better
- Example: 0.186 means 18.6% of actual subscriptions correctly identified

**Specificity:**
- Ability to identify negative cases (non-subscriptions)
- Range: [0, 1]
- Higher is better
- Example: 0.986 means 98.6% of non-subscriptions correctly identified

**Balanced Accuracy:**
- Average of sensitivity and specificity
- Useful for imbalanced datasets
- Range: [0, 1]
- Example: 0.586 indicates moderate overall performance

### 6.4 Ordered Model Coefficients

**Interpretation:**
- Coefficients represent effect on log-odds of being in a higher response category
- Positive coefficient: increases likelihood of higher response level
- Negative coefficient: decreases likelihood of higher response level
- Example: `prior_status_previous success = 2.35` strongly increases likelihood of higher response

**Cutpoints:**
- Thresholds between response categories
- Estimated during model fitting
- Used to compute predicted probabilities for each category

---

## 7. Usage Instructions

### 7.1 Initial Setup

**Step 1: Create Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Configure Student ID**
Edit the following files to replace `12345678` with your actual student ID:
- `main.py` (line 7)
- `generate_report.py` (line 164)
- `StudentID_A1.qmd` (lines 3 and 63)

### 7.2 Running the Analysis

**Execute Master Pipeline:**
```bash
python main.py
```

**Expected Output:**
```
--- 1. Data Loading & Preparation ---
Full: 45211 | Train: 36168 | Test: 9043

--- 2. Fitting Binary Response Models ---
Logit AIC: 22973.315138765378
Probit AIC: 22995.789250286954

Coefficient Comparison Table:
[... coefficient table ...]

--- 3. Out-of-Sample Test Evaluation ---
Test Brier Score: 0.0900
Test Balanced Accuracy: 0.5861

Test Set Metrics:
[... metrics table ...]

--- 4. Ordered Response Models ---
Ordered Logit AIC: 35999.03456226227
Ordered Probit AIC: 36105.760643046626

Ordered Model Coefficient Comparison:
[... ordered coefficient table ...]
```

### 7.3 Generating Reports

**Generate Quarto Source:**
```bash
python generate_report.py
```

**Compile to HTML:**
```bash
quarto render StudentID_A1.qmd --to html
```

**Compile to PDF (requires TinyTeX):**
```bash
quarto install tinytex  # First time only
quarto render StudentID_A1.qmd --to pdf
```

### 7.4 Troubleshooting

**Issue: `python: command not found`**
- Solution: Use `python3` instead of `python` on macOS

**Issue: Ordered model dtype error**
- Solution: Ensure `.astype(float)` and `.astype(int)` conversions in `ordered_models.py`

**Issue: PDF compilation fails**
- Solution: Install TinyTeX: `quarto install tinytex`

**Issue: Import errors**
- Solution: Ensure virtual environment is activated and dependencies installed

---

## 8. Submission Checklist

### 8.1 Required Files

**Main Report:**
- [ ] `StudentID_A1.pdf` (or `.html`) - Compiled Quarto report
  - 15-20 pages maximum
  - Contains only interpretation, tables, figures, references
  - NO raw code or code appendices

**Source Code:**
- [ ] `StudentID_A1.qmd` - Quarto source file
- [ ] `main.py` - Master execution pipeline
- [ ] `src/` directory with all modules

**Disclosure Form:**
- [ ] `StudentID_A1_GenAI_Disclosure.docx` - Completed GenAI disclosure form
  - Submit separately
  - Do NOT duplicate AI disclosures inside main report

### 8.2 Pre-Submission Verification

**Code Verification:**
- [ ] Student ID updated in all three files
- [ ] Code runs top-to-bottom without errors
- [ ] Results are reproducible with student ID as seed
- [ ] No data leakage between train and test sets

**Report Verification:**
- [ ] Report contains no raw code
- [ ] All tables are properly formatted
- [ ] Interpretations are clear and accurate
- [ ] References are properly cited

**File Verification:**
- [ ] All required files present
- [ ] File names follow naming convention
- [ ] PDF/HTML renders correctly
- [ ] GenAI disclosure form completed

---

## 9. Extension Possibilities

### 9.1 Potential Enhancements

**Model Extensions:**
- Regularization (L1/L2) for variable selection
- Interaction terms between predictors
- Non-linear transformations (splines, polynomials)
- Ensemble methods (random forests, gradient boosting)

**Evaluation Extensions:**
- ROC curves and AUC
- Precision-Recall curves
- Calibration plots
- Cross-validation for hyperparameter tuning

**Reporting Extensions:**
- Interactive dashboards (Shiny, Dash)
- Automated interpretation (SHAP values)
- Sensitivity analysis
- Scenario analysis

### 9.2 Dataset Extensions

**Additional Datasets:**
- Telco customer churn dataset (included in `data/`)
- Customer lifetime value modeling
- Market segmentation analysis
- A/B testing analysis

---

## 10. References and Resources

### 10.1 Statistical References

**Binary Response Models:**
- Wooldridge, J. M. (2019). *Introductory Econometrics: A Modern Approach* (7th ed.). Cengage Learning.
- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.

**Ordered Response Models:**
- Agresti, A. (2018). *Statistical Methods for the Social Sciences* (5th ed.). Pearson.
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman and Hall.

**Model Evaluation:**
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
- Steyerberg, E. W. (2019). *Clinical Prediction Models* (2nd ed.). Springer.

### 10.2 Technical References

**Python Libraries:**
- pandas Documentation: https://pandas.pydata.org/docs/
- statsmodels Documentation: https://www.statsmodels.org/stable/
- scikit-learn Documentation: https://scikit-learn.org/stable/

**Quarto:**
- Quarto Documentation: https://quarto.org/docs/
- RStudio: https://www.rstudio.com/

**Dataset:**
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/bank+marketing

### 10.3 Course Resources

**ECOM6004 Course Materials:**
- Lecture notes on binary response models
- Lecture notes on ordered response models
- Assessment brief and rubric
- Software setup guide

**University Resources:**
- Curtin University Library
- Statistical Consulting Service
- Academic Integrity guidelines

---

## 11. Appendix

### 11.1 Complete Formula Reference

**Binary Model Formula:**
```
target ~ age10 + balance1000 + C(housing) + C(loan) + C(contact) + previous + C(prior_status)
```

**Variable Definitions:**
- `target`: Binary indicator (1 = yes, 0 = no)
- `age10`: (age - 40) / 10
- `balance1000`: balance / 1000
- `C(housing)`: Categorical housing loan status
- `C(loan)`: Categorical personal loan status
- `C(contact)`: Categorical contact method
- `previous`: Number of prior contacts
- `C(prior_status)`: Categorical prior campaign outcome

### 11.2 Model Output Reference

**Binary Model Summary:**
```
                           Logit Model
==============================================================================
Dep. Variable:                  target   No. Observations:                36168
Model:                          Logit   Df Residuals:                    36158
Method:                           MLE   Df Model:                            9
Date:                [Date]             Pseudo R-squ:                 0.1234
Time:                        [Time]     Log-Likelihood:            -11477.658
converged:                       True   LL-Null:                   -13090.432
Covariance Type:            nonrobust   LLR p-value:                     0.000
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -1.8138      0.048    -37.814      0.000      -1.908      -1.720
C(housing)[T.yes]    -0.6381      0.043    -14.823      0.000      -0.723      -0.553
[...]
==============================================================================
Possibly complete quasi-separation: A fraction 0.21 of observations can be
perfectly predicted. This might indicate that there is complete
quasi-separation. In this case some coefficients will not be identified.
```

**Ordered Model Summary:**
```
                    Ordered Logit Results      
==============================================================================
Dep. Variable:          response_level   No. Observations:                45211
Model:                   OrderedModel   Df Residuals:                    45198
Method:                           MLE   Df Model:                           12
Date:                [Date]             Pseudo R-squ:                 0.0876
Time:                        [Time]     Log-Likelihood:            -17987.517
Covariance Type:            nonrobust   LL-Null:                   -19687.286
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
age10         -0.0109      0.013     -0.785      0.432      -0.037       0.015
balance1000    0.0226      0.004      5.658      0.000       0.015       0.030
[...]
0/1            1.8671      0.043     43.428      0.000       1.783       1.951
1/2           -0.1193      0.018     -6.688      0.000      -0.154      -0.084
==============================================================================
```

### 11.3 Error Code Reference

**Common Errors and Solutions:**

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Pandas data cast to numpy dtype of object` | Categorical data in ordered model | Add `.astype(float)` to design matrix |
| `ModuleNotFoundError: No module named 'statsmodels'` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `FileNotFoundError: data/bank_assessment_sem2_2026.csv` | Data file missing | Ensure CSV files are in `data/` directory |
| `ConvergenceWarning: Maximum likelihood...` | Model convergence issues | Increase iterations or check for separation |

---

## 12. Version History

**Version 1.0 (September 2026)**
- Initial project setup
- Complete implementation of binary and ordered models
- Automated report generation with Quarto
- Full documentation

---

## 13. Contact and Support

**For Technical Issues:**
- Check troubleshooting section (Section 7.4)
- Review error code reference (Section 11.3)
- Consult library documentation (Section 10.2)

**For Academic Questions:**
- Refer to course materials (Section 10.3)
- Consult assessment brief
- Contact course instructor

**For Project-Specific Issues:**
- Review this documentation thoroughly
- Check `.windsurfrules` for coding standards
- Verify student ID configuration

---

**Document Version:** 1.0  
**Last Updated:** September 25, 2026  
**Author:** ECOM6004 Assessment 1 Project Team  
**License:** Educational Use Only
