```python
# Let's inspect the files in detail to see what variables/data structure we have and generate exact boilerplate files
import pandas as pd
df_bank = pd.read_csv('bank_assessment_sem2_2026(1) (2).csv')
print("Bank Data Info:")
print(df_bank.info())


```

```text
Bank Data Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 45211 entries, 0 to 45210
Data columns (total 20 columns):
 #   Column                Non-Null Count  Dtype 
---  ------                --------------  ----- 
 0   age                   45211 non-null  int64 
 1   job                   45211 non-null  object
 2   marital               45211 non-null  object
 3   education             45211 non-null  object
 4   default               45211 non-null  object
 5   balance               45211 non-null  int64 
 6   housing               45211 non-null  object
 7   loan                  45211 non-null  object
 8   contact               45211 non-null  object
 9   day                   45211 non-null  int64 
 10  month                 45211 non-null  object
 11  duration              45211 non-null  int64 
 12  campaign              45211 non-null  int64 
 13  pdays                 45211 non-null  int64 
 14  previous              45211 non-null  int64 
 15  poutcome              45211 non-null  object
 16  y                     45211 non-null  object
 17  .row_id               45211 non-null  int64 
 18  response_level        45211 non-null  int64 
 19  response_level_label  45211 non-null  object
dtypes: int64(9), object(11)
memory usage: 6.9+ MB
None


```

Here is your complete blueprint to set up and execute **ECOM6004 Assessment 1** in **Python** using **Windsurf** and its AI Agent (Cascade).

---

## 1. Project Directory Structure

Set up your repository structure as required by the assignment brief:

```text
ECOM6004_Assessment1/
├── .windsurfrules                  # System rules for Windsurf AI Agent
├── requirements.txt                 # Python dependencies
├── data/                            # Data folder
│   ├── bank_assessment_sem2_2026.csv
│   └── telco_assessment_sem2_2026.csv
├── src/                             # Python source modules
│   ├── __init__.py
│   ├── data_loader.py              # Data prep, features & stratified split
│   ├── binary_models.py            # LPM, Logit, Probit & evaluation
│   ├── ordered_models.py           # Ordered Logit & Probit
│   └── utils.py                    # Formatter & helper functions
├── main.py                          # Master execution pipeline
├── generate_report.py              # Converts results to Quarto/Markdown report
├── StudentID_A1.qmd                 # Quarto source file for HTML/PDF report
└── StudentID_A1_GenAI_Disclosure.docx # Required GenAI Disclosure Form

```

---

## 2. Environment & Dependency Setup

### `requirements.txt`

Create a `requirements.txt` file with the exact dependencies required by the assessment:

```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
statsmodels>=0.14.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
tabulate>=0.9.0
jinja2>=3.1.0
quarto-cli

```

In your terminal inside Windsurf, run:

```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

pip install -r requirements.txt

```

---

## 3. Configuring Windsurf AI Agent (`.windsurfrules`)

To give Cascade full context on assignment constraints, create a file named `.windsurfrules` in your root folder:

```text
# ECOM6004 Assessment 1 Rules & Guidelines

- **Context**: Curtin University ECOM6004 Assessment 1 (Semester 2, 2026).
- **Core Constraints**:
  1. Never mix binary training (80%) and test (20%) samples during model development in Question 2.
  2. Question 3 evaluates the test set *once* with no refitting or threshold tuning on test outcomes.
  3. Question 4 uses the full dataset for ordered logit/probit modeling.
  4. The code MUST run top-to-bottom reproducibly using the student ID as the random seed.
  5. The output report must strictly contain NO raw code or code appendices; code should remain in `src/` and `main.py`.

- **Coding Standard**:
  - Always use `statsmodels.formula.api` for binary models (`ols`, `logit`, `probit`).
  - Use `statsmodels.miscmodels.ordinal_model.OrderedModel` for ordered response models.
  - Exclude raw intercept vectors when fitting `OrderedModel` as cutpoints are estimated separately.
  - Return all tabular outputs formatted cleanly using `pandas` dataframes.

```

---

## 4. Modular Python Source Code

### `src/data_loader.py`

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_prepare_bank_data(filepath: str, student_id: int):
    """
    Loads, prepares, and splits the Bank Marketing dataset according to ECOM6004 requirements.
    """
    df = pd.read_csv(filepath)
    
    # Required transformations
    df['target'] = (df['y'] == 'yes').astype(int)
    df['age10'] = (df['age'] - 40) / 10.0
    df['balance1000'] = df['balance'] / 1000.0
    
    # Construct prior_status
    def get_prior_status(row):
        if row['pdays'] == -1:
            return 'not previously contacted'
        elif row['poutcome'] == 'success':
            return 'previous success'
        elif row['poutcome'] == 'failure':
            return 'previous failure'
        else:
            return 'previous other/unknown'
            
    df['prior_status'] = df.apply(get_prior_status, axis=1)
    
    # Categorical Type Casting and Reference Category Setting
    df['housing'] = pd.Categorical(df['housing'], categories=['no', 'yes'], ordered=False)
    df['loan'] = pd.Categorical(df['loan'], categories=['no', 'yes'], ordered=False)
    df['contact'] = pd.Categorical(df['contact'], categories=['telephone', 'cellular', 'unknown'], ordered=False)
    df['prior_status'] = pd.Categorical(
        df['prior_status'], 
        categories=['not previously contacted', 'previous success', 'previous failure', 'previous other/unknown'],
        ordered=False
    )
    
    # Stratified 80/20 Train/Test Split using Student ID as Seed
    train, test = train_test_split(
        df, test_size=0.20, random_state=student_id, stratify=df['target']
    )
    
    return df, train.copy(), test.copy()

```

### `src/binary_models.py`

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix, brier_score_loss

FORMULA = "target ~ age10 + balance1000 + C(housing) + C(loan) + C(contact) + previous + C(prior_status)"

def fit_binary_models(train_df: pd.DataFrame):
    """Fits Linear Probability Model, Logit, and Probit models on the training sample."""
    lpm = smf.ols(FORMULA, data=train_df).fit()
    logit = smf.logit(FORMULA, data=train_df).fit(disp=False)
    probit = smf.probit(FORMULA, data=train_df).fit(disp=False)
    
    return {'lpm': lpm, 'logit': logit, 'probit': probit}

def evaluate_test_set(model, test_df: pd.DataFrame, threshold: float = 0.50):
    """Evaluates selected binary model once on the held-out test set."""
    preds = model.predict(test_df)
    pred_labels = (preds >= threshold).astype(int)
    actuals = test_df['target']
    
    cm = confusion_matrix(actuals, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    bal_acc = (sens + spec) / 2.0
    brier = brier_score_loss(actuals, preds)
    
    metrics = {
        'obs_event_rate': actuals.mean(),
        'mean_pred_prob': preds.mean(),
        'brier_score': brier,
        'sensitivity': sens,
        'specificity': spec,
        'balanced_accuracy': bal_acc,
        'confusion_matrix': cm
    }
    return metrics

```

### `src/ordered_models.py`

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

def fit_ordered_models(df: pd.DataFrame):
    """
    Fits Ordered Logit and Ordered Probit models using the full dataset.
    Note: OrderedModel design matrix should NOT include an intercept.
    """
    # Create design matrix without intercept
    X = sm.add_constant(pd.get_dummies(
        df[['age10', 'balance1000', 'housing', 'loan', 'contact', 'previous', 'prior_status']], 
        drop_first=True
    )).drop(columns=['const'])
    
    y = df['response_level']
    
    ologit = OrderedModel(y, X, distr='logit').fit(method='bfgs', disp=False)
    oprobit = OrderedModel(y, X, distr='probit').fit(method='bfgs', disp=False)
    
    return {'ologit': ologit, 'oprobit': oprobit}

```

---

## 5. Master Pipeline Execution (`main.py`)

```python
from src.data_loader import load_and_prepare_bank_data
from src.binary_models import fit_binary_models, evaluate_test_set
from src.ordered_models import fit_ordered_models

# Replace with your numeric Student ID
STUDENT_ID = 12345678

def main():
    print("--- 1. Data Loading & Preparation ---")
    full_df, train_df, test_df = load_and_prepare_bank_data("data/bank_assessment_sem2_2026.csv", STUDENT_ID)
    print(f"Full: {len(full_df)} | Train: {len(train_df)} | Test: {len(test_df)}")
    
    print("\n--- 2. Fitting Binary Response Models ---")
    binary_results = fit_binary_models(train_df)
    print("Logit AIC:", binary_results['logit'].aic)
    print("Probit AIC:", binary_results['probit'].aic)
    
    print("\n--- 3. Out-of-Sample Test Evaluation ---")
    test_metrics = evaluate_test_set(binary_results['logit'], test_df)
    print(f"Test Brier Score: {test_metrics['brier_score']:.4f}")
    print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    
    print("\n--- 4. Ordered Response Models ---")
    ordered_results = fit_ordered_models(full_df)
    print("Ordered Logit AIC:", ordered_results['ologit'].aic)
    print("Ordered Probit AIC:", ordered_results['oprobit'].aic)

if __name__ == "__main__":
    main()

```

---

## 6. How to Use Windsurf Cascade AI Agent

When running Windsurf, trigger **Cascade** (`Cmd + I` or `Ctrl + I`) and prompt it with specific tasks:

* **Prompt for Feature Calculation:**
> *"Cascade, execute `main.py` and print out the summary table comparing LPM, Logit, and Probit coefficients side-by-side with p-values."*


* **Prompt for Marginal Effects:**
> *"Cascade, generate code in `src/binary_models.py` to calculate the Average Marginal Effect (AME) for `age10` and the average discrete change for `housing` using our fitted logit model."*


* **Prompt for Quarto Report Compilation:**
> *"Cascade, draft `StudentID_A1.qmd` compiling all computed tables into a clean report format with `echo: false` so no python code appears in the output."*



---

## 7. Submission Checklist & Deliverables

| File | Submission File Name | Format | Guidelines |
| --- | --- | --- | --- |
| **Main Report** | `StudentID_A1.pdf` (or `.html`) | Compiled Document | 15–20 pages maximum. Contains **only** interpretation, tables, figures, and references. **No raw code**.

 |
| **Source Script** | `StudentID_A1.qmd` / `main.py` | Source Code | Standalone file containing the executable python code.

 |
| **Disclosure Form** | `StudentID_A1_GenAI_Disclosure.docx` | Word Document | Complete 3-page disclosure form. Submit separately; do **not** duplicate AI disclosures inside the main report.

 |