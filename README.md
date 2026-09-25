# ECOM6004 Assessment 1 - Binary and Ordered Response Models

This project implements the analysis for ECOM6004 Assessment 1 (Semester 2, 2026) using Python.

## Project Structure

```
ECOM6004_Assessment1/
├── .windsurfrules                  # AI agent constraints
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
├── StudentID_A1_GenAI_Disclosure.docx # Required GenAI Disclosure Form
└── README.md                        # This file
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# .\venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Update Student ID

Edit `main.py` and replace `STUDENT_ID = 12345678` with your actual student ID.

Also update the student ID in:
- `generate_report.py` (line 64)
- `StudentID_A1.qmd` (line 4 and line 32)

### 4. Run the Analysis

```bash
python main.py
```

This will:
- Load and prepare the Bank Marketing data
- Fit LPM, Logit, and Probit models on the training set
- Evaluate the selected model on the test set
- Fit Ordered Logit and Ordered Probit models on the full dataset
- Print coefficient comparison tables and evaluation metrics

### 5. Generate Quarto Report

```bash
python generate_report.py
```

This will generate `StudentID_A1.qmd` with the analysis results embedded.

### 6. Compile Quarto Report

To compile the Quarto report to HTML or PDF:

```bash
quarto render StudentID_A1.qmd
```

## Key Features

- **Stratified Train/Test Split**: 80/20 split using student ID as random seed
- **Binary Models**: LPM, Logit, and Probit using `statsmodels.formula.api`
- **Ordered Models**: Ordered Logit and Probit using `statsmodels.miscmodels.ordinal_model.OrderedModel`
- **Evaluation Metrics**: Brier score, sensitivity, specificity, balanced accuracy
- **Reproducibility**: All code runs top-to-bottom with fixed random seed

## Important Notes

- Never mix training and test samples during model development
- Test set is evaluated once with no refitting or threshold tuning
- Ordered models use the full dataset
- The report must contain NO raw code - code remains in `src/` and `main.py`

## Submission Files

1. **StudentID_A1.pdf** (or .html) - Compiled Quarto report (15-20 pages max)
2. **StudentID_A1.qmd** - Quarto source file
3. **StudentID_A1_GenAI_Disclosure.docx** - Completed GenAI disclosure form
