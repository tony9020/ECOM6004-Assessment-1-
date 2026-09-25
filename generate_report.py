"""
Generate Quarto Report for ECOM6004 Assessment 1
This script generates the Quarto markdown file for the assessment report.
"""

import os

def generate_quarto_report(student_id: int, output_file: str = "StudentID_A1.qmd"):
    """
    Generates a Quarto markdown report template.
    
    Args:
        student_id: Student ID number
        output_file: Output filename for the Quarto report
    """
    
    qmd_content = f"""---
title: "ECOM6004 Assessment 1 - Binary and Ordered Response Models"
author: "Student ID: {student_id}"
date: "`r Sys.Date()`"
format:
  html:
    toc: true
    toc-depth: 2
    number-sections: true
  pdf:
    toc: true
    toc-depth: 2
    number-sections: true
---

# Introduction

This report presents the analysis of the Bank Marketing dataset using binary and ordered response models. The analysis follows the ECOM6004 Assessment 1 requirements for Semester 2, 2026.

# Data Description

The Bank Marketing dataset contains `{{{{ len(full_df) }}}}` observations with the following key variables:

- **age**: Age of the client
- **balance**: Average yearly balance in euros
- **housing**: Whether the client has a housing loan
- **loan**: Whether the client has a personal loan
- **contact**: Contact communication type
- **previous**: Number of contacts performed before this campaign
- **y**: Target variable - whether the client subscribed to a term deposit

The dataset was split into training (80%) and test (20%) sets using stratified sampling with the student ID as the random seed.

# Binary Response Models

## Model Specification

Three binary response models were estimated on the training data:

1. Linear Probability Model (LPM)
2. Logit Model
3. Probit Model

All models use the following specification:

`target ~ age10 + balance1000 + C(housing) + C(loan) + C(contact) + previous + C(prior_status)`

Where:
- `age10` = (age - 40) / 10
- `balance1000` = balance / 1000
- `prior_status` is derived from `pdays` and `poutcome`

## Model Results

```{{{{python}}}}
#| echo: false
#| warning: false

from src.data_loader import load_and_prepare_bank_data
from src.binary_models import fit_binary_models
from src.utils import format_summary_table

STUDENT_ID = {student_id}
full_df, train_df, test_df = load_and_prepare_bank_data("data/bank_assessment_sem2_2026.csv", STUDENT_ID)
binary_results = fit_binary_models(train_df)
coef_table = format_summary_table(binary_results, ['lpm', 'logit', 'probit'])
coef_table
```

## Model Comparison

The table above compares the coefficients across the three models. Key observations include:

- [Interpretation of age10 coefficient]
- [Interpretation of balance1000 coefficient]
- [Interpretation of housing coefficient]
- [Interpretation of loan coefficient]
- [Interpretation of contact coefficient]
- [Interpretation of previous coefficient]
- [Interpretation of prior_status coefficients]

## Out-of-Sample Evaluation

The logit model was evaluated on the held-out test set using a 0.50 threshold.

```{{{{python}}}}
#| echo: false
#| warning: false

from src.binary_models import evaluate_test_set
from src.utils import format_metrics_table

test_metrics = evaluate_test_set(binary_results['logit'], test_df)
metrics_table = format_metrics_table(test_metrics)
metrics_table
```

The test set evaluation shows:
- Observed event rate: `{{{{ test_metrics['obs_event_rate']:.4f }}}}`
- Mean predicted probability: `{{{{ test_metrics['mean_pred_prob']:.4f }}}}`
- Brier score: `{{{{ test_metrics['brier_score']:.4f }}}}`
- Sensitivity: `{{{{ test_metrics['sensitivity']:.4f }}}}`
- Specificity: `{{{{ test_metrics['specificity']:.4f }}}}`
- Balanced accuracy: `{{{{ test_metrics['balanced_accuracy']:.4f }}}}`

# Ordered Response Models

## Model Specification

Ordered Logit and Ordered Probit models were estimated on the full dataset using the `response_level` variable as the ordered outcome.

## Model Results

```{{{{python}}}}
#| echo: false
#| warning: false

from src.ordered_models import fit_ordered_models

ordered_results = fit_ordered_models(full_df)
ordered_coef_table = format_summary_table(ordered_results, ['ologit', 'oprobit'])
ordered_coef_table
```

## Interpretation

The ordered model coefficients represent the effect of each predictor on the log-odds of being in a higher response category. Key findings include:

- [Interpretation of key coefficients]
- [Comparison between ordered logit and ordered probit]

# Conclusion

This analysis demonstrates the application of binary and ordered response models to the Bank Marketing dataset. The binary models show [summary of findings], while the ordered models provide insights into [summary of findings].

# References

[Add any references here if applicable]
"""
    
    with open(output_file, 'w') as f:
        f.write(qmd_content)
    
    print(f"Quarto report generated: {output_file}")

if __name__ == "__main__":
    # Replace with your actual Student ID
    STUDENT_ID = 12345678
    generate_quarto_report(STUDENT_ID)
