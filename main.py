import pandas as pd
import sklearn
import statsmodels
import scipy
import patsy
from src.data_loader import load_and_prepare_bank_data
from src.binary_models import fit_binary_models, evaluate_test_set, create_decision_log
from src.ordered_models import fit_ordered_models, diagnose_ordered_models
from src.utils import format_summary_table, format_metrics_table

# IMPORTANT: Replace with your actual numeric Student ID before submission
STUDENT_ID = 12345678

def main():
    print("--- 1. Data Loading & Preparation ---")
    full_df, train_df, test_df, audit_summary = load_and_prepare_bank_data("data/bank_assessment_sem2_2026.csv", STUDENT_ID)
    print(f"Full: {len(full_df)} | Train: {len(train_df)} | Test: {len(test_df)}")
    print("\nData Audit Summary:")
    print(audit_summary.to_string())
    
    print("\n--- 2. Fitting Binary Response Models ---")
    binary_results = fit_binary_models(train_df)
    print("Logit AIC:", binary_results['logit'].aic)
    print("Probit AIC:", binary_results['probit'].aic)
    
    # Generate coefficient comparison table
    coef_table = format_summary_table(binary_results, ['lpm', 'logit', 'probit'])
    print("\nCoefficient Comparison Table:")
    print(coef_table.to_string())
    
    print("\n--- 3. Out-of-Sample Test Evaluation ---")
    test_metrics = evaluate_test_set(binary_results['logit'], test_df)
    print(f"Test Brier Score: {test_metrics['brier_score']:.4f}")
    print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    
    # Format metrics table
    metrics_table = format_metrics_table(test_metrics)
    print("\nTest Set Metrics:")
    print(metrics_table.to_string())
    
    # Create decision log
    decision_log = create_decision_log(test_metrics, 'logit')
    print("\nBinary-Stage Decision Log:")
    print(decision_log.to_string())
    
    print("\n--- 4. Ordered Response Models ---")
    ordered_results = fit_ordered_models(full_df)
    print("Ordered Logit AIC:", ordered_results['ologit'].aic)
    print("Ordered Probit AIC:", ordered_results['oprobit'].aic)
    
    # Generate ordered model coefficient comparison
    ordered_coef_table = format_summary_table(ordered_results, ['ologit', 'oprobit'])
    print("\nOrdered Model Coefficient Comparison:")
    print(ordered_coef_table.to_string())
    
    # Diagnose ordered models
    ordered_diagnostics = diagnose_ordered_models(ordered_results)
    print("\nOrdered Model Diagnostics:")
    print(ordered_diagnostics.to_string())
    
    print("\n--- 5. Reproducibility Record ---")
    reproducibility_record = {
        'student_id': STUDENT_ID,
        'random_seed': STUDENT_ID,
        'train_test_split': '80/20 stratified',
        'binary_models': ['LPM', 'Logit', 'Probit'],
        'ordered_models': ['Ordered Logit', 'Ordered Probit'],
        'binary_formula': 'target ~ age10 + balance1000 + C(housing) + C(loan) + C(contact) + previous + C(prior_status)',
        'ordered_formula': 'response_level ~ age10 + balance1000 + C(housing) + C(loan) + C(contact) + previous + C(prior_status)',
        'test_evaluation_threshold': 0.50,
        'ordered_model_distribution': 'logit, probit',
        'python_version': pd.__version__,
        'pandas_version': pd.__version__,
        'sklearn_version': sklearn.__version__,
        'statsmodels_version': statsmodels.__version__,
        'scipy_version': scipy.__version__,
        'patsy_version': patsy.__version__
    }
    reproducibility_df = pd.DataFrame([reproducibility_record]).T
    print(reproducibility_df.to_string())
    
    # Save reproducibility record to CSV for submission
    reproducibility_df.to_csv('reproducibility_record.csv')
    print("\nReproducibility record saved to 'reproducibility_record.csv'")

if __name__ == "__main__":
    main()
