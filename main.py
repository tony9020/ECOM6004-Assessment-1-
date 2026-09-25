from src.data_loader import load_and_prepare_bank_data
from src.binary_models import fit_binary_models, evaluate_test_set
from src.ordered_models import fit_ordered_models
from src.utils import format_summary_table, format_metrics_table

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
    
    print("\n--- 4. Ordered Response Models ---")
    ordered_results = fit_ordered_models(full_df)
    print("Ordered Logit AIC:", ordered_results['ologit'].aic)
    print("Ordered Probit AIC:", ordered_results['oprobit'].aic)
    
    # Generate ordered model coefficient comparison
    ordered_coef_table = format_summary_table(ordered_results, ['ologit', 'oprobit'])
    print("\nOrdered Model Coefficient Comparison:")
    print(ordered_coef_table.to_string())

if __name__ == "__main__":
    main()
