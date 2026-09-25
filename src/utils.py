import pandas as pd
import numpy as np

def format_summary_table(model_results: dict, model_names: list = None):
    """
    Formats a summary table comparing coefficients from multiple models.
    
    Args:
        model_results: Dictionary of fitted model objects
        model_names: List of model names for the table columns
    
    Returns:
        pandas DataFrame with formatted coefficient comparison
    """
    if model_names is None:
        model_names = list(model_results.keys())
    
    # Get coefficient names from the first model
    first_model = model_results[model_names[0]]
    param_names = first_model.params.index.tolist()
    
    # Build comparison table
    comparison_data = []
    for param in param_names:
        row = {'Parameter': param}
        for name in model_names:
            model = model_results[name]
            row[f'{name}_coef'] = model.params[param]
            row[f'{name}_pval'] = model.pvalues[param]
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df

def format_metrics_table(metrics: dict):
    """
    Formats evaluation metrics into a clean pandas DataFrame.
    
    Args:
        metrics: Dictionary of evaluation metrics
    
    Returns:
        pandas DataFrame with formatted metrics
    """
    # Extract scalar metrics (excluding confusion matrix)
    scalar_metrics = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    
    df = pd.DataFrame([scalar_metrics]).T
    df.columns = ['Value']
    return df
