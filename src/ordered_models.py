import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
import patsy

def fit_ordered_models(df: pd.DataFrame):
    """
    Fits Ordered Logit and Ordered Probit models using the full dataset.
    
    Empirical Setting: Predicting response_level to assess client contact efficiency.
    - response_level: 0=no subscription, 1=subscription after 2+ contacts, 2=subscription on first contact
    Higher values indicate more efficient conversion (fewer contacts needed).
    
    Note: OrderedModel design matrix should NOT include an intercept.
    Uses formula interface to respect categorical reference categories.
    """
    # Use formula to create design matrix without intercept, respecting categorical encoding
    formula = "age10 + balance1000 + C(housing) + C(loan) + C(contact) + previous + C(prior_status)"
    X = patsy.dmatrix(formula, data=df, return_type='dataframe')
    
    y = df['response_level'].astype(int)
    
    ologit = OrderedModel(y, X, distr='logit').fit(method='bfgs', disp=False)
    oprobit = OrderedModel(y, X, distr='probit').fit(method='bfgs', disp=False)
    
    return {'ologit': ologit, 'oprobit': oprobit}

def diagnose_ordered_models(ordered_results: dict):
    """
    Diagnoses ordered logit and probit models.
    Returns diagnostic summary as pandas DataFrame.
    """
    diagnostics = {}
    for model_name, model in ordered_results.items():
        diagnostics[f'{model_name}_aic'] = model.aic
        diagnostics[f'{model_name}_bic'] = model.bic
        diagnostics[f'{model_name}_llf'] = model.llf
        diagnostics[f'{model_name}_nobs'] = model.nobs
        diagnostics[f'{model_name}_converged'] = model.mle_retvals['converged']
    
    return pd.DataFrame([diagnostics]).T
